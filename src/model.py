"""
A modern, compact decoder-only Transformer ("modern nanoGPT").

Upgrades over the vanilla GPT-2 blocks in the TinyStories notebook:
  - RoPE (rotary position embeddings) instead of learned absolute positions
  - RMSNorm instead of LayerNorm (pre-norm)
  - SwiGLU feed-forward instead of GELU MLP
  - FlashAttention via F.scaled_dot_product_attention
  - Optional Grouped-Query Attention (n_kv_heads < n_heads)
  - Weight tying between input embedding and output projection

Everything is plain PyTorch and intentionally readable — this is a teaching artifact.
"""
from __future__ import annotations
import math
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # compute in fp32 for stability, cast back
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def precompute_rope(head_dim: int, max_seq_len: int, base: float = 10000.0):
    """Returns cos, sin of shape (max_seq_len, head_dim) using the Llama/rotate_half convention."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (head_dim/2,)
    t = torch.arange(max_seq_len).float()                                          # (T,)
    freqs = torch.outer(t, inv_freq)                                               # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)                                        # (T, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    # q, k: (B, n_head, T, head_dim); cos/sin: (T, head_dim)
    T = q.size(-2)
    cos = cos[:T].unsqueeze(0).unsqueeze(0)   # (1,1,T,hd)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads or cfg.n_heads
        assert cfg.dim % cfg.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0
        self.head_dim = cfg.dim // cfg.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(cfg.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, cfg.dim, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        if self.n_rep > 1:  # GQA: expand kv heads to match q heads
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.resolved_mlp_hidden()
        self.w1 = nn.Linear(cfg.dim, hidden, bias=False)   # gate
        self.w3 = nn.Linear(cfg.dim, hidden, bias=False)   # value
        self.w2 = nn.Linear(hidden, cfg.dim, bias=False)   # down
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        head_dim = cfg.dim // cfg.n_heads
        cos, sin = precompute_rope(head_dim, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scaled init on residual projections (GPT-2 trick)
        for pn, p in self.named_parameters():
            if pn.endswith("wo.weight") or pn.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, f"seq len {T} > max {self.cfg.max_seq_len}"
        x = self.drop(self.tok_emb(idx))
        cos, sin = self.rope_cos, self.rope_sin
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])  # only last position at inference
            return logits, None

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.cfg.tie_embeddings:
            n -= self.tok_emb.weight.numel()
        return n

    def configure_optimizers(self, weight_decay, lr, betas, device_type):
        # 2D params (matmuls, embeddings) get weight decay; 1D params (norms, biases) do not.
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = "fused" in inspect.signature(torch.optim.AdamW).parameters and device_type == "cuda"
        opt = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8,
                                **({"fused": True} if fused else {}))
        return opt

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=200, eos_id=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.max_seq_len else idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if eos_id is not None and (idx_next == eos_id).all():
                break
        return idx
