"""
Configuration objects for the pharma SLM.

Everything an experiment needs is captured in a plain dict that gets merged onto
these dataclass defaults. Plain dicts keep Modal serialization trivial and let the
parallel sweep generate experiments programmatically.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    dim: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: Optional[int] = None      # None => Multi-Head Attention (n_kv_heads = n_heads)
    max_seq_len: int = 2048
    mlp_hidden: Optional[int] = None      # None => derived from dim via mlp_mult / multiple_of
    mlp_mult: float = 8 / 3               # SwiGLU keeps ~2/3 * 4 = 8/3 of the dense FFN size
    multiple_of: int = 256
    rope_base: float = 10000.0
    dropout: float = 0.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    def resolved_mlp_hidden(self) -> int:
        if self.mlp_hidden is not None:
            return self.mlp_hidden
        h = int(self.mlp_mult * self.dim)
        # round up to a multiple of `multiple_of` (kernel-friendly)
        return self.multiple_of * ((h + self.multiple_of - 1) // self.multiple_of)


@dataclass
class TrainConfig:
    # ---- run identity ----
    name: str = "exp"
    group: str = "sweep"

    # ---- data ----
    data_dir: str = "/vol/data"
    tokenizer_path: str = "/vol/tokenizer/tokenizer.json"
    # mixing weights over per-source .bin files (must reference files that exist in data_dir)
    mix: dict = field(default_factory=lambda: {"fineweb_edu": 0.65, "pubmed": 0.35})

    # ---- optimization ----
    seq_len: int = 2048
    batch_size: int = 16          # per-GPU micro-batch (number of sequences)
    grad_accum: int = 8           # micro-steps accumulated before an optimizer step
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 500
    max_steps: int = 20000
    target_tokens: Optional[int] = None   # if set, stop at this many tokens (overrides max_steps as the budget)
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # ---- eval / logging ----
    eval_interval: int = 500
    eval_steps: int = 100
    log_interval: int = 20

    # ---- checkpointing ----
    out_dir: str = "/vol/runs"
    save_interval: int = 2000     # also always saves the best-by-val checkpoint

    # ---- early abandonment (the "kill bad runs" logic) ----
    abandon_enable: bool = True
    abandon_min_steps: int = 1500     # never abandon before this many steps
    abandon_patience: int = 4         # evals with no val improvement before giving up
    abandon_loss_ceiling: float = 12.0  # if val loss is still above this after min_steps, kill
    abandon_nan_tolerance: int = 3    # consecutive non-finite losses => kill immediately

    # ---- runtime ----
    dtype: str = "bfloat16"
    compile: bool = True
    seed: int = 1337
    resume: bool = True   # resume from latest.pt if present (survives Modal container retries)

    def tokens_per_step(self, world_size: int = 1) -> int:
        return self.batch_size * self.grad_accum * self.seq_len * world_size


def merge_config(defaults_cls, overrides: dict):
    """Build a dataclass instance from defaults + an overrides dict (ignores unknown keys with a warning)."""
    base = asdict(defaults_cls())
    unknown = [k for k in overrides if k not in base]
    if unknown:
        print(f"[config] WARNING: ignoring unknown keys for {defaults_cls.__name__}: {unknown}")
    base.update({k: v for k, v in overrides.items() if k in base})
    return defaults_cls(**base)
