"""
Live inference endpoint for the pharma SLM (prod_winner_v2 + SFT).

Deploy:  modal deploy modal_app/serve.py
Gives a stable URL; routes:
  GET  /health             -> model info
  POST /generate {prompt}  -> chat answer (decodes only newly generated tokens)
  POST /mcq {question, options, answer?} -> per-option likelihood, model's pick

CORS open so the static site can call it. GPU L4 (cheap), scales to zero when idle.
"""
import pathlib, sys
import modal

SRC = str(pathlib.Path(__file__).parent.parent / "src")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy==2.1.2", "tokenizers==0.21.0", "fastapi[standard]")
    .add_local_dir(SRC, remote_path="/root/src")
)
app = modal.App("pharma-slm-serve", image=image)
vol = modal.Volume.from_name("pharma-slm-vol")

CKPT = "/vol/runs/prod_winner_v2/best_sft.pt"
TOK = "/vol/tokenizer/tokenizer.json"


@app.function(gpu="L4", volumes={"/vol": vol}, scaledown_window=600, timeout=600)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web():
    import torch, torch.nn.functional as F
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    sys.path.insert(0, "/root/src")
    from config import ModelConfig
    from model import GPT
    from tokenizers import Tokenizer

    vol.reload()
    ck = torch.load(CKPT, map_location="cuda")
    mcfg = ModelConfig(**ck["model_config"])
    model = GPT(mcfg).to("cuda").eval()
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()})
    tok = Tokenizer.from_file(TOK)
    eot = tok.token_to_id("<|endoftext|>")

    @torch.no_grad()
    def score(prompt, continuation):
        p = tok.encode(prompt).ids
        c = tok.encode(continuation).ids
        ids = (p + c)[-mcfg.max_seq_len:]
        x = torch.tensor(ids, dtype=torch.long, device="cuda").unsqueeze(0)
        h = model.drop(model.tok_emb(x[:, :-1]))
        for blk in model.blocks:
            h = blk(h, model.rope_cos, model.rope_sin)
        h = model.norm(h)
        logits = model.lm_head(h)
        lp = F.log_softmax(logits.float(), dim=-1)
        tgt = x[:, 1:]
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
        n = max(len(c), 1)
        return float(tok_lp[-n:].mean().item())

    api = FastAPI()
    api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @api.get("/health")
    def health():
        return {"ok": True, "model": "prod_winner_v2+SFT",
                "params_dim": mcfg.dim, "layers": mcfg.n_layers, "vocab": mcfg.vocab_size}

    @api.post("/generate")
    @torch.no_grad()
    def generate(body: dict):
        prompt = (body.get("prompt") or "").strip()
        mnt = int(body.get("max_new_tokens", 110))
        temp = float(body.get("temperature", 0.7))
        if not prompt:
            return {"answer": ""}
        text = f"<|user|>\n{prompt}<|assistant|>\n"
        ids = tok.encode(text).ids
        x = torch.tensor(ids, dtype=torch.long, device="cuda").unsqueeze(0)
        y = model.generate(x, mnt, temperature=temp, top_k=200, eos_id=eot)
        new = y[0].tolist()[len(ids):]
        ans = tok.decode(new).strip()
        return {"answer": ans}

    @api.post("/mcq")
    def mcq(body: dict):
        import math
        q = (body.get("question") or "").strip()
        opts = body.get("options", [])
        ans = body.get("answer", None)
        prompt = f"Question: {q}\nAnswer:"
        scores = [score(prompt, " " + str(o)) for o in opts]
        pred = int(max(range(len(scores)), key=lambda i: scores[i])) if scores else -1
        m = max(scores) if scores else 0.0
        exps = [math.exp(s - m) for s in scores]
        z = sum(exps) or 1.0
        conf = [e / z for e in exps]
        return {"scores": scores, "pred": pred, "confidence": conf,
                "correct": (None if ans is None else int(pred == ans))}

    return api
