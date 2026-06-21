"""
Live inference endpoint for the pharma SLM — now serving the 1.3B (prod_1b) + RAG.

Deploy:  modal deploy modal_app/serve.py
Routes:
  GET  /health
  POST /generate {prompt}                       -> chat answer (1.3B)
  POST /mcq {question, options, answer?}         -> per-option likelihood, model's pick
  POST /rag {question, k?}                        -> retrieval-augmented answer + cited sources

RAG retrieves top-k biomedical passages (FAISS over PubMed/guidelines/textbooks) and has the
1.3B model answer grounded in them — fixing hallucination without scaling the model.
"""
import pathlib, sys
import modal

SRC = str(pathlib.Path(__file__).parent.parent / "src")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy==2.1.2", "tokenizers==0.21.0", "fastapi[standard]",
                 "sentence-transformers==3.3.1", "faiss-cpu==1.9.0")
    .add_local_dir(SRC, remote_path="/root/src")
)
app = modal.App("pharma-slm-serve", image=image)
vol = modal.Volume.from_name("pharma-slm-vol")

CKPT = "/vol/runs/prod_1b/best_sft.pt"     # the 1.3B SFT model
TOK = "/vol/tokenizer/tokenizer.json"
RAG_DIR = "/vol/rag"


@app.function(gpu="L4", volumes={"/vol": vol}, scaledown_window=900, timeout=600)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web():
    import os, json, torch, torch.nn.functional as F
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
    n_params = sum(p.numel() for p in model.parameters())

    # ---- RAG assets (optional; endpoint still works without them) ----
    embedder = faiss_index = passages = None
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        meta = json.load(open(f"{RAG_DIR}/meta.json"))
        embedder = SentenceTransformer(meta["model"], device="cuda")
        faiss_index = faiss.read_index(f"{RAG_DIR}/index.faiss")
        passages = json.load(open(f"{RAG_DIR}/passages.json"))
        print(f"[serve] RAG loaded: {len(passages)} passages, model={meta['model']}")
    except Exception as e:
        print(f"[serve] RAG not available yet: {e}")

    @torch.no_grad()
    def chat(prompt, max_new_tokens=160, temperature=0.6):
        text = f"<|user|>\n{prompt}<|assistant|>\n"
        ids = tok.encode(text).ids[-(mcfg.max_seq_len - max_new_tokens):]
        x = torch.tensor(ids, dtype=torch.long, device="cuda").unsqueeze(0)
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=200, eos_id=eot)
        return tok.decode(y[0].tolist()[len(ids):]).strip()

    @torch.no_grad()
    def score(prompt, continuation):
        p, c = tok.encode(prompt).ids, tok.encode(continuation).ids
        ids = (p + c)[-mcfg.max_seq_len:]
        x = torch.tensor(ids, dtype=torch.long, device="cuda").unsqueeze(0)
        h = model.drop(model.tok_emb(x[:, :-1]))
        for blk in model.blocks:
            h = blk(h, model.rope_cos, model.rope_sin)
        logits = model.lm_head(model.norm(h))
        lp = F.log_softmax(logits.float(), dim=-1)
        tgt_lp = lp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)[0]
        return float(tgt_lp[-max(len(c), 1):].mean().item())

    def retrieve(q, k=4):
        if embedder is None:
            return []
        import numpy as np
        qe = embedder.encode([q], normalize_embeddings=True).astype("float32")
        D, I = faiss_index.search(qe, k)
        return [{"text": passages[i]["text"], "source": passages[i]["source"], "score": float(D[0][j])}
                for j, i in enumerate(I[0]) if i >= 0]

    api = FastAPI()
    api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @api.get("/health")
    def health():
        return {"ok": True, "model": "prod_1b+SFT", "params_M": round(n_params / 1e6, 1),
                "dim": mcfg.dim, "layers": mcfg.n_layers, "rag": embedder is not None,
                "rag_passages": len(passages) if passages else 0}

    @api.post("/generate")
    def generate(body: dict):
        q = (body.get("prompt") or "").strip()
        if not q:
            return {"answer": ""}
        return {"answer": chat(q, int(body.get("max_new_tokens", 160)), float(body.get("temperature", 0.6)))}

    @api.post("/mcq")
    def mcq(body: dict):
        import math
        q, opts = (body.get("question") or "").strip(), body.get("options", [])
        ans = body.get("answer", None)
        prompt = f"Question: {q}\nAnswer:"
        scores = [score(prompt, " " + str(o)) for o in opts]
        pred = int(max(range(len(scores)), key=lambda i: scores[i])) if scores else -1
        m = max(scores) if scores else 0.0
        ex = [math.exp(s - m) for s in scores]; z = sum(ex) or 1.0
        return {"scores": scores, "pred": pred, "confidence": [e / z for e in ex],
                "correct": (None if ans is None else int(pred == ans))}

    @api.post("/rag")
    def rag(body: dict):
        q = (body.get("question") or body.get("prompt") or "").strip()
        k = int(body.get("k", 4))
        if not q:
            return {"answer": "", "sources": []}
        hits = retrieve(q, k)
        if not hits:
            return {"answer": chat(q), "sources": [], "grounded": False}
        ctx = "\n\n".join(f"[{j+1}] {h['text'][:520]}" for j, h in enumerate(hits))
        prompt = (f"Use the following medical sources to answer the question accurately. "
                  f"Base your answer only on these sources.\n\nSources:\n{ctx}\n\nQuestion: {q}")
        ans = chat(prompt, max_new_tokens=180, temperature=0.5)
        return {"answer": ans, "grounded": True,
                "sources": [{"text": h["text"][:300], "source": h["source"], "score": round(h["score"], 3)}
                            for h in hits]}

    return api
