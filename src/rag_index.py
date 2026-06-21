"""
Build a biomedical retrieval index for RAG.

Streams pharma passages (PubMed abstracts, MedRAG, clinical guidelines, textbooks),
chunks the long ones, embeds them with a biomedical sentence-embedding model, and writes
a FAISS cosine index + the passage texts to the Modal volume. The RAG endpoint then
retrieves top-k passages at query time so the 1.3B model answers *grounded in real sources*.
"""
from __future__ import annotations
import os, sys, json, argparse

PRIMARY_MODEL = "NeuML/pubmedbert-base-embeddings"   # PubMedBERT tuned for embeddings (768-d)
FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# (source_name, max_docs) — pulled from the same registry used for pretraining
RAG_SOURCES = [("pubmed", 220000), ("medrag_pubmed", 60000),
               ("guidelines", 4000), ("med_textbooks", 4000)]


def chunk(text, size=1200, overlap=150):
    text = " ".join(text.split())
    if len(text) <= size:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


def build(out_dir="/vol/rag", model_name=PRIMARY_MODEL, batch=384, max_passages=300000):
    import numpy as np, faiss
    from sentence_transformers import SentenceTransformer
    sys.path.insert(0, "/root/src")
    from sources import iter_texts

    os.makedirs(out_dir, exist_ok=True)
    passages = []
    for name, cap in RAG_SOURCES:
        n0 = len(passages)
        try:
            for txt in iter_texts(name, max_docs=cap):
                t = txt.strip()
                if len(t) < 80:
                    continue
                for c in chunk(t):
                    if len(c) > 80:
                        passages.append({"text": c, "source": name})
                if len(passages) >= max_passages:
                    break
        except Exception as e:
            print(f"[rag] source {name} skipped: {e}")
        print(f"[rag] {name}: +{len(passages)-n0} passages (total {len(passages)})")
        if len(passages) >= max_passages:
            break

    try:
        model = SentenceTransformer(model_name, device="cuda")
    except Exception as e:
        print(f"[rag] {model_name} failed ({e}); using {FALLBACK_MODEL}")
        model_name = FALLBACK_MODEL
        model = SentenceTransformer(model_name, device="cuda")

    texts = [p["text"] for p in passages]
    print(f"[rag] embedding {len(texts)} passages with {model_name} ...")
    emb = model.encode(texts, batch_size=batch, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True).astype("float32")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    json.dump(passages, open(os.path.join(out_dir, "passages.json"), "w"))
    json.dump({"model": model_name, "n": len(passages), "dim": int(emb.shape[1])},
              open(os.path.join(out_dir, "meta.json"), "w"))
    print(f"[rag] DONE: {len(passages)} passages, dim={emb.shape[1]}, model={model_name}")
    return {"n_passages": len(passages), "dim": int(emb.shape[1]), "model": model_name}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/vol/rag")
    ap.add_argument("--max_passages", type=int, default=300000)
    args = ap.parse_args()
    build(args.out, max_passages=args.max_passages)
