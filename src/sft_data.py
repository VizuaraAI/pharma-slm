"""
Build the SFT (instruction-tuning) dataset as jsonl of {prompt, response}.

Sources (all decontaminated from the eval splits we hold out):
  - MedMCQA *train* split        -> MCQ + correct answer + explanation
  - PubMedQA *artificial* split  -> context+question -> long answer + yes/no/maybe
  - a slice of general instructions (alpaca-cleaned) so it stays a generalist

We hold out MedMCQA *validation*, MedQA *test*, PubMedQA *labeled test* for evaluation.
"""
from __future__ import annotations
import os, json, argparse, random

LETTERS = ["A", "B", "C", "D", "E"]


def medmcqa_examples(limit):
    from datasets import load_dataset
    ds = load_dataset("openlifescienceai/medmcqa", split="train", streaming=True)
    out = []
    for ex in ds:
        opts = [ex["opa"], ex["opb"], ex["opc"], ex["opd"]]
        ci = int(ex["cop"])
        if ci < 0 or ci >= len(opts):
            continue
        opt_block = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(opts))
        prompt = f"{ex['question'].strip()}\n{opt_block}"
        exp = (ex.get("exp") or "").strip()
        resp = f"The correct answer is {LETTERS[ci]}) {opts[ci]}."
        if exp:
            resp += f" {exp}"
        out.append({"prompt": prompt, "response": resp})
        if len(out) >= limit:
            break
    return out


def pubmedqa_examples(limit):
    from datasets import load_dataset
    try:
        ds = load_dataset("bigbio/pubmed_qa", "pubmed_qa_artificial_source",
                          split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"[sft_data] pubmedqa skipped: {e}")
        return []
    out = []
    for ex in ds:
        ctx = ex.get("CONTEXTS")
        ctx = " ".join(ctx) if isinstance(ctx, list) else str(ctx or "")
        q = ex.get("QUESTION", "")
        la = (ex.get("LONG_ANSWER") or "").strip()
        dec = ex.get("final_decision", "")
        if not q or not la:
            continue
        prompt = f"Context: {ctx}\nQuestion: {q}"
        resp = f"{la} Answer: {dec}."
        out.append({"prompt": prompt, "response": resp})
        if len(out) >= limit:
            break
    return out


def general_examples(limit):
    from datasets import load_dataset
    try:
        ds = load_dataset("yahma/alpaca-cleaned", split="train", streaming=True)
    except Exception as e:
        print(f"[sft_data] general skipped: {e}")
        return []
    out = []
    for ex in ds:
        instr, inp, outp = ex.get("instruction", ""), ex.get("input", ""), ex.get("output", "")
        if not instr or not outp:
            continue
        prompt = instr if not inp else f"{instr}\n{inp}"
        out.append({"prompt": prompt, "response": outp})
        if len(out) >= limit:
            break
    return out


def build(out_path, n_medmcqa=30000, n_pubmedqa=30000, n_general=10000, seed=1337):
    data = []
    data += medmcqa_examples(n_medmcqa); print(f"[sft_data] medmcqa: {len(data)}")
    n = len(data); data += pubmedqa_examples(n_pubmedqa); print(f"[sft_data] pubmedqa: {len(data)-n}")
    n = len(data); data += general_examples(n_general); print(f"[sft_data] general: {len(data)-n}")
    random.Random(seed).shuffle(data)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
    print(f"[sft_data] wrote {len(data)} examples -> {out_path}")
    return {"total": len(data), "path": out_path}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/vol/sft/train.jsonl")
    ap.add_argument("--n_medmcqa", type=int, default=30000)
    ap.add_argument("--n_pubmedqa", type=int, default=30000)
    ap.add_argument("--n_general", type=int, default=10000)
    args = ap.parse_args()
    build(args.out, args.n_medmcqa, args.n_pubmedqa, args.n_general)
