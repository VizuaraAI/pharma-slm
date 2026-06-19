"""
Zero-shot evaluation by answer-likelihood.

For a multiple-choice question we score each option by the model's length-normalized
log-probability of that option text (conditioned on the question), and pick the argmax.
This needs NO instruction tuning, so we can track real pharma knowledge throughout
pretraining. Tasks: MedMCQA, MedQA (USMLE), PubMedQA, MMLU medical subsets.
"""
from __future__ import annotations
import os, sys, argparse, json
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ModelConfig
from model import GPT


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    mcfg = ModelConfig(**ckpt["model_config"])
    model = GPT(mcfg).to(device).eval()
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd)
    return model, mcfg


@torch.no_grad()
def score(model, tok, device, prompt, continuation, max_len):
    p_ids = tok.encode(prompt).ids
    c_ids = tok.encode(continuation).ids
    ids = (p_ids + c_ids)[-max_len:]
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    # full forward to get per-position logits (model.forward returns only the last token)
    h = model.drop(model.tok_emb(x[:, :-1]))
    for blk in model.blocks:
        h = blk(h, model.rope_cos, model.rope_sin)
    h = model.norm(h)
    logits = model.lm_head(h)
    logprobs = F.log_softmax(logits.float(), dim=-1)
    targets = x[:, 1:]
    tgt_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
    n_cont = len(c_ids)
    cont_lp = tgt_lp[-n_cont:] if n_cont > 0 else tgt_lp
    return cont_lp.mean().item()


def eval_mcq(model, tok, device, items, max_len, limit=500):
    correct = 0
    n = 0
    for it in items[:limit]:
        scores = [score(model, tok, device, it["prompt"], " " + opt, max_len)
                  for opt in it["options"]]
        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        correct += int(pred == it["answer"])
        n += 1
    return correct / max(n, 1), n


def load_task(task, limit=500):
    from datasets import load_dataset
    items = []
    if task == "medmcqa":
        ds = load_dataset("openlifescienceai/medmcqa", split="validation", streaming=True)
        for ex in ds:
            opts = [ex["opa"], ex["opb"], ex["opc"], ex["opd"]]
            items.append({"prompt": f"Question: {ex['question']}\nAnswer:",
                          "options": opts, "answer": int(ex["cop"])})
            if len(items) >= limit:
                break
    elif task == "medqa":
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
        for ex in ds:
            opts = list(ex["options"].values()) if isinstance(ex["options"], dict) else ex["options"]
            ans_key = ex.get("answer_idx") or ex.get("answer")
            keys = list(ex["options"].keys()) if isinstance(ex["options"], dict) else ["A","B","C","D"]
            ans = keys.index(ans_key) if ans_key in keys else 0
            items.append({"prompt": f"Question: {ex['question']}\nAnswer:",
                          "options": opts, "answer": ans})
            if len(items) >= limit:
                break
    elif task == "pubmedqa":
        ds = load_dataset("bigbio/pubmed_qa", "pubmed_qa_labeled_fold0_source",
                          split="test", trust_remote_code=True)
        labels = ["yes", "no", "maybe"]
        for ex in ds:
            ctx = " ".join(ex["CONTEXTS"]) if isinstance(ex.get("CONTEXTS"), list) else str(ex.get("CONTEXTS",""))
            items.append({"prompt": f"{ctx}\nQuestion: {ex['QUESTION']}\nAnswer:",
                          "options": labels, "answer": labels.index(ex["final_decision"])})
            if len(items) >= limit:
                break
    else:
        raise ValueError(f"unknown task {task}")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="/vol/tokenizer/tokenizer.json")
    ap.add_argument("--tasks", default="medmcqa,pubmedqa")
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mcfg = load_model(args.ckpt, device)
    tok = Tokenizer.from_file(args.tokenizer)

    results = {}
    for task in args.tasks.split(","):
        items = load_task(task.strip(), args.limit)
        acc, n = eval_mcq(model, tok, device, items, mcfg.max_seq_len, args.limit)
        results[task.strip()] = {"accuracy": acc, "n": n}
        print(f"[eval] {task}: acc={acc:.4f} (n={n})")
    print(json.dumps(results))
    return results


if __name__ == "__main__":
    main()
