"""
Supervised fine-tuning (instruction tuning) on top of a pretrained checkpoint.

Turns the next-token "completer" into something that answers questions, using the
chat template:  <|user|>\n{prompt}<|assistant|>\n{response}<|endoftext|>
Loss is masked to the RESPONSE tokens only (the single most important SFT detail).
"""
from __future__ import annotations
import os, sys, json, math, argparse, random
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ModelConfig
from model import GPT

IGNORE = -1


def build_examples(jsonl_path, tok, max_len):
    u = tok.token_to_id("<|user|>")
    a = tok.token_to_id("<|assistant|>")
    eot = tok.token_to_id("<|endoftext|>")
    nl = tok.encode("\n").ids
    examples = []
    with open(jsonl_path) as f:
        for line in f:
            ex = json.loads(line)
            prompt, response = ex["prompt"], ex["response"]
            p_ids = [u] + nl + tok.encode(prompt).ids + [a] + nl
            r_ids = tok.encode(response).ids + [eot]
            input_ids = (p_ids + r_ids)[:max_len]
            labels = ([IGNORE] * len(p_ids) + r_ids)[:max_len]
            if len(input_ids) >= 4 and any(l != IGNORE for l in labels[1:]):
                examples.append((input_ids, labels))
    return examples


def collate(batch, pad_id, device):
    maxlen = max(len(x) for x, _ in batch)
    X, Y = [], []
    for ids, labels in batch:
        padn = maxlen - len(ids)
        X.append(ids + [pad_id] * padn)
        Y.append(labels + [IGNORE] * padn)
    x = torch.tensor(X, dtype=torch.long, device=device)
    y = torch.tensor(Y, dtype=torch.long, device=device)
    return x[:, :-1], y[:, 1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--data", required=True, help="jsonl with {prompt,response}")
    ap.add_argument("--tokenizer", default="/vol/tokenizer/tokenizer.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=1024)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" \
        else torch.amp.autocast(device_type="cpu", enabled=False)

    ckpt = torch.load(args.base_ckpt, map_location=device)
    mcfg = ModelConfig(**ckpt["model_config"])
    model = GPT(mcfg).to(device)
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()})
    model.train()

    tok = Tokenizer.from_file(args.tokenizer)
    pad_id = tok.token_to_id("<|pad|>")
    examples = build_examples(args.data, tok, args.max_len)
    print(f"[sft] {len(examples)} examples")

    opt = model.configure_optimizers(0.0, args.lr, (0.9, 0.95), device)
    steps_per_epoch = math.ceil(len(examples) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs

    step = 0
    for epoch in range(args.epochs):
        random.shuffle(examples)
        for i in range(0, len(examples), args.batch_size):
            batch = examples[i:i + args.batch_size]
            x, y = collate(batch, pad_id, device)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * step / total_steps))
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with ctx:
                # GPT.forward returns full logits + masked loss when targets are passed
                # (targets use IGNORE=-1 on prompt tokens, matching the model's ignore_index)
                _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 20 == 0:
                print(f"[sft] epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f} lr {lr:.2e}")
            step += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model": model.state_dict(), "model_config": mcfg.__dict__}, args.out)
    print(f"[sft] saved -> {args.out}")


if __name__ == "__main__":
    main()
