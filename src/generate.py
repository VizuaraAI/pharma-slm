"""Inference: load a checkpoint + tokenizer and generate completions."""
from __future__ import annotations
import os, sys, argparse
import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ModelConfig
from model import GPT


def load(ckpt_path, tokenizer_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    mcfg = ModelConfig(**ckpt["model_config"])
    model = GPT(mcfg).to(device).eval()
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd)
    tok = Tokenizer.from_file(tokenizer_path)
    return model, tok


def generate(model, tok, device, prompt, max_new_tokens=200, temperature=0.8,
             top_k=200, chat=False):
    if chat:
        prompt = f"<|user|>\n{prompt}<|assistant|>\n"
    ids = tok.encode(prompt).ids
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    eos = tok.token_to_id("<|endoftext|>")
    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, eos_id=eos)
    text = tok.decode(y[0].tolist())
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="/vol/tokenizer/tokenizer.json")
    ap.add_argument("--prompt", default="The mechanism of action of ibuprofen is")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--chat", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load(args.ckpt, args.tokenizer, device)
    print(generate(model, tok, device, args.prompt, args.max_new_tokens,
                   args.temperature, chat=args.chat))


if __name__ == "__main__":
    main()
