"""
Scale-up to ~1.3B params. Two artifacts:

  configs/scale_1b/*.json   — a short LEARNING-RATE mini-ablation at 1.3B scale
                              (our 1e-3 was tuned for 350M; bigger models usually want lower LR)
  configs/prod_1b.json      — the full 1.3B production config (LR filled in after the ablation)

~1.3B = dim 2048 · 24 layers · 16 heads (head_dim 128) + 32k tied embedding.

Usage: python configs/scale_1b.py
"""
import os, json, copy

HERE = os.path.dirname(os.path.abspath(__file__))

BASE_MODEL_1B = dict(vocab_size=32000, dim=2048, n_layers=24, n_heads=16,
                     max_seq_len=2048, dropout=0.05)

# short LR-screening run: ~0.4B tokens on 4x H100, seq 1024 (fast)
ABLATION_TRAIN = dict(
    group="scale1b", data_dir="/vol/data", tokenizer_path="/vol/tokenizer/tokenizer.json",
    mix={"fineweb_edu": 0.6, "pubmed": 0.25, "pmc_commercial": 0.15},
    seq_len=1024, batch_size=12, grad_accum=8,
    min_lr=1e-5, warmup_steps=120, max_steps=4000, target_tokens=400_000_000,
    eval_interval=200, eval_steps=40, log_interval=25, save_interval=4000,
    abandon_enable=True, abandon_min_steps=600, abandon_patience=4,
    abandon_loss_ceiling=8.0, dtype="bfloat16", compile=True, seed=1337,
)


def ablation_configs():
    runs = []
    for lr, tag in [(3e-4, "3e4"), (5e-4, "5e4"), (8e-4, "8e4")]:
        t = copy.deepcopy(ABLATION_TRAIN); t["name"] = f"scale1b_lr{tag}"; t["lr"] = lr
        runs.append({"model": copy.deepcopy(BASE_MODEL_1B), "train": t})
    return runs


def production_1b(lr=5e-4, target_tokens=45_000_000_000, name="prod_1b"):
    t = dict(
        group="prod_1b", data_dir="/vol/data", tokenizer_path="/vol/tokenizer/tokenizer.json",
        mix={"fineweb_edu": 0.55, "pubmed": 0.20, "pmc_commercial": 0.20,
             "medrag_pubmed": 0.03, "guidelines": 0.015, "med_textbooks": 0.005},
        seq_len=2048, batch_size=10, grad_accum=12,
        lr=lr, min_lr=lr / 10, warmup_steps=1000, max_steps=22000, target_tokens=target_tokens,
        eval_interval=400, eval_steps=100, log_interval=50, save_interval=1000,
        abandon_enable=True, abandon_min_steps=4000, abandon_patience=15,
        abandon_loss_ceiling=4.0, dtype="bfloat16", compile=True, seed=1337, name=name,
    )
    return {"model": copy.deepcopy(BASE_MODEL_1B), "train": t}


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "scale_1b"), exist_ok=True)
    for c in ablation_configs():
        p = os.path.join(HERE, "scale_1b", f"{c['train']['name']}.json")
        json.dump(c, open(p, "w"), indent=2); print("wrote", p)
    json.dump(production_1b(), open(os.path.join(HERE, "prod_1b.json"), "w"), indent=2)
    print("wrote", os.path.join(HERE, "prod_1b.json"), "(LR is a placeholder until the ablation finishes)")
