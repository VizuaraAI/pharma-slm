"""
Generate the parallel PRODUCTION fleet: same model + LR (settled by the sweep), varying
only the data mix across the pharma<->fluency dial. We train all of them, SFT + eval each,
and let actual pharma-QA accuracy (+ general fluency) pick the winner.

Usage: python configs/fleet.py   # writes configs/prod_fleet/*.json
"""
import os, json, copy

HERE = os.path.dirname(os.path.abspath(__file__))

BASE_MODEL = dict(vocab_size=32000, dim=1024, n_layers=24, n_heads=16,
                  max_seq_len=2048, dropout=0.05)

BASE_TRAIN = dict(
    group="prod_fleet",
    data_dir="/vol/data", tokenizer_path="/vol/tokenizer/tokenizer.json",
    seq_len=2048, batch_size=16, grad_accum=8,
    lr=1e-3, min_lr=1e-4, warmup_steps=700, max_steps=11500,
    target_tokens=12_000_000_000,                 # ~34x/param; over-trained but bounded repetition
    eval_interval=250, eval_steps=100, log_interval=50, save_interval=1500,
    abandon_enable=True, abandon_min_steps=3000, abandon_patience=15,
    abandon_loss_ceiling=4.0, dtype="bfloat16", compile=True, seed=1337,
)

# domain (pubmed) fraction across the dial
VARIANTS = {"prod_d30": 0.30, "prod_d45": 0.45, "prod_d60": 0.60, "prod_d75": 0.75}


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "prod_fleet"), exist_ok=True)
    for name, d in VARIANTS.items():
        t = copy.deepcopy(BASE_TRAIN)
        t["name"] = name
        t["mix"] = {"fineweb_edu": round(1 - d, 2), "pubmed": round(d, 2)}
        cfg = {"model": copy.deepcopy(BASE_MODEL), "train": t}
        p = os.path.join(HERE, "prod_fleet", f"{name}.json")
        json.dump(cfg, open(p, "w"), indent=2)
        print(f"wrote {p}  domain={d}")
