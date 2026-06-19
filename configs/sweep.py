"""
Generate the parallel ablation sweep + the production config.

Philosophy (the "research team"): cheap, short runs that vary the few highest-impact
knobs — learning rate / stability, data mix ratio, and one architecture variant — each
run self-abandoning if it diverges or plateaus. The winner is promoted to the full run.

Usage:
    python configs/sweep.py            # writes configs/ablations/*.json + configs/production.json
"""
import os, json, copy

HERE = os.path.dirname(os.path.abspath(__file__))

# ~350M params: dim 1024 x 24 layers x 16 heads (+32k vocab embedding)
BASE_MODEL = dict(vocab_size=32000, dim=1024, n_layers=24, n_heads=16, max_seq_len=2048)

# short ablation: ~0.35B tokens on 1 GPU
ABLATION_TRAIN = dict(
    group="sweep_r1",
    data_dir="/vol/data", tokenizer_path="/vol/tokenizer/tokenizer.json",
    mix={"fineweb_edu": 0.65, "pubmed": 0.35},
    seq_len=1024, batch_size=24, grad_accum=6,
    lr=6e-4, min_lr=6e-5, warmup_steps=150, max_steps=6000,
    target_tokens=350_000_000,
    eval_interval=250, eval_steps=40, log_interval=25, save_interval=2000,
    abandon_enable=True, abandon_min_steps=800, abandon_patience=3,
    abandon_loss_ceiling=8.0, dtype="bfloat16", compile=True,
)


def cfg(name, model_over=None, train_over=None):
    m = copy.deepcopy(BASE_MODEL); m.update(model_over or {})
    t = copy.deepcopy(ABLATION_TRAIN); t.update(train_over or {}); t["name"] = name
    return {"model": m, "train": t}


def ablation_configs():
    runs = []
    # Round 1: learning-rate / stability sweep (mix fixed 0.65/0.35)
    for lr, tag in [(3e-4, "3e4"), (6e-4, "6e4"), (1e-3, "1e3"), (1.5e-3, "15e4")]:
        runs.append(cfg(f"r1_lr{tag}", train_over={"lr": lr}))
    # Round 1: data-mix sweep (lr fixed 6e-4)
    runs.append(cfg("r1_mix50", train_over={"mix": {"fineweb_edu": 0.5, "pubmed": 0.5}}))
    runs.append(cfg("r1_mix80", train_over={"mix": {"fineweb_edu": 0.8, "pubmed": 0.2}}))
    # Round 1: architecture variant — grouped-query attention (4 kv heads)
    runs.append(cfg("r1_gqa", model_over={"n_kv_heads": 4}))
    return runs


def production_config(winner_train_over=None, name="prod_350m"):
    t = dict(
        group="production",
        data_dir="/vol/data", tokenizer_path="/vol/tokenizer/tokenizer.json",
        mix={"fineweb_edu": 0.65, "pubmed": 0.35},
        seq_len=2048, batch_size=16, grad_accum=8,
        lr=6e-4, min_lr=6e-5, warmup_steps=700, max_steps=40000,
        target_tokens=40_000_000_000,
        eval_interval=500, eval_steps=100, log_interval=50, save_interval=2000,
        abandon_enable=True, abandon_min_steps=4000, abandon_patience=8,
        abandon_loss_ceiling=6.0, dtype="bfloat16", compile=True,
    )
    t.update(winner_train_over or {})
    t["name"] = name
    return {"model": copy.deepcopy(BASE_MODEL), "train": t}


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "ablations"), exist_ok=True)
    for c in ablation_configs():
        p = os.path.join(HERE, "ablations", f"{c['train']['name']}.json")
        json.dump(c, open(p, "w"), indent=2)
        print("wrote", p)
    p = os.path.join(HERE, "production.json")
    json.dump(production_config(), open(p, "w"), indent=2)
    print("wrote", p)
