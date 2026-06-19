"""
DDP pretraining loop for the pharma SLM (hand-written, nanoGPT-style).

Run via torchrun (launched by the Modal harness):
    torchrun --standalone --nproc-per-node=N src/train.py --config_file /vol/runs/<name>/config.json

Live metrics are pushed to a modal.Dict (so the orchestrator can watch loss curves
and cancel losers in real time) AND appended to <out_dir>/<name>/metrics.jsonl.
Implements early abandonment: NaN guard, val-loss ceiling, and patience-based plateau kill.
"""
from __future__ import annotations
import os, sys, json, time, math, argparse, signal
from contextlib import nullcontext

import numpy as np
import torch
import torch._dynamo  # module-level so it doesn't shadow `torch` as a local inside main()
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ModelConfig, TrainConfig, merge_config
from model import GPT
from data import MixDataLoader


# ----------------------------- metrics sink ----------------------------------
class Metrics:
    """Writes to modal.Dict (live) + jsonl (durable). Rank-0 only."""
    def __init__(self, run_name, out_dir, enabled=True):
        self.run_name = run_name
        self.enabled = enabled
        self.path = os.path.join(out_dir, "metrics.jsonl")
        self.history = []
        self.mdict = None
        if enabled:
            os.makedirs(out_dir, exist_ok=True)
            dict_name = os.environ.get("MODAL_METRICS_DICT")
            if dict_name:
                try:
                    import modal
                    self.mdict = modal.Dict.from_name(dict_name, create_if_missing=True)
                except Exception as e:
                    print(f"[metrics] modal.Dict unavailable: {e}")

    def log(self, record: dict):
        if not self.enabled:
            return
        self.history.append(record)
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if self.mdict is not None:
            try:
                self.mdict[self.run_name] = {
                    "history": self.history[-200:],
                    "last": record,
                    "updated_at": record.get("wallclock"),
                }
            except Exception as e:
                print(f"[metrics] dict write failed: {e}")

    def set_status(self, status, **extra):
        if not self.enabled:
            return
        rec = {"status": status, **extra}
        with open(os.path.join(os.path.dirname(self.path), "status.json"), "w") as f:
            json.dump(rec, f, indent=2)
        if self.mdict is not None:
            try:
                cur = self.mdict.get(self.run_name, {}) or {}
                cur["status"] = status
                cur["status_detail"] = extra
                self.mdict[self.run_name] = cur
            except Exception:
                pass


# ----------------------------- lr schedule -----------------------------------
def get_lr(step, cfg: TrainConfig):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    if step >= cfg.max_steps:
        return cfg.min_lr
    ratio = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, loaders, cfg, ctx):
    out = {}
    model.eval()
    for split, loader in loaders.items():
        losses = torch.zeros(cfg.eval_steps)
        for k in range(cfg.eval_steps):
            x, y = loader.get_batch()
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_file", type=str, required=True)
    args = ap.parse_args()

    with open(args.config_file) as f:
        raw = json.load(f)
    tcfg = merge_config(TrainConfig, raw.get("train", raw))
    mcfg = merge_config(ModelConfig, raw.get("model", {}))
    mcfg.max_seq_len = max(mcfg.max_seq_len, tcfg.seq_len)

    # ---- DDP setup ----
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        master = rank == 0
    else:
        rank, local_rank, world = 0, 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        master = True

    device_type = "cuda" if "cuda" in device else "cpu"
    torch.manual_seed(tcfg.seed + rank)
    np.random.seed(tcfg.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[tcfg.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    run_dir = os.path.join(tcfg.out_dir, tcfg.name)
    if master:
        os.makedirs(run_dir, exist_ok=True)
    metrics = Metrics(tcfg.name, run_dir, enabled=master)

    # ---- data ----
    train_loader = MixDataLoader(tcfg.data_dir, tcfg.mix, tcfg.seq_len, tcfg.batch_size,
                                 device, split="train", seed=tcfg.seed + rank)
    val_loader = MixDataLoader(tcfg.data_dir, tcfg.mix, tcfg.seq_len, tcfg.batch_size,
                               device, split="val", seed=tcfg.seed + rank)
    # per-source val loaders: mix-agnostic signal (domain-val loss is comparable across runs)
    per_source_val = {}
    for src in tcfg.mix:
        try:
            per_source_val[src] = MixDataLoader(tcfg.data_dir, {src: 1.0}, tcfg.seq_len,
                                                tcfg.batch_size, device, split="val",
                                                seed=tcfg.seed + rank)
        except Exception as e:
            print(f"[train] no per-source val for {src}: {e}")

    # ---- model ----
    model = GPT(mcfg).to(device)
    if master:
        n = model.num_params()
        print(f"[train] model params: {n/1e6:.1f}M  | tokens/step: "
              f"{tcfg.tokens_per_step(world)/1e6:.3f}M  | world={world}")
        metrics.log({"event": "init", "n_params": n, "world": world,
                     "tokens_per_step": tcfg.tokens_per_step(world), "wallclock": 0.0})

    raw_model = model
    if tcfg.compile:
        torch._dynamo.config.suppress_errors = True  # degrade to eager on any compile error
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[local_rank])

    optimizer = raw_model.configure_optimizers(tcfg.weight_decay, tcfg.lr,
                                               (tcfg.beta1, tcfg.beta2), device_type)

    # ---- training loop ----
    best_val = float("inf")
    evals_since_improve = 0
    nan_streak = 0
    tokens_seen = 0
    t0 = time.time()
    step = 0

    def save_ckpt(tag, val_loss):
        if not master:
            return
        ckpt = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": mcfg.__dict__,
            "train_config": tcfg.__dict__,
            "step": step, "val_loss": val_loss, "tokens_seen": tokens_seen,
            "best_val": best_val, "evals_since_improve": evals_since_improve,
        }
        torch.save(ckpt, os.path.join(run_dir, f"{tag}.pt"))

    def finish(status, **extra):
        if master:
            metrics.set_status(status, step=step, tokens_seen=tokens_seen,
                               best_val=best_val, **extra)
            print(f"[train] FINISHED status={status} step={step} best_val={best_val:.4f} {extra}")
        if ddp:
            dist.barrier()
            dist.destroy_process_group()

    # ---- resume from latest.pt if a prior container left one (Modal retry safety) ----
    resume_path = os.path.join(run_dir, "latest.pt")
    if getattr(tcfg, "resume", True) and os.path.exists(resume_path):
        try:
            ck = torch.load(resume_path, map_location=device)
            raw_model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()})
            if "optimizer" in ck:
                optimizer.load_state_dict(ck["optimizer"])
            step = ck.get("step", 0)
            tokens_seen = ck.get("tokens_seen", 0)
            best_val = ck.get("best_val", best_val)
            evals_since_improve = ck.get("evals_since_improve", 0)
            if master:
                print(f"[train] RESUMED from {resume_path} at step {step} (best_val={best_val:.4f})")
        except Exception as e:
            if master:
                print(f"[train] resume failed ({e}); starting fresh")

    model.train()
    while True:
        # stopping criteria
        if step >= tcfg.max_steps:
            save_ckpt("final", best_val); finish("completed", reason="max_steps"); return
        if tcfg.target_tokens and tokens_seen >= tcfg.target_tokens:
            save_ckpt("final", best_val); finish("completed", reason="target_tokens"); return

        # ---- periodic eval ----
        if step % tcfg.eval_interval == 0:
            loaders = {"train": train_loader, "val": val_loader}
            for src, ldr in per_source_val.items():
                loaders[f"val_{src}"] = ldr
            losses = estimate_loss(model, loaders, tcfg, ctx)
            if master:
                dt = time.time() - t0
                rec = {"event": "eval", "step": step, "tokens": tokens_seen,
                       "train_loss": losses["train"], "val_loss": losses["val"],
                       "lr": get_lr(step, tcfg), "wallclock": dt}
                for src in per_source_val:
                    rec[f"val_{src}"] = losses[f"val_{src}"]
                metrics.log(rec)
                ps = " ".join(f"{s}={losses[f'val_{s}']:.3f}" for s in per_source_val)
                print(f"[eval] step {step} train {losses['train']:.4f} val {losses['val']:.4f} "
                      f"[{ps}] toks {tokens_seen/1e9:.3f}B {dt:.0f}s")
                if losses["val"] < best_val - 1e-4:
                    best_val = losses["val"]; evals_since_improve = 0
                    save_ckpt("best", best_val)
                else:
                    evals_since_improve += 1
                if step > 0 and step % tcfg.save_interval == 0:
                    save_ckpt("latest", losses["val"])

            # ---- early abandonment (broadcast decision from rank 0) ----
            abandon = torch.zeros(1, device=device)
            if master and tcfg.abandon_enable and step >= tcfg.abandon_min_steps:
                if best_val > tcfg.abandon_loss_ceiling:
                    abandon[0] = 1; reason = f"val>{tcfg.abandon_loss_ceiling} after {step} steps"
                elif evals_since_improve >= tcfg.abandon_patience:
                    abandon[0] = 1; reason = f"no improvement for {evals_since_improve} evals"
            if ddp:
                dist.broadcast(abandon, src=0)
            if abandon.item() > 0:
                save_ckpt("final", best_val)
                finish("abandoned", reason=reason if master else "")
                return

        # ---- one optimizer step (with grad accumulation) ----
        lr = get_lr(step, tcfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for micro in range(tcfg.grad_accum):
            x, y = train_loader.get_batch()
            if ddp:
                model.require_backward_grad_sync = (micro == tcfg.grad_accum - 1)
            with ctx:
                _, loss = model(x, y)
                loss = loss / tcfg.grad_accum
            loss.backward()
            accum_loss += loss.item()

        if not math.isfinite(accum_loss):
            nan_streak += 1
            if master:
                print(f"[train] non-finite loss at step {step} (streak {nan_streak})")
            if nan_streak >= tcfg.abandon_nan_tolerance:
                finish("abandoned", reason="nan_loss"); return
            optimizer.zero_grad(set_to_none=True)
            step += 1
            continue
        nan_streak = 0

        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()
        tokens_seen += tcfg.tokens_per_step(world)

        if master and step % tcfg.log_interval == 0:
            dt = time.time() - t0
            tok_per_s = tokens_seen / max(dt, 1e-6)
            print(f"[step {step}] loss {accum_loss:.4f} lr {lr:.2e} "
                  f"{tokens_seen/1e9:.3f}B toks {tok_per_s/1e3:.0f}k tok/s")
            metrics.log({"event": "step", "step": step, "train_loss": accum_loss,
                         "lr": lr, "tokens": tokens_seen, "tok_per_s": tok_per_s,
                         "wallclock": dt})
        step += 1


if __name__ == "__main__":
    main()
