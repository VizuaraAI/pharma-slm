"""
Modal harness for the pharma SLM.

Functions:
  build_tokenizer  - CPU: train the 32k BPE on a blend of sources -> /vol/tokenizer
  prepare_one      - CPU: tokenize one source -> /vol/data/<source>.{train,val}.bin
  train            - 1x H100: pretraining run (used for the parallel ablation sweep)
  train_big        - 8x H100: the full production run
  inspect          - utility: ls the volume / read run status

Local entrypoints (run with `modal run modal_app/app.py::<name>`):
  smoke            - end-to-end mini test (tiny tokenizer + tiny data + 60 steps)
  setup_tokenizer  - build the real tokenizer
  prepare_all      - tokenize all sources in parallel
  launch           - launch one training run from a config json
  ls               - list volume contents
"""
import os, sys, json, subprocess, pathlib, time, glob
import modal

APP_NAME = "pharma-slm"
VOL_NAME = "pharma-slm-vol"
METRICS_DICT = "pharma-slm-metrics"

SRC = str(pathlib.Path(__file__).parent.parent / "src")
REPO = str(pathlib.Path(__file__).parent.parent)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "numpy==2.1.2",
        "datasets==3.2.0",
        "tokenizers==0.21.0",
        "huggingface_hub==0.27.0",
        "tqdm==4.67.1",
        "zstandard",
        "modal==1.4.1",   # so the torchrun subprocess can write live metrics to modal.Dict
    )
    .env({"PYTHONPATH": "/root/src", "HF_HOME": "/tmp/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(SRC, remote_path="/root/src")
)

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
metrics = modal.Dict.from_name(METRICS_DICT, create_if_missing=True)
VOLUMES = {"/vol": vol}


# ----------------------------- data prep -------------------------------------
@app.function(cpu=8.0, memory=32768, timeout=24 * 3600, volumes=VOLUMES)
def build_tokenizer(general_docs=1_200_000, domain_docs=1_200_000, vocab_size=32000,
                    out="/vol/tokenizer/tokenizer.json", only_sources=None):
    sys.path.insert(0, "/root/src")
    from tokenizer_train import train_tokenizer
    train_tokenizer(out, vocab_size, general_docs, domain_docs, only_sources)
    vol.commit()
    return out


@app.function(cpu=8.0, memory=32768, timeout=24 * 3600, volumes=VOLUMES)
def prepare_shard(source, files, shard_id, tokenizer_path, budget,
                  val_tokens, data_dir="/vol/data"):
    sys.path.insert(0, "/root/src")
    from prepare_data import tokenize_files
    train_path = f"{data_dir}/shards/{source}/{shard_id:04d}.train.bin"
    val_path = f"{data_dir}/shards/{source}/{shard_id:04d}.val.bin"
    try:
        res = tokenize_files(files, source, tokenizer_path, train_path, val_path,
                             token_budget=budget, val_tokens=val_tokens)
        res["shard"] = shard_id
    except Exception as e:
        print(f"[prepare_shard] {source} shard {shard_id} FAILED: {e}")
        res = {"source": source, "shard": shard_id, "error": str(e)}
    vol.commit()
    return res


@app.function(volumes=VOLUMES, timeout=2 * 3600)
def finalize(source, data_dir="/vol/data"):
    sys.path.insert(0, "/root/src")
    from prepare_data import finalize_source
    vol.reload()  # warm containers may hold a stale mount; see all committed shards
    r = finalize_source(source, data_dir)
    vol.commit()
    return r


@app.function(timeout=900)
def list_files(source):
    sys.path.insert(0, "/root/src")
    from sources import list_parquet_files
    files = list_parquet_files(source)
    return {"source": source, "n_files": len(files), "sample": files[:3]}


# ----------------------------- training --------------------------------------
def _launch_training(config: dict):
    """Runs inside a GPU container: write config, launch torchrun, commit checkpoints."""
    import torch
    vol.reload()  # see freshly-prepared data / tokenizer committed by other containers
    n_gpu = torch.cuda.device_count()
    name = config["train"]["name"]
    run_dir = f"/vol/runs/{name}"
    os.makedirs(run_dir, exist_ok=True)
    cfg_path = f"{run_dir}/config.json"
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    vol.commit()

    env = dict(os.environ)
    env["MODAL_METRICS_DICT"] = METRICS_DICT
    env["PYTHONPATH"] = "/root/src"

    cmd = ["python", "-m", "torch.distributed.run", "--standalone",
           f"--nproc-per-node={n_gpu}", "/root/src/train.py", "--config_file", cfg_path]
    print(f"[train] launching on {n_gpu} GPU(s): {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env)
    vol.commit()
    status_path = f"{run_dir}/status.json"
    status = json.load(open(status_path)) if os.path.exists(status_path) else {"status": "unknown"}
    status["returncode"] = proc.returncode
    return status


@app.function(gpu="H100:1", timeout=24 * 3600, volumes=VOLUMES)
def train(config: dict):
    return _launch_training(config)


@app.function(gpu="H100:8", timeout=24 * 3600, volumes=VOLUMES)
def train_big(config: dict):
    return _launch_training(config)


@app.function(gpu="H100:4", timeout=24 * 3600, volumes=VOLUMES)
def train4(config: dict):
    return _launch_training(config)


# ----------------------------- utilities -------------------------------------
@app.function(timeout=1800)
def probe_hf(hf_id, config=None, text_key=("text", "content", "contents", "clean_text", "abstract"), n=400):
    """Probe an arbitrary HF dataset for streamability, schema, throughput, avg size."""
    import time as _t
    from datasets import load_dataset
    kw = dict(split="train", streaming=True)
    if config:
        kw["name"] = config
    ds = None
    for trc in (False, True):
        try:
            ds = load_dataset(hf_id, trust_remote_code=trc, **kw) if trc else load_dataset(hf_id, **kw)
            break
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
    if ds is None:
        return {"hf_id": hf_id, "ok": False, "error": err[:200]}
    keys = ("text", "content", "contents", "clean_text", "abstract") if text_key is None else text_key
    t0 = _t.time(); chars = 0; cnt = 0; sample = None; schema = None
    try:
        for ex in ds:
            if schema is None:
                schema = list(ex.keys())
            v = None
            for k in keys:
                if k in ex and isinstance(ex[k], str) and len(ex[k]) > 10:
                    v = ex[k]; break
            if v is None:
                for kk, vv in ex.items():
                    if isinstance(vv, str) and len(vv) > 20:
                        v = vv; break
            if v:
                chars += len(v); cnt += 1
                if sample is None:
                    sample = v[:160]
            if cnt >= n:
                break
    except Exception as e:
        return {"hf_id": hf_id, "ok": False, "error": f"iter: {type(e).__name__}: {e}"[:200], "schema": schema}
    dt = _t.time() - t0
    return {"hf_id": hf_id, "ok": True, "schema": schema, "docs": cnt, "sec": round(dt, 1),
            "avg_chars": chars // max(cnt, 1), "est_tok_per_doc": chars // max(cnt, 1) // 4,
            "sample": sample}


@app.local_entrypoint()
def probe_bio():
    cands = [
        ("MedRAG/pubmed", None),
        ("MedRAG/textbooks", None),
        ("epfl-llm/guidelines", None),
        ("ncbi/pubmed", None),
        ("TaylorAI/pubmed_commercial", None),
    ]
    for r in probe_hf.starmap(cands, return_exceptions=True):
        if isinstance(r, dict):
            print({k: r.get(k) for k in ("hf_id", "ok", "schema", "docs", "sec", "avg_chars", "est_tok_per_doc", "error")})
            if r.get("sample"):
                print("   sample:", r["sample"])
        else:
            print("ERR", r)


@app.function(timeout=1200)
def probe_source(name, n=200):
    """Time streaming the first n docs of a source; return throughput + a sample."""
    import time as _t
    sys.path.insert(0, "/root/src")
    from sources import iter_texts, SOURCES
    res = {"name": name, "spec": {k: SOURCES[name].get(k) for k in ("hf_id", "config")}}
    try:
        t0 = _t.time()
        it = iter_texts(name, max_docs=n)
        first = next(it)
        t_first = _t.time() - t0
        cnt = 1
        for _ in it:
            cnt += 1
        dt = _t.time() - t0
        res.update(ok=True, n=cnt, t_first=round(t_first, 1), t_total=round(dt, 1),
                   docs_per_s=round(cnt / max(dt, 1e-6), 1), sample=first[:200])
    except Exception as e:
        res.update(ok=False, error=f"{type(e).__name__}: {e}")
    return res


@app.local_entrypoint()
def probe():
    names = ["fineweb_edu", "pubmed", "pubmed25", "drug_labels", "pmc"]
    for r in probe_source.map(names, kwargs={"n": 200}, return_exceptions=True):
        print(r if not isinstance(r, dict) else
              {k: r.get(k) for k in ("name", "ok", "n", "t_first", "docs_per_s", "error", "spec")})


@app.function(volumes=VOLUMES, timeout=600)
def inspect(path="/vol"):
    out = {}
    for root, dirs, files in os.walk(path):
        depth = root[len(path):].count(os.sep)
        if depth > 2:
            continue
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                out[fp] = os.path.getsize(fp)
            except OSError:
                pass
    return out


# ----------------------------- entrypoints -----------------------------------
@app.local_entrypoint()
def ls():
    files = inspect.remote()
    for p in sorted(files):
        print(f"{files[p]/1e6:10.2f} MB  {p}")


@app.local_entrypoint()
def setup_tokenizer(general_docs: int = 1_200_000, domain_docs: int = 1_200_000,
                    vocab_size: int = 32000):
    print(build_tokenizer.remote(general_docs, domain_docs, vocab_size))


@app.local_entrypoint()
def files(sources: str = None):
    sys.path.insert(0, SRC)
    from sources import SOURCES
    names = sources.split(",") if sources else list(SOURCES)
    for r in list_files.map(names, return_exceptions=True):
        print(r)


@app.function(timeout=24 * 3600, volumes=VOLUMES)
def prepare_all_remote(n_workers=24, tokenizer_path="/vol/tokenizer/tokenizer.json",
                       data_dir="/vol/data", only=None):
    """Server-side driver: spawn shard workers + finalize. Survives client disconnects."""
    sys.path.insert(0, "/root/src")
    from sources import SOURCES, list_parquet_files
    summary = {}
    items = [(s, SOURCES[s]) for s in only] if only else list(SOURCES.items())
    for source, spec in items:
        all_files = list_parquet_files(source)
        nw = max(1, min(n_workers, len(all_files)))
        groups = [all_files[i::nw] for i in range(nw)]          # round-robin file assignment
        budget_per = spec["target_tokens"] // nw
        print(f"=== {source}: {len(all_files)} files -> {nw} shards, "
              f"~{budget_per/1e9:.2f}B tok/shard (target {spec['target_tokens']/1e9:.0f}B) ===")
        args = [(source, groups[i], i, tokenizer_path, budget_per,
                 2_000_000 if i == 0 else 0, data_dir) for i in range(nw)]
        results = list(prepare_shard.starmap(args, return_exceptions=True))
        ok = sum(1 for r in results if isinstance(r, dict) and "error" not in r)
        toks = sum(r.get("train_tokens", 0) for r in results if isinstance(r, dict))
        fin = finalize.remote(source, data_dir)
        summary[source] = {"shards_ok": ok, "n_shards": nw, "train_tokens": toks, "finalize": fin}
        print(f"  {source}: shards ok={ok}/{nw}  train~{toks/1e9:.2f}B  finalize={fin}")
    vol.commit()
    return summary


@app.local_entrypoint()
def prepare_all(n_workers: int = 24, only: str = None):
    only_list = only.split(",") if only else None
    # spawn (fire-and-forget) so the server-side driver survives client disconnects
    call = prepare_all_remote.spawn(n_workers, only=only_list)
    print(f"spawned prepare_all_remote: {call.object_id}")


@app.local_entrypoint()
def smoke():
    """End-to-end mini test: tiny tokenizer -> tiny data -> 60 training steps on 1 GPU."""
    sys.path.insert(0, SRC)
    from sources import list_parquet_files
    print(">> building tiny tokenizer (vocab 8k, few docs)")
    build_tokenizer.remote(general_docs=12000, domain_docs=12000, vocab_size=8000,
                           out="/vol/tokenizer/smoke.json",
                           only_sources=["fineweb_edu", "pubmed"])
    print(">> tokenizing tiny slices (1 file/source)")
    for src in ["fineweb_edu", "pubmed"]:
        f = list_parquet_files(src)[:1]
        print(prepare_shard.remote(src, f, 0, "/vol/tokenizer/smoke.json",
                                   6_000_000, 1_000_000, "/vol/data_smoke"))
        print(finalize.remote(src, "/vol/data_smoke"))
    print(">> tiny training run")
    cfg = {
        "model": {"vocab_size": 8000, "dim": 256, "n_layers": 4, "n_heads": 4,
                  "max_seq_len": 256},
        "train": {"name": "smoke", "tokenizer_path": "/vol/tokenizer/smoke.json",
                  "data_dir": "/vol/data_smoke",
                  "mix": {"fineweb_edu": 0.6, "pubmed": 0.4},
                  "seq_len": 256, "batch_size": 16, "grad_accum": 2,
                  "max_steps": 60, "warmup_steps": 10, "eval_interval": 20,
                  "eval_steps": 10, "log_interval": 5, "compile": False,
                  "abandon_enable": False},
    }
    print(train.remote(cfg))
    print(">> smoke metrics:", metrics.get("smoke"))


@app.local_entrypoint()
def launch(config_file: str, big: bool = False):
    with open(config_file) as f:
        cfg = json.load(f)
    fn = train_big if big else train
    print(fn.remote(cfg))


# ----------------------------- the parallel sweep ----------------------------
def _summarize(name):
    m = metrics.get(name) or {}
    hist = m.get("history", [])
    evals = [h for h in hist if h.get("event") == "eval"]
    status = m.get("status", "running")
    if not evals:
        return dict(name=name, status=status, step=0, val=None, domain=None, gen=None, best_domain=None)
    last = evals[-1]
    bd = min((e["val_pubmed"] for e in evals if e.get("val_pubmed") is not None), default=None)
    return dict(
        name=name, status=status, step=last.get("step"),
        val=round(last["val_loss"], 4) if last.get("val_loss") is not None else None,
        domain=round(last["val_pubmed"], 4) if last.get("val_pubmed") is not None else None,
        gen=round(last["val_fineweb_edu"], 4) if last.get("val_fineweb_edu") is not None else None,
        best_domain=round(bd, 4) if bd is not None else None,
    )


def _print_board(names, t0):
    rows = sorted((_summarize(n) for n in names),
                  key=lambda r: (r["best_domain"] is None, r["best_domain"] or 9e9))
    print(f"\n=== sweep leaderboard  t+{int((time.time()-t0)/60)}m ===")
    for r in rows:
        print(f"  {r['name']:12s} {r['status']:10s} step={str(r['step']):>5} "
              f"val={r['val']} domain={r['domain']} gen={r['gen']} best_domain={r['best_domain']}")
    return rows


@app.local_entrypoint()
def sweep(poll: int = 30, max_wait_min: int = 120):
    """Spawn all ablation configs in parallel, stream a live leaderboard, rank by domain val."""
    cfgs = sorted(glob.glob(f"{REPO}/configs/ablations/*.json"))
    names = []
    for cf in cfgs:
        cfg = json.load(open(cf))
        name = cfg["train"]["name"]; names.append(name)
        try:
            metrics.pop(name)
        except Exception:
            pass
        train.spawn(cfg)
        print(f"spawned {name}  (lr={cfg['train'].get('lr')} mix={cfg['train'].get('mix')})")

    t0 = time.time()
    terminal = set()
    TERMINAL = {"completed", "abandoned", "failed"}
    while len(terminal) < len(names) and (time.time() - t0) < max_wait_min * 60:
        time.sleep(poll)
        rows = _print_board(names, t0)
        for r in rows:
            if r["status"] in TERMINAL:
                terminal.add(r["name"])

    rows = _print_board(names, t0)
    ranked = [r for r in rows if r["best_domain"] is not None]
    winner = ranked[0]["name"] if ranked else None
    json.dump({"results": rows, "winner": winner},
              open(f"{REPO}/configs/sweep_results.json", "w"), indent=2)
    print(f"\n=== WINNER (lowest domain val): {winner} ===")


@app.local_entrypoint()
def status(group: str = "ablations"):
    """One-shot leaderboard from the metrics Dict. group = ablations | prod_fleet."""
    cfgs = sorted(glob.glob(f"{REPO}/configs/{group}/*.json"))
    names = [json.load(open(cf))["train"]["name"] for cf in cfgs]
    _print_board(names, time.time())


@app.local_entrypoint()
def prod_fleet():
    """Spawn the parallel production fleet (different mixes), each on 4x H100, detached."""
    cfgs = sorted(glob.glob(f"{REPO}/configs/prod_fleet/*.json"))
    for cf in cfgs:
        cfg = json.load(open(cf))
        name = cfg["train"]["name"]
        try:
            metrics.pop(name)
        except Exception:
            pass
        train4.spawn(cfg)
        print(f"spawned {name}  mix={cfg['train']['mix']}  target={cfg['train']['target_tokens']/1e9:.0f}B")
    print(f"launched {len(cfgs)} production runs on 4x H100 each")


@app.function(timeout=24 * 3600, volumes=VOLUMES)
def sft_eval_fleet_remote(names, eval_limit=1000, do_sft=True):
    """Server-side: SFT each base checkpoint (parallel), then eval base + SFT on pharma MCQ."""
    vol.reload()
    sft_handles = {}
    if do_sft:
        for n in names:
            sft_handles[n] = sft.spawn(f"/vol/runs/{n}/best.pt", "/vol/sft/train.jsonl",
                                       f"/vol/runs/{n}/best_sft.pt")
        for n, h in sft_handles.items():
            print(f"SFT {n}:", h.get())
    vol.reload()
    eb = {n: evaluate_ckpt.spawn(f"/vol/runs/{n}/best.pt", "medmcqa,pubmedqa", eval_limit) for n in names}
    es = {n: evaluate_ckpt.spawn(f"/vol/runs/{n}/best_sft.pt", "medmcqa,pubmedqa", eval_limit)
          for n in names} if do_sft else {}
    out = {}
    for n in names:
        out[n] = {"base": eb[n].get()}
        if do_sft:
            try:
                out[n]["sft"] = es[n].get()
            except Exception as e:
                out[n]["sft"] = {"error": str(e)}
        print(n, out[n])
    return out


@app.local_entrypoint()
def sft_eval_fleet(eval_limit: int = 1000):
    names = [json.load(open(cf))["train"]["name"]
             for cf in sorted(glob.glob(f"{REPO}/configs/prod_fleet/*.json"))]
    res = sft_eval_fleet_remote.remote(names, eval_limit)
    json.dump(res, open(f"{REPO}/configs/fleet_eval_results.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


@app.local_entrypoint()
def launch_prod(config_file: str = None):
    cf = config_file or f"{REPO}/configs/production.json"
    cfg = json.load(open(cf))
    try:
        metrics.pop(cfg["train"]["name"])
    except Exception:
        pass
    call = train_big.spawn(cfg)   # spawn so the long run survives client disconnects
    print(f"spawned train_big {cfg['train']['name']} on 8x H100: {call.object_id}")


# ----------------------------- SFT / eval / generate -------------------------
@app.function(cpu=8.0, memory=32768, timeout=6 * 3600, volumes=VOLUMES)
def build_sft_data(n_medmcqa=30000, n_pubmedqa=30000, n_general=10000, out="/vol/sft/train.jsonl"):
    sys.path.insert(0, "/root/src")
    from sft_data import build
    r = build(out, n_medmcqa, n_pubmedqa, n_general)
    vol.commit()
    return r


@app.function(gpu="H100:1", timeout=12 * 3600, volumes=VOLUMES)
def sft(base_ckpt, data="/vol/sft/train.jsonl", out=None,
        tokenizer="/vol/tokenizer/tokenizer.json", epochs=3, lr=1.5e-5):
    out = out or base_ckpt.replace(".pt", "_sft.pt")
    vol.reload()
    cmd = ["python", "/root/src/sft.py", "--base_ckpt", base_ckpt, "--data", data,
           "--tokenizer", tokenizer, "--out", out, "--epochs", str(epochs), "--lr", str(lr)]
    env = dict(os.environ); env["PYTHONPATH"] = "/root/src"
    subprocess.run(cmd, env=env, check=True)
    vol.commit()
    return out


@app.function(gpu="H100:1", timeout=2 * 3600, volumes=VOLUMES)
def evaluate_ckpt(ckpt, tasks="medmcqa,pubmedqa", limit=500,
                  tokenizer="/vol/tokenizer/tokenizer.json"):
    sys.path.insert(0, "/root/src")
    from evaluate import load_model, load_task, eval_mcq
    from tokenizers import Tokenizer
    vol.reload()
    device = "cuda"
    model, mcfg = load_model(ckpt, device)
    tok = Tokenizer.from_file(tokenizer)
    res = {}
    for t in tasks.split(","):
        items = load_task(t.strip(), limit)
        acc, n = eval_mcq(model, tok, device, items, mcfg.max_seq_len, limit)
        res[t.strip()] = {"accuracy": round(acc, 4), "n": n}
        print(f"[eval] {t}: {acc:.4f} (n={n})")
    return res


@app.function(gpu="H100:1", timeout=1800, volumes=VOLUMES)
def generate_ckpt(ckpt, prompt, tokenizer="/vol/tokenizer/tokenizer.json",
                  max_new_tokens=200, chat=False, temperature=0.8):
    sys.path.insert(0, "/root/src")
    from generate import load, generate
    vol.reload()
    device = "cuda"
    model, tok = load(ckpt, tokenizer, device)
    return generate(model, tok, device, prompt, max_new_tokens, temperature, chat=chat)


@app.function(gpu="H100:1", timeout=12 * 3600, volumes=VOLUMES)
def finalize_one_remote(name, eval_limit=1000):
    """Server-side: SFT a base ckpt, eval base+SFT on pharma MCQ, gen samples -> eval.json."""
    import json as _j
    sys.path.insert(0, "/root/src")
    vol.reload()
    base = f"/vol/runs/{name}/best.pt"
    sft_out = f"/vol/runs/{name}/best_sft.pt"
    env = dict(os.environ); env["PYTHONPATH"] = "/root/src"
    subprocess.run(["python", "/root/src/sft.py", "--base_ckpt", base, "--data",
                    "/vol/sft/train.jsonl", "--tokenizer", "/vol/tokenizer/tokenizer.json",
                    "--out", sft_out, "--epochs", "3"], env=env, check=True)
    vol.commit()

    from evaluate import load_model, load_task, eval_mcq
    from generate import generate as gen_fn
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("/vol/tokenizer/tokenizer.json")
    res = {"name": name}
    for tag, ckpt in [("base", base), ("sft", sft_out)]:
        model, mcfg = load_model(ckpt, "cuda")
        r = {}
        for t in ["medmcqa", "pubmedqa"]:
            items = load_task(t, eval_limit)
            acc, n = eval_mcq(model, tok, "cuda", items, mcfg.max_seq_len, eval_limit)
            r[t] = round(acc, 4)
        res[tag] = r
        del model
    # sample answers from the SFT model
    model, mcfg = load_model(sft_out, "cuda")
    qs = ["What is the mechanism of action of metformin?",
          "What are the common side effects of warfarin?",
          "What is amoxicillin used to treat?"]
    res["samples"] = {q: gen_fn(model, tok, "cuda", q, max_new_tokens=110, temperature=0.7, chat=True)
                      for q in qs}
    _j.dump(res, open(f"/vol/runs/{name}/eval.json", "w"), indent=2)
    vol.commit()
    print("FINALIZE RESULT:", _j.dumps({k: res[k] for k in ("base", "sft")}))
    return res


@app.local_entrypoint()
def finalize_winner(name: str = "prod_winner_v2"):
    call = finalize_one_remote.spawn(name)
    print(f"spawned finalize for {name}: {call.object_id}  -> results at /vol/runs/{name}/eval.json")


@app.local_entrypoint()
def sft_data(n_medmcqa: int = 30000, n_pubmedqa: int = 30000, n_general: int = 10000):
    print(build_sft_data.remote(n_medmcqa, n_pubmedqa, n_general))


@app.local_entrypoint()
def run_sft(base_ckpt: str, epochs: int = 3, lr: float = 1.5e-5):
    print(sft.remote(base_ckpt, epochs=epochs, lr=lr))


@app.local_entrypoint()
def evaluate(ckpt: str, tasks: str = "medmcqa,pubmedqa", limit: int = 500):
    print(evaluate_ckpt.remote(ckpt, tasks, limit))


@app.local_entrypoint()
def gen(ckpt: str, prompt: str = "The mechanism of action of ibuprofen is",
        max_new_tokens: int = 200, chat: bool = False):
    print(generate_ckpt.remote(ckpt, prompt, max_new_tokens=max_new_tokens, chat=chat))
