# Pharma SLM — a domain-specific Small Language Model, built **from scratch**

A complete, reproducible pipeline that takes you from **raw medical text → a working ~350M-parameter
pharmaceutical language model**, trained entirely from scratch (no borrowed weights) on Modal H100 GPUs.

This repo is the full record of the project: every experiment, every GPU, every cost, and every command
needed to reproduce it end-to-end.

- **Live demo:** _(Vercel link added on deploy)_ — chat with the model + watch it take a medical MCQ exam.
- **Live inference API:** `https://teamvizuara--pharma-slm-serve-web.modal.run` (`/generate`, `/mcq`, `/health`)
- **Final model:** `prod_winner_v2` — 350M params, trained on ~20B tokens, then instruction-tuned (SFT).

> **One-line result:** the model writes **fluent, mostly-accurate pharma prose** (drug uses, side effects,
> mechanisms), but is **near-chance on hard multiple-choice exams** (~0.29 vs 0.25 guessing) — a *capacity*
> limit of a 350M model, not a data/training limit. We proved this: 5× more training + 3× more data did
> **not** move the exam score. See [Results](#results).

---

## Table of contents
1. [What "from scratch" means here](#what-from-scratch-means-here)
2. [Results](#results)
3. [The complete experiment log](#the-complete-experiment-log)
4. [Cost & GPUs consumed](#cost--gpus-consumed)
5. [Architecture](#architecture)
6. [Data — exact sources, sizes, licenses](#data--exact-sources-sizes-licenses)
7. [Reproduce it end-to-end](#reproduce-it-end-to-end)
8. [Repo layout](#repo-layout)
9. [Methodology, decisions & lessons](#methodology-decisions--lessons)
10. [Bugs we hit (and the fixes)](#bugs-we-hit-and-the-fixes)
11. [Limitations & next steps](#limitations--next-steps)

---

## What "from scratch" means here

No pre-trained model is fine-tuned. We build all six stages ourselves:

```
Gather data → Train tokenizer → Define model → Pretrain → Instruction-tune (SFT) → Evaluate
```

This is the same recipe behind GPT-style models, shrunk to a size one person can train, explain, and afford.
It's the path the best small biomedical models (BioGPT-347M, PubMedBERT, BioMedLM) actually took — and going
from scratch is what lets us train a **custom domain tokenizer** on day one (impossible when fine-tuning).

---

## Results

### Final model vs. the first quick model

| Metric | First model (`prod_d60`, ~4B tok) | **Winner (`prod_winner_v2`, 20B tok)** | Read |
|---|---|---|---|
| MedMCQA (4-opt, chance 0.25) | 0.300 | **0.290** | flat — *capacity-bound* |
| PubMedQA (chance ~0.33–0.5) | 0.608 | **0.608** | flat — *capacity-bound* |
| PubMed val-loss ↓ | 2.49 | **2.27** | **better** (fluency) |
| General val-loss ↓ | ~3.1 | **2.79** | **better** (fluency) |
| Open-ended answer quality | shaky | **coherent & mostly accurate** | **the real win** |

**Plain English — what "MCQ capacity-bound at 0.29" means:** on a 4-option medical exam the model is right
~29% of the time; pure guessing is 25%. So on *hard reasoning* questions it's barely above a coin-toss.
"Capacity-bound" = the model is **too small (350M params)** to do the multi-step reasoning those questions
need — and we proved it's a *size* limit because training 5× longer on 3× more data did not raise the score.
The **same model writes good pharma explanations**, because generating fluent text needs far less reasoning
capacity than acing trick MCQs. **It's a useful pharma writer/explainer, not a board-exam solver.** To pass
exams, scale to ~1B+ params (the pipeline already supports it).

### Sample generations (real, from `prod_winner_v2` + SFT)

> **Q: What is the mechanism of action of aspirin?**
> *"…it acts by inhibiting cyclooxygenase, an enzyme that produces prostaglandins and thromboxanes, which
> are responsible for many of the inflammatory effects… aspirin can reduce the risk of certain cancers,
> including cancers of the stomach and esophagus…"* ✅ accurate

> **Q: What are the common side effects of warfarin?**
> *"The most common side effect is bleeding — bruising, bleeding gums… Coumarin anticoagulants can cause
> nausea, vomiting… neurological: dizziness, confusion."* ✅ accurate

### MCQ — right vs. wrong (curated demo set, 3/6 correct)

| Question | Model picked | Correct? |
|---|---|---|
| First-line oral med for type-2 diabetes | Metformin | ✅ |
| Antibiotic class of amoxicillin | Beta-lactam | ✅ |
| ACE-inhibitor cough is caused by accumulation of… | Bradykinin | ✅ |
| Which is a proton pump inhibitor? | Metformin | ❌ (Omeprazole) |
| Which is a beta-blocker? | Omeprazole | ❌ (Atenolol) |
| Warfarin + which antibiotic ↑ bleeding most? | Nitrofurantoin | ❌ (Metronidazole) |

It nails some recall and even one reasoning item, but flips on easy ones — visibly **unreliable**, which is
exactly the near-chance / capacity-bound picture.

---

## The complete experiment log

Every model we trained, in order. All training on [Modal](https://modal.com) (workspace `teamvizuara`).

| # | Experiment | What it answered | Hardware | Tokens | Key outcome |
|---|---|---|---|---|---|
| 0 | **Smoke ×3** (tiny 8k-vocab model) | does the pipeline work end-to-end? | 1×H100 | ~0.0005B | caught 4 bugs → green |
| 1 | **Tokenizer** (32k BPE) | custom vs off-the-shelf | CPU (8-core) | — | fertility **1.57** on pharma text |
| 2 | **Data prep** (shard-parallel) | tokenize the corpus fast | CPU ×24 | ~13B written | `.bin` shards on a Modal Volume |
| 3 | **Ablation sweep** — 7 runs | best LR / mix / arch | 7 × (1×H100) | ~0.35B ea | **LR 1e-3** best; GQA ~free |
| 4 | **Production fleet** — 4 runs | best general-vs-pharma mix | 4 × (4×H100) | ~4B ea | **45–60% domain** wins; 75% overfits |
| 5 | **Winner run** `prod_winner_v2` | a long run on enriched data | 8×H100 | **20B** | best base model, no overfitting |
| 6 | **SFT** (instruction tuning) | turn completer → Q&A | 1×H100 | 70k examples | improves every metric |
| 7 | **Eval** (MedMCQA/PubMedQA) | how good is it, really? | 1×H100 | — | see [Results](#results) |
| 8 | **Live endpoint** | serve it on the web | L4 | — | `/generate`, `/mcq` |

### 3 — Ablation sweep (7 runs, judged at a matched step 2250, by PubMed/domain val-loss)

| Run | Tests | Domain val ↓ | General val ↓ |
|---|---|---|---|
| `r1_mix50` | 50/50 mix, lr 6e-4 | **2.843** | 3.595 |
| `r1_lr1e3` | 65/35, lr **1e-3** | 2.900 | 3.527 |
| `r1_lr6e4` | 65/35, lr 6e-4 | 2.909 | 3.540 |
| `r1_lr15e4` | 65/35, lr 1.5e-3 | 2.910 | 3.539 |
| `r1_gqa` | 65/35, grouped-query attn | 2.922 | 3.556 |
| `r1_lr3e4` | 65/35, lr 3e-4 | 2.970 | 3.612 |
| `r1_mix80` | 80/20 mix | 3.013 | 3.496 |

→ LR **1e-3** chosen; **3e-4** clearly worst; GQA tied (free inference speedup, not used in the final for max quality).

### 4 — Production fleet (4 mixes, SFT'd, evaluated)

| Run | Domain mix | MedMCQA (base→SFT) | PubMedQA (base→SFT) |
|---|---|---|---|
| `prod_d30` | 30% | _lost to a checkpoint-overwrite bug (since fixed)_ | — |
| `prod_d45` | 45% | 0.289 → **0.304** | 0.548 → 0.598 |
| `prod_d60` ✅ | 60% | 0.291 → 0.300 | 0.528 → **0.608** |
| `prod_d75` | 75% | 0.294 → 0.288 | **0.346** → 0.590 |

→ **`prod_d75` had the best pretraining loss but the worst downstream QA** — it over-repeated the 1B PubMed
corpus and overfit the *style*. Balanced mix (`d60`) won. Lesson: **lower perplexity ≠ better answers.**

### 5 — Winner run `prod_winner_v2`
350M · **20B tokens** · 8×H100 · mix 50% general / 50% domain (size-weighted across 5 domain sources) ·
LR 1e-3 · dropout 0.05 · completed cleanly (step 9537, best val-loss **2.508**) · **no overfitting** the whole run.

---

## Cost & GPUs consumed

Modal bills **per-second**. Rates at time of project: **H100 ≈ $3.95/GPU-hr**, **L4 ≈ $0.80/GPU-hr**, CPU
negligible. The table below is an **estimate** computed as `GPU-hours × rate` from each run's wall-clock and
GPU count. For the **exact billed figure**, see the Modal dashboard → `modal.com/settings/usage` (workspace
`teamvizuara`).

| Phase | GPUs | ~GPU-hours | ~Cost |
|---|---|---:|---:|
| Smoke tests ×3 | 1×H100 | 0.5 | $2 |
| Tokenizer (32k BPE) | CPU 8-core | — | $1 |
| Data prep (shard-parallel, 24 workers) | CPU 8-core ×24 | — | $5 |
| Ablation sweep (7 runs × ~0.5h) | 1×H100 each | ~3.6 | $14 |
| Production fleet (4 runs × ~2.5h) | 4×H100 each | ~40 | ~$158 |
| Winner run (20B tokens, ~6.4h) | 8×H100 | ~51 | ~$202 |
| SFT + eval + finalize | 1×H100 | ~6 | ~$24 |
| Live endpoint (scale-to-zero) | L4 | per-use | ~$1/session |
| **Total (estimated)** | | **~100 H100-hrs** | **≈ $400–700** |

> The range reflects uncertainty in exact fleet wall-clock. Budget set for the project: **$1,000**.
> Throughput observed: ~190k tok/s on 1×H100 (350M, seq 1024); ~865k tok/s on 8×H100 (seq 2048).

**Why H100s, and how many at once:** the sweep ran **7 H100s in parallel** (one per ablation); the fleet ran
**16 H100s in parallel** (4 runs × 4 GPUs); the winner used **8 H100s** (DDP). Data prep used **~24 CPU
containers** in parallel. The serving endpoint uses a single cheap **L4** that scales to zero when idle.

---

## Architecture

**Model** (`src/model.py`) — a modern, compact decoder-only Transformer (hand-written, plain PyTorch):

| Component | Choice |
|---|---|
| Params | ~350M |
| dim / layers / heads | 1024 / 24 / 16 |
| Context length | 2048 |
| Vocab | 32,000 (custom BPE) |
| Positional encoding | **RoPE** (rotary) |
| Normalization | **RMSNorm** (pre-norm) |
| Feed-forward | **SwiGLU** |
| Attention | **FlashAttention** via `F.scaled_dot_product_attention`; optional **GQA** |
| Tied embeddings | yes |
| Precision | bf16 + `torch.compile` |

**Tokenizer** (`src/tokenizer_train.py`) — 32k **byte-level BPE** (HuggingFace `tokenizers`), trained on a
blend of general + pharma text. Special tokens: `<|endoftext|> <|pad|> <|system|> <|user|> <|assistant|>`.
Measured **fertility 1.57** tokens/word on dense pharma text (e.g. `pharmacokinetics`, `chromatography`
become 1–2 tokens instead of 5–6).

**Training** (`src/train.py`) — hand-written DDP loop: cosine LR schedule with warmup, gradient accumulation,
**per-source validation loss** (so domain vs general is comparable across mixes), live metrics to a
`modal.Dict`, **self-abandonment** (kills diverging/plateaued runs), and **resume from `latest.pt`**.

**SFT** (`src/sft.py`) — chat template `<|user|>\n{q}<|assistant|>\n{a}<|endoftext|>`, **loss masked to the
response tokens only**.

**Eval** (`src/evaluate.py`) — zero-shot multiple-choice by **answer-likelihood**: score each option by the
model's length-normalized log-probability, pick the argmax. No task-specific training.

---

## Data — exact sources, sizes, licenses

All open, streamed from HuggingFace as parquet (shard-parallel tokenization). Final corpus ≈ **13B tokens**.

| Source (HF id) | Role | Tokens | License notes |
|---|---|---:|---|
| `HuggingFaceFW/fineweb-edu` (`sample-10BT`) | general English | 10.09B | ODC-BY |
| `TaylorAI/pubmed_commercial` | PMC **full-text** | 1.34B | commercial-use PMC subset |
| `casinca/PUBMED_title_abstracts_2019_baseline` | PubMed abstracts | 1.01B | NLM terms |
| `MedRAG/pubmed` | PubMed snippets | 0.45B | see dataset card |
| `epfl-llm/guidelines` | clinical guidelines | 0.10B | see dataset card |
| `MedRAG/textbooks` | medical textbooks | 0.02B | see dataset card |
| **Domain total** | | **~2.92B** | |

**SFT data** (`src/sft_data.py`, 70k examples → `/vol/sft/train.jsonl`): 30k `openlifescienceai/medmcqa`
(train), 30k `bigbio/pubmed_qa` (artificial), 10k `yahma/alpaca-cleaned` (general).

**Held-out eval** (never trained on): MedMCQA *validation*, `GBaker/MedQA-USMLE-4-options` *test*,
PubMedQA *labeled test*.

> Note: HF auto-converts large non-parquet datasets to a *partial* `refs/convert/parquet` branch, so MedRAG/PMC
> came in smaller than their full size. `src/sources.py` resolves both the main and the convert branch.

---

## Reproduce it end-to-end

**Prereqs:** Python 3.11, a [Modal](https://modal.com) account.

```bash
pip install modal
modal token set --token-id <YOUR_ID> --token-secret <YOUR_SECRET>
git clone https://github.com/VizuaraAI/pharma-slm && cd pharma-slm
```

```bash
# 1) Build the custom 32k tokenizer  (CPU, ~10 min)
modal run --detach modal_app/app.py::setup_tokenizer \
    --general-docs 200000 --domain-docs 200000 --vocab-size 32000

# 2) Tokenize the corpus, shard-parallel  (CPU, ~30-60 min)
modal run --detach modal_app/app.py::prepare_all --only fineweb_edu,pubmed
modal run --detach modal_app/app.py::prepare_all --only medrag_pubmed,pmc_commercial,guidelines,med_textbooks

# 3) (optional) end-to-end smoke test on tiny data  (1×H100, ~3 min)
modal run modal_app/app.py::smoke

# 4) Ablation sweep — find the best LR/mix/arch  (7×H100, ~35 min)
python configs/sweep.py
modal run --detach modal_app/app.py::sweep
modal run modal_app/app.py::status --group ablations          # live leaderboard

# 5) Production fleet — compare data mixes  (16×H100)
python configs/fleet.py
modal run --detach modal_app/app.py::prod_fleet
modal run modal_app/app.py::status --group prod_fleet

# 6) Build the SFT dataset  (CPU)
modal run modal_app/app.py::sft_data

# 7) Train the winner  (8×H100, ~6.5h)  — config in configs/winner_v2.json
modal run --detach modal_app/app.py::launch_prod --config-file configs/winner_v2.json

# 8) SFT + eval + sample answers  (1×H100, ~25 min) -> /vol/runs/prod_winner_v2/eval.json
modal run --detach modal_app/app.py::finalize_winner

# 9) Chat with it from the CLI
modal run modal_app/app.py::gen \
    --ckpt /vol/runs/prod_winner_v2/best_sft.pt \
    --prompt "What is the mechanism of action of ibuprofen?" --chat

# 10) Deploy the live web API + the demo site
modal deploy modal_app/serve.py
python scripts/bake_demo.py        # bakes real model outputs into the site
cd site && vercel --prod
```

**Monitoring:** all long jobs are launched with `--detach` (survive laptop sleep/disconnect). Watch progress
via `modal run modal_app/app.py::status --group <ablations|prod_fleet>` or the metrics `modal.Dict`.

---

## Repo layout

```
src/
  config.py          dataclass configs (model + training); dict-merge for experiments
  model.py           the Transformer (RoPE/RMSNorm/SwiGLU/Flash/GQA)
  data.py            mixing data loader — samples sources by weight at batch time
  sources.py         HF source registry + parquet listing + streaming-shard
  tokenizer_train.py train the 32k byte-level BPE
  prepare_data.py    shard-parallel tokenize -> /vol/data/<src>.{train,val}.bin (uint16)
  train.py           DDP pretraining: eval, per-source val, checkpoint, self-abandon, resume
  sft.py             instruction tuning (chat template, loss masked to response)
  sft_data.py        build the 70k-example SFT jsonl
  evaluate.py        zero-shot MCQ by answer-likelihood (MedMCQA/MedQA/PubMedQA)
  generate.py        inference / sampling
modal_app/
  app.py             Modal harness: image, Volume, metrics Dict, all GPU functions + entrypoints
  serve.py           live FastAPI inference endpoint (/generate, /mcq, /health)
configs/
  sweep.py           generate the 7 ablation configs + production.json
  fleet.py           generate the 4 production-fleet configs
  ablations/*.json   the 7 ablation experiment configs
  prod_fleet/*.json  the 4 production experiment configs
  winner_v2.json     the final winning-model config
  *_results.json     recorded experiment results
scripts/
  bake_demo.py       call the live endpoint and bake real outputs into the site
site/
  index.html         the Shopflo-themed demo site (live chat + MCQ)
  demo_data.json     baked real model outputs
```

---

## Methodology, decisions & lessons

- **Why ~50% domain, not 99%?** Famous medical models (Meditron, SaulLM) use ~99% domain — but those are
  *continued-pretraining* on a base that already speaks English. We train **from scratch**, so we keep far
  more general text. The fleet measured the sweet spot empirically: **~45–60% domain**.
- **Over-train past Chinchilla.** Chinchilla-optimal for 350M is ~7B tokens; we trained 20B (~57×) because
  small models keep improving well past the compute-optimal point — that's what makes them *fluent*.
- **Custom tokenizer is free when from-scratch** and the one thing you can't do when fine-tuning. We took it.
- **Judge on real tasks, not loss.** The 75%-domain model had the lowest loss and the worst answers.
- **Run experiments in parallel and let them self-abandon** — like a small research team, not one run at a time.

## Bugs we hit (and the fixes)

These are documented because they're the non-obvious part of doing this for real:

1. **Missing config field** — configs referenced `target_tokens`, dataclass didn't declare it → silently dropped. *Fix: declare it.*
2. **`cannot mmap an empty file`** — a Modal **warm container held a stale Volume mount**, so `finalize` saw 0 shards. *Fix: `vol.reload()` before every read; guard empty files.*
3. **Live metrics never appeared** — the `modal` package wasn't in the training image, so the torchrun subprocess couldn't write the `modal.Dict`. *Fix: add `modal` to the image.*
4. **`UnboundLocalError: torch`** — `import torch._dynamo` inside `main()` shadowed the module-level `torch`. *Fix: import at module level.*
5. **`.view()` on non-contiguous SFT targets** — training stacked targets (contiguous) but SFT slices them. *Fix: `.reshape()`.*
6. **Laptop sleep killed a non-detached run** — overnight, a blocking `modal run` died. *Fix: `--detach` + `.spawn()` for all long jobs; server-side drivers.*
7. **Modal auto-retry overwrote a good checkpoint** (`prod_d30` lost). *Fix: resume from `latest.pt` instead of restarting.*

## Limitations & next steps

- **350M is capacity-bound on hard MCQ.** For board-exam-level accuracy, scale to **~1B+** params — change
  `dim`/`n_layers` in a config and re-run `launch_prod`; the entire pipeline already supports it.
- **Domain corpus is ~2.9B unique tokens** (HF partial-parquet limited the big sources). Pulling full
  MedRAG/PMC via the streaming-shard path (`iter_source_shard` in `src/sources.py`) would add more.
- Not a medical device; outputs can be wrong. For education/research only.

---

_Built end-to-end as a from-scratch SLM project. Every answer on the demo site is produced by the model trained here._
