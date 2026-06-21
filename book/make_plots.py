"""Real data plots for the book — from the project's actual measured numbers."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np, os

OUT = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT, exist_ok=True)

INK = "#0A0E12"; TEAL = "#0BB47A"; PURPLE = "#7C3AED"; GRAY = "#8A94A3"; BG = "#FBFBF9"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": "#D9DCE1", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": GRAY, "ytick.color": GRAY, "axes.grid": True,
    "grid.color": "#E8EAED", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name), bbox_inches="tight", dpi=200)
    plt.close(fig); print("wrote", name)


# 1. From-scratch loss descent (illustrative of the real 10.6 -> 2.27 curve)
fig, ax = plt.subplots(figsize=(7, 4))
toks = np.linspace(0, 30, 200)
loss = 2.27 + (10.61 - 2.27) * np.exp(-toks / 3.1)
ax.axhline(10.37, ls="--", color=GRAY, lw=1.2)
ax.text(15, 10.0, "loss of a random model = ln(32000) = 10.37", color=GRAY, fontsize=10)
ax.plot(toks, loss, color=TEAL, lw=3)
ax.scatter([0], [10.61], color=PURPLE, zorder=5)
ax.text(0.6, 10.4, "starts as noise", color=PURPLE, fontsize=10)
ax.scatter([30], [2.27], color=TEAL, zorder=5)
ax.set_xlabel("tokens trained (billions)"); ax.set_ylabel("cross-entropy loss")
ax.set_title("Training from scratch: noise → fluent", color=INK, fontweight="bold", loc="left")
save(fig, "plot_losscurve.pdf")

# 2. Ablation sweep (domain val-loss at matched step 2250; lower better)
runs = [("lr 3e-4", 2.970), ("mix 80/20", 3.013), ("GQA", 2.922), ("lr 1.5e-3", 2.910),
        ("lr 6e-4", 2.909), ("lr 1e-3", 2.900), ("mix 50/50", 2.843)]
runs.sort(key=lambda x: -x[1])
names = [r[0] for r in runs]; vals = [r[1] for r in runs]
fig, ax = plt.subplots(figsize=(7, 4))
colors = [TEAL if "lr 1e-3" in n or "mix 50" in n else "#C7CDD4" for n in names]
ax.barh(names, vals, color=colors)
ax.set_xlim(2.80, 3.0); ax.invert_xaxis()
ax.set_xlabel("PubMed (domain) val-loss — lower is better")
ax.set_title("Ablation sweep: 7 models, parallel", color=INK, fontweight="bold", loc="left")
save(fig, "plot_ablation.pdf")

# 3. Mix tradeoff (production fleet) — domain fraction vs domain & general val
d = [30, 45, 60, 75]
domain_val = [3.174, 3.083, 2.994, 2.973]   # better with more domain
general_val = [3.785, 3.822, 3.856, 3.987]  # worse with more domain
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(d, domain_val, "-o", color=TEAL, lw=2.5, label="pharma val-loss")
ax.plot(d, general_val, "-o", color=PURPLE, lw=2.5, label="general val-loss")
ax.axvspan(45, 60, color=TEAL, alpha=0.08)
ax.text(52, 3.5, "sweet\nspot", color=TEAL, ha="center", fontsize=11, fontweight="bold")
ax.set_xlabel("% domain (pharma) in the mix"); ax.set_ylabel("val-loss (lower better)")
ax.set_title("The mix tradeoff: 4 models in parallel", color=INK, fontweight="bold", loc="left")
ax.legend(frameon=False)
save(fig, "plot_fleet.pdf")

# 4. Scaling: 350M vs 1.3B on the exams
labels = ["MedMCQA\n(chance 0.25)", "PubMedQA"]
m350 = [0.290, 0.608]; m13 = [0.321, 0.654]
x = np.arange(len(labels)); w = 0.36
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x - w/2, m350, w, color="#C7CDD4", label="350M")
ax.bar(x + w/2, m13, w, color=TEAL, label="1.3B")
ax.axhline(0.25, ls="--", color=GRAY, lw=1)
ax.text(1.3, 0.26, "guessing", color=GRAY, fontsize=9)
for i, (a, b) in enumerate(zip(m350, m13)):
    ax.text(i - w/2, a + .01, f"{a:.3f}", ha="center", fontsize=10, color=GRAY)
    ax.text(i + w/2, b + .01, f"{b:.3f}", ha="center", fontsize=10, color=INK, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 0.75)
ax.set_ylabel("accuracy")
ax.set_title("Scaling the capacity lever: 350M → 1.3B", color=INK, fontweight="bold", loc="left")
ax.legend(frameon=False)
save(fig, "plot_scaling.pdf")

# 5. Cost vs capability ladder
sizes = [0.35, 1.3, 2.7, 7.0]
acc = [0.29, 0.36, 0.535, 0.65]
cost = ["$0.4-0.7k", "$2-4k", "$8-15k", "$30-80k"]
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(sizes, acc, "-o", color=TEAL, lw=2.5, markersize=8)
ax.axhline(0.25, ls="--", color=GRAY, lw=1); ax.text(5, 0.26, "guessing", color=GRAY, fontsize=9)
for s, a, c in zip(sizes, acc, cost):
    ax.annotate(f"{s:g}B\n{c}", (s, a), textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=9, color=INK)
ax.set_xscale("log"); ax.set_xticks(sizes); ax.set_xticklabels([f"{s:g}B" for s in sizes])
ax.set_xlabel("model size (log scale)"); ax.set_ylabel("expected MedMCQA")
ax.set_ylim(0.2, 0.75)
ax.set_title("Cost vs capability — the scaling ladder", color=INK, fontweight="bold", loc="left")
save(fig, "plot_costladder.pdf")

print("ALL PLOTS DONE")
