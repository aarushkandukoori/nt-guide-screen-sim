"""Two figures from results/replicates.csv and results/summary.csv (Okabe-Ito colors)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
S = pd.read_csv(os.path.join(HERE, "results", "summary.csv"))
D = pd.read_csv(os.path.join(HERE, "results", "diagnostic_verdicts.csv"), index_col=0)

RULES = ["R0_keep_all", "R1_circular", "R2_split_genes", "R2b_split_cells",
         "R3_diagnose_then_split_cells", "Oracle_model_batch", "Oracle_batch_then_split_cells"]
LABELS = ["R0 keep all", "R1 circular", "R2 split genes", "R2b split cells",
          "R3 diagnose, then R2b", "Model batch (oracle)", "Model batch + R2b (oracle)"]
COLORS = ["#999999", "#D55E00", "#56B4E9", "#0072B2", "#009E73", "#E69F00", "#CC79A7"]
SCEN = ["a", "b", "c", "d"]
SCEN_LAB = ["(a) all clean", "(b) planted\noff-target", "(c) omitted\ncovariate", "(d) both"]


def grouped(ax, col, ylabel, title, hline=None):
    w = 0.115
    x = np.arange(len(SCEN))
    for i, (r, lab, c) in enumerate(zip(RULES, LABELS, COLORS)):
        vals = [S[(S.scenario == s) & (S.rule == r)][col].values[0] for s in SCEN]
        ax.bar(x + (i - 3) * w, vals, w, label=lab, color=c, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(SCEN_LAB, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    if hline is not None:
        ax.axhline(hline, ls="--", color="black", lw=0.8)


plt.rcParams.update({"font.size": 9})

# Figure 1 - is the post-rule calibration check honest, and which guides get dropped?
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
grouped(axes[0], "calib_pass_rate", "Share of replicates",
        "Calibration check passes after the rule")
for xi, ok in zip(range(4), ["should pass", "should pass\nonly if bad\nguides dropped",
                             "should fail\n(model wrong)", "should fail\n(model wrong)"]):
    axes[0].text(xi, 1.04, ok, ha="center", va="bottom", fontsize=6.5)
axes[0].set_ylim(0, 1.3)
grouped(axes[1], "clean_drop_rate", "Share of clean NT guides dropped",
        "Clean guides thrown out")
sub = S[S.scenario.isin(["b", "d"])]
w = 0.115
for i, (r, lab, c) in enumerate(zip(RULES, LABELS, COLORS)):
    vals = [sub[(sub.scenario == s) & (sub.rule == r)].planted_catch_rate.values[0] for s in ["b", "d"]]
    axes[2].bar(np.arange(2) + (i - 3) * w, vals, w, color=c, edgecolor="black", linewidth=0.4)
axes[2].set_xticks(np.arange(2))
axes[2].set_xticklabels(SCEN_LAB[1:2] + SCEN_LAB[3:4], fontsize=8)
axes[2].set_ylabel("Share of planted guides dropped")
axes[2].set_title("Planted off-target guides caught", fontsize=10)
axes[2].set_ylim(0, 1.05)
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=7, fontsize=7.5,
           frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(os.path.join(HERE, "figures", "fig1_check_honesty_and_drops.png"), dpi=200)

# Figure 2 - false discoveries on targeting pairs, and the diagnostic verdicts
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.7, 1]})
grouped(axes[0], "mean_fdp", "Mean false discovery proportion",
        "Targeting pairs, BH at q = 0.1", hline=0.1)
axes[0].legend(fontsize=7, frameon=False, ncol=2)
cols = ["clean", "offtarget", "covariate"]
vcol = {"clean": "#999999", "offtarget": "#0072B2", "covariate": "#D55E00"}
vlab = {"clean": "no signal", "offtarget": "concentrated (off-target-like)",
        "covariate": "spread out (covariate-like)"}
bottom = np.zeros(4)
for v in cols:
    vals = np.array([D.loc[s, v] if v in D.columns else 0 for s in SCEN], dtype=float)
    vals = vals / D.loc[SCEN].sum(axis=1).values
    axes[1].bar(np.arange(4), vals, 0.6, bottom=bottom, color=vcol[v], label=vlab[v],
                edgecolor="black", linewidth=0.4)
    bottom += vals
axes[1].set_xticks(np.arange(4))
axes[1].set_xticklabels(SCEN_LAB, fontsize=8)
axes[1].set_ylabel("Share of replicates")
axes[1].set_title("Diagnostic verdict (all NT guides, all genes)", fontsize=10)
axes[1].legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2))
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "fig2_false_discoveries_and_diagnostic.png"), dpi=200)
print("figures written")
