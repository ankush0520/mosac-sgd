import sys, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
out = sys.argv[1]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
C = {"Louvain (disjoint)": "#8a8a8a", "SLPA": "#2a78b8", "DEMON": "#d9622b"}
v = pd.read_csv(f"{out}/validation_wmin_sweep.csv")
fig, ax = plt.subplots(figsize=(5.2, 3.3))
for m, g in v.groupby("method", sort=False):
    ax.plot(g.w_min, g.gF1, "o-", color=C[m], label=m, lw=2)
ax.set_xlabel("edge threshold $w_{min}$"); ax.set_ylabel("group F1@30 (validation)")
ax.set_ylim(0, 1.05); ax.set_xticks([0.05, 0.3, 0.6, 0.9]); ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(f"{out}/fig_wmin_sweep.pdf")
d = pd.read_csv(f"{out}/results_raw.csv")
d = d[d.prop.isin(["single", "noisy_or"])]
m = d.groupby("method", sort=False)[["gF1", "gRecAll", "MMR"]].agg(["mean", "std"])
labels = ["Group F1@30", "Campaigns recovered\n(J ≥ 0.5)", "Multi-campaign spammers\nfound in ≥2 campaigns (MMR)"]
x = np.arange(3); w = 0.26
fig, ax = plt.subplots(figsize=(7.2, 3.3))
for i, meth in enumerate(m.index):
    mu = [m.loc[meth, (k, "mean")] for k in ["gF1", "gRecAll", "MMR"]]
    sd = [m.loc[meth, (k, "std")] for k in ["gF1", "gRecAll", "MMR"]]
    b = ax.bar(x + (i - 1) * w, mu, w, yerr=sd, capsize=3, color=C[meth], label=meth)
    for xi, val, e in zip(x + (i - 1) * w, mu, sd): ax.text(xi, val + e + 0.03, f"{val:.2f}", ha="center", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5); ax.set_ylim(0, 1.22); ax.set_ylabel("mean over 5 seeds")
ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=3)
fig.tight_layout(); fig.savefig(f"{out}/fig_stage1_bars.pdf")
