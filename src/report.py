"""Turn results/*.csv into figures (results/figs/) and results/RESULTS.md.

    python -m src.report results
"""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
PAL = ["#2a78b8", "#d9622b", "#3a9a5b", "#8a8a8a", "#a05cb8", "#c9a227", "#5b5b5b"]


def ms(s, nd=3):
    s = pd.Series(s).dropna()
    if len(s) == 0:
        return "--"
    return f"{s.mean():.{nd}f} ± {s.std(ddof=1) if len(s) > 1 else 0:.{nd}f}"


def table(df, index, cols, nd=3, sort=False):
    g = df.groupby(index, sort=sort)
    lines = ["| " + " | ".join([index if isinstance(index, str) else " / ".join(index)] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for k, sub in g:
        k = k if isinstance(k, str) else " / ".join(map(str, k)) if isinstance(k, tuple) else str(k)
        lines.append("| " + " | ".join([k] + [ms(sub[c], nd) if c in sub else "--" for c in cols]) + " |")
    return "\n".join(lines)


def wil(df, a, b, metric, key="method", by=("host", "seed")):
    x = df[df[key] == a].set_index(list(by))[metric]; y = df[df[key] == b].set_index(list(by))[metric]
    j = x.index.intersection(y.index)
    if len(j) < 2 or np.allclose(x[j].values, y[j].values):
        return np.nan
    try:
        return float(wilcoxon(x[j].values, y[j].values).pvalue)
    except ValueError:
        return np.nan


def main(out="results"):
    figs = os.path.join(out, "figs"); os.makedirs(figs, exist_ok=True)
    md = ["# MOSAC-SGD: measured results", "",
          "All numbers below were produced by `python -m src.run` (see README). Mean ± std over seeds; "
          "group metrics use Jaccard ≥ 0.5; user-level k = number of injected spammer accounts.", ""]
    p = lambda f: os.path.join(out, f)
    if os.path.exists(p("instances.csv")):
        d = pd.read_csv(p("instances.csv"))
        md += ["## Dataset and graph statistics (test injections)", "",
               table(d, "host", ["users", "items", "reviews", "graph_edges", "campaigns", "spammers", "multi",
                                 "campaign_reviews", "camouflage_reviews", "benign_llm_reviews"], 0), ""]
    for hn in sorted(f for f in os.listdir(out) if f.startswith("selected_")):
        md += [f"Validation-selected settings ({hn[9:-5]}): `{json.load(open(p(hn), encoding='utf-8'))}`", ""]
    if os.path.exists(p("stage1.csv")):
        d = pd.read_csv(p("stage1.csv"))
        md += ["## Stage 1: group discovery (RQ2)", "",
               table(d, ["host", "algo"], ["ONMI", "gRecAll", "MR", "MMR", "n_groups", "t_stage1"]), ""]
    if os.path.exists(p("e1_main.csv")):
        d = pd.read_csv(p("e1_main.csv"))
        md += ["## E1: main comparison (T2 LLM-diverse text, benign-LLM control on, overlap 25%)", "",
               table(d, ["host", "method"], ["gP", "gF1", "uP", "uAP", "MR"]), ""]
        for hn, sub in d.groupby("host"):
            base = sub[sub.method != "MOSAC-SGD"].groupby("method").gF1.mean().idxmax()
            md.append(f"- {hn}: best baseline by group F1 = **{base}**; Wilcoxon p (MOSAC-SGD vs it): "
                      f"gF1 {wil(sub, 'MOSAC-SGD', base, 'gF1')}, user AUC-PR {wil(sub, 'MOSAC-SGD', base, 'uAP')} "
                      f"(n = {sub.seed.nunique()} seeds; with 5 seeds the smallest attainable two-sided p is 0.0625).")
        md.append("")
        c = pd.read_csv(p("e1_pcurve.csv"))
        fig, ax = plt.subplots(figsize=(5.4, 3.3))
        prim = c.host.iloc[0]
        for i, (m, g) in enumerate(c[c.host == prim].groupby("method")):
            mm = g.groupby("k").P.mean(); ax.plot(mm.index, mm.values, "o-", ms=3, lw=1.6, color=PAL[i % len(PAL)], label=m)
        ax.set_xlabel("k (top-k groups)"); ax.set_ylabel("group precision@k"); ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=7); ax.set_title(prim, fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(figs, "e1_precision_at_k.png"), dpi=160); plt.close(fig)
        md += ["![precision@k](figs/e1_precision_at_k.png)", ""]
    if os.path.exists(p("e2_text.csv")):
        d = pd.read_csv(p("e2_text.csv"))
        d["cond"] = d.text + " / ctrl " + d.control.map({True: "on", False: "off"})
        md += ["## E2 (RQ1): does text help against LLM campaigns? Group F1@k", "",
               "| method | " + " | ".join(sorted(d.cond.unique())) + " |", "|---|" + "---|" * d.cond.nunique()]
        for m, sub in d.groupby("method", sort=False):
            md.append(f"| {m} | " + " | ".join(ms(sub[sub.cond == c].gF1, 2) for c in sorted(d.cond.unique())) + " |")
        md += ["", "User AUC-PR:", "", "| method | " + " | ".join(sorted(d.cond.unique())) + " |", "|---|" + "---|" * d.cond.nunique()]
        for m, sub in d.groupby("method", sort=False):
            md.append(f"| {m} | " + " | ".join(ms(sub[sub.cond == c].uAP, 2) for c in sorted(d.cond.unique())) + " |")
        md.append("")
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        conds = sorted(d.cond.unique()); meths = list(d.method.unique()); x = np.arange(len(conds)); wd = 0.8 / len(meths)
        for i, m in enumerate(meths):
            mu = [d[(d.method == m) & (d.cond == c)].gF1.mean() for c in conds]
            sd = [d[(d.method == m) & (d.cond == c)].gF1.std() for c in conds]
            ax.bar(x + (i - (len(meths) - 1) / 2) * wd, mu, wd, yerr=sd, capsize=2, color=PAL[i], label=m)
        ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=8); ax.set_ylabel("group F1@k"); ax.set_ylim(0, 1.1)
        ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper left")
        fig.tight_layout(); fig.savefig(os.path.join(figs, "e2_text_conditions.png"), dpi=160); plt.close(fig)
        md += ["![E2](figs/e2_text_conditions.png)", ""]
        if os.path.exists(p("e2_scores.csv")):
            s = pd.read_csv(p("e2_scores.csv"))
            fig, axs = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
            for ax, (m, g) in zip(axs, s.groupby("method")):
                for i, k in enumerate(["other user", "benign-LLM user", "LLM spammer"]):
                    v = g[g.kind == k].p
                    if len(v):
                        ax.hist(v, bins=25, range=(0, 1), alpha=0.6, density=True, color=PAL[i], label=f"{k} (n={len(v)})")
                ax.set_title(m, fontsize=9); ax.set_xlabel("user score p_u (users in candidate groups)")
            axs[0].legend(frameon=False, fontsize=7)
            fig.tight_layout(); fig.savefig(os.path.join(figs, "e2_score_distribution.png"), dpi=160); plt.close(fig)
            md += ["![scores](figs/e2_score_distribution.png)", ""]
    if os.path.exists(p("e3_overlap.csv")):
        d = pd.read_csv(p("e3_overlap.csv"))
        top = d[d.overlap == d.overlap.max()]
        md += [f"## E3 (RQ2): multi-campaign spammers (overlap {top.overlap.iloc[0]:.0%})", "",
               table(top, ["grouping", "propagation"], ["MR", "MMR", "R_multi", "R_c1", "R_c2", "R_c3", "uP", "uAP"]), ""]
        fig, axs = plt.subplots(1, 2, figsize=(8, 3.1))
        sel = d[(d.propagation.isin(["single group", "noisy_or"]))]
        for i, (gname, g) in enumerate(sel.groupby("grouping")):
            mm = g.groupby("overlap")[["MR", "R_multi"]].mean()
            axs[0].plot(mm.index * 100, mm.MR, "o-", color=PAL[i], label=gname)
            axs[1].plot(mm.index * 100, mm.R_multi, "o-", color=PAL[i], label=gname)
        axs[0].set_xlabel("overlap rate (%)"); axs[0].set_ylabel("membership recall MR")
        axs[1].set_xlabel("overlap rate (%)"); axs[1].set_ylabel("user R@k on multi-campaign spammers")
        axs[0].legend(frameon=False, fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(figs, "e3_overlap.png"), dpi=160); plt.close(fig)
        md += ["![E3](figs/e3_overlap.png)", ""]
    if os.path.exists(p("e4_ablation.csv")):
        d = pd.read_csv(p("e4_ablation.csv"))
        full = d[d.variant == "Full MOSAC-SGD"].set_index("seed")
        d["dF1"] = d.apply(lambda r: r.gF1 - full.loc[r.seed, "gF1"], axis=1)
        d["dAP"] = d.apply(lambda r: r.uAP - full.loc[r.seed, "uAP"], axis=1)
        md += ["## E4: ablations (primary host)", "", table(d, "variant", ["gF1", "dF1", "uAP", "dAP", "MR"]), ""]
    if os.path.exists(p("e5_sensitivity.csv")):
        d = pd.read_csv(p("e5_sensitivity.csv"))
        md += ["## E5: sensitivity, robustness and runtime (primary host)", "",
               table(d, ["param", "value"], ["gF1", "uAP", "MR", "reviews", "edges", "t_graph", "t_stage1", "t_embed", "t_stage23"], 3), ""]
        fig, axs = plt.subplots(1, 3, figsize=(10, 2.9))
        for ax, prm in zip(axs[:2], ["sigma_t", "w_min"]):
            g = d[d.param == prm].assign(v=lambda x: x.value.astype(float)).groupby("v")[["gF1", "uAP"]].mean()
            ax.plot(g.index, g.gF1, "o-", color=PAL[0], label="group F1@k"); ax.plot(g.index, g.uAP, "s--", color=PAL[1], label="user AUC-PR")
            ax.set_xlabel(prm); ax.set_ylim(0, 1.05)
        axs[0].legend(frameon=False, fontsize=7)
        g = d[d.param == "size"].groupby("reviews")[["t_graph", "t_stage1", "t_embed", "t_stage23"]].mean()
        for i, c in enumerate(g.columns):
            axs[2].loglog(g.index, g[c] + 1e-2, "o-", color=PAL[i], label=c[2:])
        axs[2].set_xlabel("reviews"); axs[2].set_ylabel("seconds"); axs[2].legend(frameon=False, fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(figs, "e5_sensitivity_runtime.png"), dpi=160); plt.close(fig)
        md += ["![E5](figs/e5_sensitivity_runtime.png)", ""]
    if os.path.exists(p("case_study.json")):
        c = json.load(open(p("case_study.json"), encoding="utf-8"))
        md += ["## Case study: top-scored group (seed 0, primary host)", "", "```", json.dumps(c, indent=1)[:3000], "```", ""]
    if os.path.exists(p("e6_real_labels.csv")):
        d = pd.read_csv(p("e6_real_labels.csv"))
        md += ["## E6: agreement with real labels (CARE-GNN YelpChi / Amazon; behaviour only, no text available)", "",
               "```\n" + d.round(4).to_string(index=False) + "\n```", ""]
    if os.path.exists(p("opspam_text_sanity.csv")):
        d = pd.read_csv(p("opspam_text_sanity.csv"))
        md += ["## Text-branch sanity check (OpSpam)", "", "```", d.round(4).to_string(index=False), "```", ""]
    if os.path.exists(p("runtime_experiments.csv")):
        md += ["## Wall-clock per experiment (seconds)", "", "```", pd.read_csv(p("runtime_experiments.csv"), index_col=0).rename(columns={"0": "seconds"}).round(0).to_string(), "```", ""]
    with open(p("RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("wrote", p("RESULTS.md"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
