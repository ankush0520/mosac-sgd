"""Run every experiment of the end-sem report and write results/.

    python -m src.run                         # full run (all experiments, 5 seeds, 2 hosts)
    python -m src.run --quick                 # 2 seeds, 1 host, small E5: a fast smoke run
    python -m src.run --only E1 E3            # a subset (tuning always runs first, cached)

Outputs (results/): tuning.csv, selected.json, stage1.csv, e1_main.csv, e1_pcurve.csv,
e2_text.csv, e2_scores.csv, e3_overlap.csv, e4_ablation.csv, e5_*.csv, e6_real_labels.csv,
opspam_text_sanity.csv, instances.csv, case_study.json, and RESULTS.md (via src.report).
"""
import argparse, json, os, time, copy, collections
import numpy as np, pandas as pd
from .config import load_config, with_overrides
from . import pipeline as P, baselines as BL, stage3_detect as S3, evaluate as EV, realdata as RD

OVERLAP_ALGOS = ["BigCLAM", "DEMON", "SLPA"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(rows, out, name):
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out, name), index=False); return df


# ------------------------------------------------------------------ tuning (validation injection)
def tune(host, cfg, out):
    path = os.path.join(out, "selected.json")
    if os.path.exists(path):
        sel = json.load(open(path, encoding="utf-8"))
        if sel.get("host") == host.name:
            log("tuning: using cached", path); return sel
    log("tuning on validation injection, seed", cfg["run"]["val_seed"])
    inst = P.Instance(host, cfg, cfg["run"]["val_seed"])
    rows = []
    for w in cfg["graph"]["w_grid"]:
        for algo in cfg["stage1"]["algorithms"]:
            r = inst.stage1_row(algo, w); r["score"] = (r["gRecAll"] + (r["MR"] if r["MR"] == r["MR"] else 0)) / 2
            rows.append(r); log("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    t = pd.DataFrame(rows)
    best_w = {a: float(t[t.algo == a].sort_values(["score", "w_min"], ascending=[False, False]).iloc[0].w_min)
              for a in cfg["stage1"]["algorithms"]}
    # Pipeline choices (overlapping algorithm, scorer, tau_g) on the same validation
    # instance; score = mean of validation group F1@k, user AUC-PR and membership recall.
    df, E, _ = inst.texts()
    srows = []
    for algo in OVERLAP_ALGOS:
        w = best_w[algo]; groups, _ = inst.groups(algo, w)
        for scorer in ("mahalanobis", "iforest", "directional"):
            for tau in (0.0, 0.5, 0.8):
                c = with_overrides(cfg, stage3__scorer=scorer, stage3__tau_g=tau)
                S, p, _ = P.run_mosac(inst, df, E, groups, c, "full", 0, w)
                r = dict(algo=algo, w_min=w, scorer=scorer, tau_g=tau, **inst.eval_groups(groups, S), **inst.eval_users(p))
                r["score"] = (r["gF1"] + r["uAP"] + r["MR"]) / 3; srows.append(r)
                log("  pipeline", algo, scorer, tau, "gF1", round(r["gF1"], 3), "uAP", round(r["uAP"], 3), "MR", round(r["MR"], 3))
    s3 = pd.DataFrame(srows).sort_values(["score", "gF1"], ascending=False).iloc[0]
    algo = s3.algo
    t.to_csv(os.path.join(out, "tuning.csv"), index=False)
    pd.DataFrame(srows).to_csv(os.path.join(out, "tuning_stage3.csv"), index=False)
    # SC-Com-lite is the one supervised baseline: train its RF on the validation instance
    sel = dict(host=host.name, best_w=best_w, algo=algo, scorer=s3.scorer, tau_g=float(s3.tau_g))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sel, f, indent=1)
    log("selected", sel)
    return sel


def apply_sel(cfg, sel):
    return with_overrides(cfg, stage3__scorer=sel["scorer"], stage3__tau_g=sel["tau_g"])


def train_sccom(host, cfg, sel):
    inst = P.Instance(host, cfg, cfg["run"]["val_seed"])
    w = sel["best_w"]["Louvain"]; groups, _ = inst.groups("Louvain", w)
    return BL.sccom_fit(groups, inst.graph(w), inst.struct, inst.camps)


# ------------------------------------------------------------------ method suite on one instance
def method_suite(inst, df, E, cfg, sel, clf, with_curves=False):
    rows, curves = [], []
    ks = list(range(5, 2 * cfg["inject"]["K"] + 1, 5))
    a, w = sel["algo"], sel["best_w"][sel["algo"]]

    def add(name, groups, S, p, sup="no", **kw):
        r = dict(method=name, supervised=sup, n_groups=len(groups) if groups is not None else np.nan, **kw)
        if groups is not None:
            r.update(inst.eval_groups(groups, S))
            if with_curves:
                for k, v in EV.precision_curve(groups, S, inst.camps, ks).items():
                    curves.append(dict(method=name, k=k, P=v))
        r.update(inst.eval_users(p)); rows.append(r)

    add("Behavioural IF", None, None, BL.behavioural_if(inst.ufeat(w), inst.seed))
    gl, _ = inst.groups("Louvain", sel["best_w"]["Louvain"])
    Sg = BL.sccom_score(clf, gl, inst.graph(sel["best_w"]["Louvain"]), inst.struct)
    add("SC-Com-lite", gl, Sg, S3.propagate(gl, Sg, "max", 0.0), sup="yes (val. labels)")
    gt, St = BL.tkccl_lite(inst.struct, seed=inst.seed)
    add("TKCCL-lite", gt, St, S3.propagate(gt, St, "max", 0.0))
    ge, _ = inst.groups("EgoSplit", sel["best_w"]["EgoSplit"])
    S, p, _ = P.run_mosac(inst, df, E, ge, cfg, "behaviour_only", inst.seed, sel["best_w"]["EgoSplit"])
    add("OSG-behav. (Ego-Splitting)", ge, S, p)
    g, t1 = inst.groups(a, w)
    S, p, _ = P.run_mosac(inst, df, E, g, cfg, "text_only", inst.seed, w)
    add("Text-only", g, S, p)
    t0 = time.time()
    S, p, ex = P.run_mosac(inst, df, E, g, cfg, "full", inst.seed, w)
    add("MOSAC-SGD", g, S, p, t_stage1=t1, t_stage23=time.time() - t0)
    return rows, curves, (g, S, p)


# ------------------------------------------------------------------ experiments
def e1(hosts, cfg, sels, clfs, seeds, out):
    rows, curves, s1rows, inst_rows = [], [], [], []
    case = None
    for hn, host in hosts.items():
        c = apply_sel(cfg, sels[hn])
        for seed in seeds:
            t0 = time.time()
            inst = P.Instance(host, c, seed)
            df, E, n_ctrl = inst.texts()
            r, cv, (g, S, p) = method_suite(inst, df, E, c, sels[hn], clfs[hn], with_curves=True)
            for x in r: x.update(host=hn, seed=seed)
            for x in cv: x.update(host=hn, seed=seed)
            rows += r; curves += cv
            for algo in cfg["stage1"]["algorithms"]:
                s1 = inst.stage1_row(algo, sels[hn]["best_w"][algo]); s1.update(host=hn, seed=seed); s1rows.append(s1)
            Gm = inst.graph(sels[hn]["best_w"][sels[hn]["algo"]])
            inst_rows.append(dict(host=hn, seed=seed, users=len(inst.users), items=inst.struct.item.nunique(),
                                  reviews=len(inst.struct), graph_nodes=Gm.number_of_nodes(), graph_edges=Gm.number_of_edges(),
                                  campaigns=len(inst.camps), benign_llm_reviews=n_ctrl, **inst.info,
                                  **{f"t_{k}": v for k, v in inst.timing.items()}))
            if case is None:
                case = case_study(inst, df, g, S)
            host.emb.save()
            log(f"E1 {hn} seed {seed} done in {time.time() - t0:.0f}s:",
                {x["method"]: round(x.get("gF1", np.nan), 3) for x in r})
    save(rows, out, "e1_main.csv"); save(curves, out, "e1_pcurve.csv"); save(s1rows, out, "stage1.csv")
    save(inst_rows, out, "instances.csv")
    with open(os.path.join(out, "case_study.json"), "w", encoding="utf-8") as f:
        json.dump(case, f, indent=1, default=str)


def case_study(inst, df, groups, S):
    i = int(np.argmax(S)); g = groups[i]
    truth = [set(c["members"]) for c in inst.camps]
    j = int(np.argmax([EV.jacc(g, t) for t in truth]))
    sub = df[df.user.isin(g)]
    cnt = sub.item.value_counts(); shared = cnt[cnt >= max(2, 0.5 * len(g))].index.tolist()
    rv = sub[sub.item.isin(shared[:1])].sort_values("t").head(2)
    return dict(group_size=len(g), S=float(S[i]), best_match_campaign=j, jaccard=EV.jacc(g, truth[j]),
                shared_items=shared, time_window_days=float(sub[sub.item.isin(shared)].t.max() - sub[sub.item.isin(shared)].t.min()),
                campaign_window_days=inst.camps[j]["window"],
                two_member_reviews=rv[["user", "item", "rating", "t", "text"]].to_dict("records"))


def e2(host, cfg, sel, seeds, out):
    rows, scores = [], []
    c = apply_sel(cfg, sel); a, w = sel["algo"], sel["best_w"][sel["algo"]]
    for seed in seeds:
        inst = P.Instance(host, c, seed); g, _ = inst.groups(a, w)
        for text in ("T0", "T1", "T2"):
            for ctrl in (False, True):
                df, E, _ = inst.texts(text, ctrl)
                for var, name in (("behaviour_only", "Behaviour-only"), ("text_only", "Text-only"), ("full", "Fused (MOSAC-SGD)")):
                    S, p, _ = P.run_mosac(inst, df, E, g, c, var, seed, w)
                    r = dict(seed=seed, text=text, control=ctrl, method=name, **inst.eval_groups(g, S), **inst.eval_users(p))
                    rows.append(r)
                    if ctrl and text == "T2" and var != "behaviour_only":
                        llm_users = set(df.user[df.llm.eq(1)]) - inst.spam
                        for u, s in p.items():
                            kind = "LLM spammer" if u in inst.spam else ("benign-LLM user" if u in llm_users else "other user")
                            scores.append(dict(seed=seed, method=name, kind=kind, p=s))
        host.emb.save()
        log(f"E2 seed {seed}:", pd.DataFrame([r for r in rows if r["seed"] == seed]).pivot_table(
            index="method", columns=["text", "control"], values="gF1").round(2).to_string())
    save(rows, out, "e2_text.csv"); save(scores, out, "e2_scores.csv")


def e3(host, cfg, sel, seeds, out, levels=(0.0, 0.1, 0.2, 0.3)):
    rows = []
    for ov in levels:
        c = apply_sel(with_overrides(cfg, inject__overlap=ov), sel)
        for seed in seeds:
            inst = P.Instance(host, c, seed); df, E, _ = inst.texts()
            gl, _ = inst.groups("Louvain", sel["best_w"]["Louvain"])
            S, p, _ = P.run_mosac(inst, df, E, gl, c, "no_propagation", seed, sel["best_w"]["Louvain"])
            rows.append(dict(overlap=ov, seed=seed, grouping="Louvain (disjoint)", propagation="single group",
                             **inst.eval_groups(gl, S), **inst.eval_users(p)))
            ge, _ = inst.groups("EgoSplit", sel["best_w"]["EgoSplit"])
            S, p, _ = P.run_mosac(inst, df, E, ge, c, "behaviour_only", seed, sel["best_w"]["EgoSplit"])
            rows.append(dict(overlap=ov, seed=seed, grouping="OSG-behav. (Ego-Splitting, no text)", propagation=c["stage3"]["propagation"],
                             **inst.eval_groups(ge, S), **inst.eval_users(p)))
            for algo in OVERLAP_ALGOS:
                w = sel["best_w"][algo]; g, _ = inst.groups(algo, w)
                for how in ("mean", "max", "noisy_or"):
                    cc = with_overrides(c, stage3__propagation=how)
                    S, p, _ = P.run_mosac(inst, df, E, g, cc, "full", seed, w)
                    rows.append(dict(overlap=ov, seed=seed, grouping=f"Overlapping ({algo})", propagation=how,
                                     **inst.eval_groups(g, S), **inst.eval_users(p)))
            host.emb.save()
            log(f"E3 overlap {ov} seed {seed} done")
    save(rows, out, "e3_overlap.csv")


ABLATIONS = [("full", "Full MOSAC-SGD"), ("no_semantic", "- semantic branch (z^s, phi_text)"),
             ("no_phi_text", "- group text features phi_text only"), ("concat", "- contrastive fusion (plain concatenation)"),
             ("overlap", "- overlap (Louvain instead)"), ("no_propagation", "- propagation (best single group)"),
             ("no_behaviour", "- behavioural branch (semantic only)"), ("svd", "GNN -> SVD embedding"),
             ("minmax", "Rank-norm -> min-max S(g)")]


def e4(host, cfg, sel, seeds, out):
    rows = []
    c = apply_sel(cfg, sel); a, w = sel["algo"], sel["best_w"][sel["algo"]]
    for seed in seeds:
        inst = P.Instance(host, c, seed); df, E, _ = inst.texts()
        for var, name in ABLATIONS:
            if var == "overlap":
                wl = sel["best_w"]["Louvain"]; g, _ = inst.groups("Louvain", wl)
                S, p, _ = P.run_mosac(inst, df, E, g, c, "full", seed, wl)
            else:
                g, _ = inst.groups(a, w); S, p, _ = P.run_mosac(inst, df, E, g, c, var, seed, w)
            rows.append(dict(seed=seed, variant=name, **inst.eval_groups(g, S), **inst.eval_users(p)))
        log(f"E4 seed {seed} done")
    save(rows, out, "e4_ablation.csv")


def e5(host, cfg, sel, seeds, out, quick=False):
    rows = []
    a = sel["algo"]

    def one(tag, value, c, seed, w=None, subsample=None, algo=None):
        algo = algo or a; w = sel["best_w"][algo] if w is None else w
        t0 = time.time(); inst = P.Instance(host, c, seed, subsample=subsample); t_inj = time.time() - t0
        t0 = time.time(); G = inst.graph(w); t_graph = time.time() - t0 + inst.timing["pairs"]
        g, t1 = inst.groups(algo, w)
        t0 = time.time(); df, E, _ = inst.texts(); t_emb = time.time() - t0
        t0 = time.time(); S, p, ex = P.run_mosac(inst, df, E, g, c, "full", seed, w); t23 = time.time() - t0
        r = dict(param=tag, value=value, seed=seed, reviews=len(inst.struct), edges=G.number_of_edges(),
                 t_graph=t_graph, t_stage1=t1, t_embed=t_emb, t_stage23=t23, **inst.eval_groups(g, S), **inst.eval_users(p))
        rows.append(r); host.emb.save()
        log(f"E5 {tag}={value} seed {seed}: gF1={r['gF1']:.3f} uAP={r['uAP']:.3f}")

    c0 = apply_sel(cfg, sel)
    for seed in seeds:
        for st in (1, 3, 7, 14):
            one("sigma_t", st, with_overrides(c0, graph__sigma_t=float(st)), seed)
        for w in cfg["graph"]["w_grid"]:
            one("w_min", w, c0, seed, w=w)
        for tau in (0.0, 0.25, 0.5, 0.75, 0.9):
            one("tau_g", tau, with_overrides(c0, stage3__tau_g=tau), seed)
        for K in ((10, 30) if quick else (10, 30, 50)):
            one("K", K, with_overrides(c0, inject__K=K), seed)
        for frac in ((0.5, 1.0) if quick else (0.25, 0.5, 1.0)):
            one("size", frac, c0, seed, subsample=frac)
        # harder campaigns: slower, sparser, less synchronised
        hard = with_overrides(c0, inject__window=[5, 30], inject__hit_prob=0.6)
        one("hard_campaigns", "window 5-30d, hit 0.6", hard, seed)
    save(rows, out, "e5_sensitivity.csv")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    seeds = cfg["run"]["seeds"][:2] if a.quick else cfg["run"]["seeds"]
    host_names = cfg["run"]["hosts"][:1] if a.quick else cfg["run"]["hosts"]
    out = a.out or (cfg["paths"]["results"] + ("_quick" if a.quick else ""))
    os.makedirs(out, exist_ok=True)
    todo = set(a.only or ["E1", "E2", "E3", "E4", "E5", "E6"])
    T = {}
    hosts = {h: P.Host(cfg, h) for h in host_names}
    primary = host_names[0]
    sels, clfs = {}, {}
    for hn, host in hosts.items():
        sp = os.path.join(out, f"selected_{hn}.json")
        o2 = os.path.join(out, f"_tune_{hn}"); os.makedirs(o2, exist_ok=True)
        sels[hn] = tune(host, cfg, o2)
        clfs[hn] = train_sccom(host, apply_sel(cfg, sels[hn]), sels[hn])
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(sels[hn], f, indent=1)
    steps = [("E1", lambda: e1(hosts, cfg, sels, clfs, seeds, out)),
             ("E2", lambda: e2(hosts[primary], cfg, sels[primary], seeds, out)),
             ("E3", lambda: e3(hosts[primary], cfg, sels[primary], seeds, out)),
             ("E4", lambda: e4(hosts[primary], cfg, sels[primary], seeds, out)),
             ("E5", lambda: e5(hosts[primary], cfg, sels[primary], seeds[:3], out, a.quick)),
             ("E6", lambda: (RD.main(cfg, out), RD.opspam_check(cfg, out)))]
    for name, fn in steps:
        if name in todo:
            t0 = time.time(); log(f"===== {name} ====="); fn(); T[name] = time.time() - t0
            log(f"===== {name} done in {T[name]:.0f}s =====")
            pd.Series(T).to_csv(os.path.join(out, "runtime_experiments.csv"))
    from . import report
    report.main(out)


if __name__ == "__main__":
    main()
