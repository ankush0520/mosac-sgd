"""MOSAC-SGD mid-sem pilot: behavioural pipeline (co-review graph -> overlapping CD ->
group scoring -> overlap propagation) on a synthetic marketplace with injected,
overlapping collusive campaigns. Text/semantic branch is NOT part of this pilot.

Usage: python3 pilot.py OUT_DIR
"""
import sys, json, time, itertools, collections
import numpy as np, pandas as pd, networkx as nx
from scipy.stats import norm
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

DAY = 1.0
CFG = dict(n_users=4000, n_items=1200, horizon_days=730, sigma_t=7.0, sigma_r=1.0,
           w_min=0.05, min_common=2, cap=500, K=30, overlap_frac=0.25, k_groups=30, k_users=None)


# ---------------------------------------------------------------- data
def make_background(rng, c):
    pop = rng.zipf(1.6, c["n_items"]).astype(float); pop = np.minimum(pop, 400); pop /= pop.sum()
    item_q = np.clip(rng.normal(3.8, 0.6, c["n_items"]), 1.5, 5)
    rows = []
    act = np.minimum(rng.geometric(0.18, c["n_users"]) + 2, 60)  # users >= 3 reviews
    for u in range(c["n_users"]):
        items = rng.choice(c["n_items"], size=act[u], replace=False, p=pop)
        t0 = rng.uniform(0, c["horizon_days"])
        for it in items:
            t = (t0 + rng.normal(0, 120)) % c["horizon_days"]
            r = int(np.clip(np.rint(item_q[it] + rng.normal(0, 1.0)), 1, 5))
            rows.append((f"u{u}", f"i{it}", r, t, 0))
    # organic bursts (launch days / events): genuine crowds reviewing an item together
    for it in rng.choice(c["n_items"], 40, replace=False):
        day = rng.uniform(0, c["horizon_days"]); crowd = rng.choice(c["n_users"], rng.integers(8, 25), replace=False)
        for u in crowd:
            rows.append((f"u{u}", f"i{it}", int(np.clip(np.rint(item_q[it] + rng.normal(0, .8)), 1, 5)),
                         day + rng.uniform(0, 3), 0))
    return pd.DataFrame(rows, columns=["user", "item", "rating", "t", "spam"]), pop, item_q


def inject(rng, df, pop, item_q, c):
    """Protocol from the deck: K campaigns, 3-8 targets, 5-30 members, 1-10 day window,
    20-30% of spammers in 2-3 campaigns, 5-15 camouflage reviews each."""
    sizes = rng.integers(5, 31, c["K"])
    # primary membership: every spammer starts in exactly one campaign
    spammers, camp_members = [], [set() for _ in range(c["K"])]
    for k in range(c["K"]):
        for _ in range(sizes[k]):
            s = f"s{len(spammers)}"; spammers.append(s); camp_members[k].add(s)
    # overlap: overlap_frac of spammers also join 1-2 further campaigns (=> 2-3 campaigns in total)
    for s in rng.choice(spammers, int(c["overlap_frac"] * len(spammers)), replace=False):
        own = [k for k in range(c["K"]) if s in camp_members[k]][0]
        others = [k for k in range(c["K"]) if k != own]
        for k in rng.choice(others, rng.integers(1, 3), replace=False):
            camp_members[k].add(s)
    camps, rows = [], []
    for k in range(c["K"]):
        targets = rng.choice(c["n_items"], rng.integers(3, 9), replace=False)
        start = rng.uniform(0, c["horizon_days"] - 10); win = rng.uniform(1, 10)
        promote = rng.random() < 0.8
        for s in camp_members[k]:
            for it in targets:
                if rng.random() < 0.85:  # not every member hits every target
                    r = 5 if promote else 1
                    if rng.random() < 0.15: r = 4 if promote else 2
                    rows.append((s, f"i{it}", r, start + rng.uniform(0, win), 1))
        camps.append(dict(id=k, members=sorted(camp_members[k]), items=[f"i{t}" for t in targets],
                          start=start, window=win))
    for s in spammers:  # camouflage
        for it in rng.choice(c["n_items"], rng.integers(5, 16), replace=False, p=pop):
            rows.append((s, f"i{it}", int(np.clip(np.rint(item_q[it] + rng.normal(0, 1)), 1, 5)),
                         rng.uniform(0, c["horizon_days"]), 0))
    inj = pd.DataFrame(rows, columns=df.columns)
    return pd.concat([df, inj], ignore_index=True), camps


# ---------------------------------------------------------------- step 0
def build_graph(df, c, rng):
    edges = {}
    for it, g in df.groupby("item"):
        if len(g) > c["cap"]:
            g = g.sample(c["cap"], random_state=0)
        u, t, r = g.user.values, g.t.values, g.rating.values
        if len(g) < 2: continue
        a, b = np.triu_indices(len(g), 1)
        sim = 4 * norm.cdf(-np.abs(t[a] - t[b]) / c["sigma_t"]) * norm.cdf(-np.abs(r[a] - r[b]) / c["sigma_r"])
        keep = (sim >= c["w_min"]) & (u[a] != u[b])
        for x, y, s in zip(u[a][keep], u[b][keep], sim[keep]):
            key = (x, y) if x < y else (y, x)
            e = edges.setdefault(key, [0.0, 0])
            e[0] = max(e[0], s); e[1] += 1
    # keep a pair only if it co-reviewed >= min_common items within the similarity window
    G = nx.Graph(); G.add_weighted_edges_from((x, y, s) for (x, y), (s, n) in edges.items()
                                              if n >= c.get("min_common", 1))
    return G


# ---------------------------------------------------------------- stage 1
def slpa(G, T=21, r=0.1, seed=0):
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes()); mem = {n: [n] for n in nodes}
    nbrs = {n: list(G.neighbors(n)) for n in nodes}
    wts = {n: np.array([G[n][m]["weight"] for m in nbrs[n]]) for n in nodes}
    for _ in range(T):
        for n in rng.permutation(nodes):
            if not nbrs[n]: continue
            heard = collections.Counter()
            for m, w in zip(nbrs[n], wts[n]):
                heard[mem[m][rng.integers(len(mem[m]))]] += w
            mem[n].append(max(heard.items(), key=lambda kv: kv[1])[0])
    comms = collections.defaultdict(set)
    for n in nodes:
        cnt = collections.Counter(mem[n]); L = len(mem[n])
        for lab, f in cnt.items():
            if f / L >= r: comms[lab].add(n)
    return dedup([c for c in comms.values()])


def demon(G, eps=0.25, min_size=3, seed=0):
    comms = []
    for n in G.nodes():
        ego = G.subgraph(G.neighbors(n))
        if ego.number_of_nodes() < 2: continue
        for c in nx.community.asyn_lpa_communities(ego, weight="weight", seed=seed):
            c = set(c) | {n}
            if len(c) >= min_size: comms.append(c)
    # merge bottom-up: merge c into an existing community if |c ∩ d| / |smaller| >= eps... use
    # the DEMON rule on the smaller set; iterate until stable (sorted by size for speed)
    comms.sort(key=len, reverse=True); merged = []
    index = collections.defaultdict(set)
    for c in comms:
        cand = set().union(*(index[x] for x in c)) if c else set()
        hit = None
        for j in cand:
            d = merged[j]
            if len(c & d) / min(len(c), len(d)) >= 1 - eps:  # strict merge to avoid giant blobs
                hit = j; break
        if hit is None:
            merged.append(set(c)); j = len(merged) - 1
        else:
            merged[hit] |= c; j = hit
        for x in merged[j]: index[x].add(j)
    return dedup(merged)


def louvain(G, seed=0):
    return [set(c) for c in nx.community.louvain_communities(G, weight="weight", seed=seed)]


def dedup(comms):
    seen, out = set(), []
    for c in comms:
        k = frozenset(c)
        if len(c) >= 3 and k not in seen:
            seen.add(k); out.append(set(c))
    return out


# ---------------------------------------------------------------- stage 3
def group_features(groups, df, G):
    by_user = {u: g for u, g in df.groupby("user")}
    feats = []
    for g in groups:
        sub = G.subgraph(g); m = sub.number_of_edges(); n = len(g)
        dens = 2 * m / (n * (n - 1)) if n > 1 else 0
        mw = np.mean([d["weight"] for *_, d in sub.edges(data=True)]) if m else 0
        rv = pd.concat([by_user[u] for u in g if u in by_user])
        ic = rv.groupby("item").user.nunique()
        shared = ic[ic >= max(2, 0.5 * n)].index
        sr = rv[rv.item.isin(shared)]
        span = (sr.groupby("item").t.std().mean() if len(shared) else 365.0)
        ext = (np.abs(sr.rating - 3).mean() if len(shared) else 0)
        feats.append([np.log(n), dens, mw, len(shared), np.log1p(span if span == span else 365), ext])
    return np.array(feats)


def score_groups(X, seed):
    if len(X) < 5: return np.zeros(len(X))
    s = -IsolationForest(n_estimators=300, random_state=seed).fit(X).score_samples(X)
    return pd.Series(s).rank(pct=True).values


def propagate(groups, sg, how):
    mem = collections.defaultdict(list)
    for g, s in zip(groups, sg):
        for u in g: mem[u].append(s)
    out = {}
    for u, s in mem.items():
        s = np.array(s)
        out[u] = 1 - np.prod(1 - s) if how == "noisy_or" else s.max() if how == "max" else s.mean()
    return out


def propagate_weighted(groups, sg, theta=0.8):
    """Deck's fix for noisy-OR inflation: propagate only from groups with s_g >= theta,
    exponent w_g = 1/log(1+|g|).  Users with no such group fall back to max."""
    mem = collections.defaultdict(list); best = collections.defaultdict(float)
    for g, s in zip(groups, sg):
        w = 1 / np.log(1 + len(g))
        for u in g:
            best[u] = max(best[u], s)
            if s >= theta: mem[u].append((s, w))
    out = {}
    for u in best:
        if mem[u]:
            out[u] = 1 - np.prod([(1 - s) ** w for s, w in mem[u]])
        else:
            out[u] = best[u] * theta  # below every propagated score
    return out


# ---------------------------------------------------------------- metrics
def jacc(a, b): return len(a & b) / len(a | b)


def eval_groups(groups, sg, camps, k):
    truth = [set(c["members"]) for c in camps]
    order = np.argsort(-sg)[:k]; top = [groups[i] for i in order]
    p = np.mean([max(jacc(g, t) for t in truth) >= 0.5 for g in top]) if top else 0
    r = np.mean([any(jacc(g, t) >= 0.5 for g in top) for t in truth])
    rec_all = np.mean([any(jacc(g, t) >= 0.5 for g in groups) for t in truth])
    f1 = 2 * p * r / (p + r) if p + r else 0
    return p, r, f1, rec_all


def eval_membership(groups, camps):
    """MR: over (multi-campaign spammer, campaign) pairs, share recovered by a detected group
    containing the user with J>=0.5 to that campaign. MMR: share of multi-campaign spammers
    appearing in >=2 groups that each match a distinct true campaign."""
    truth = [set(c["members"]) for c in camps]
    cu = collections.defaultdict(list)
    for i, t in enumerate(truth):
        for u in t: cu[u].append(i)
    multi = {u: cs for u, cs in cu.items() if len(cs) >= 2}
    match = collections.defaultdict(set)  # campaign -> detected groups matching it
    for j, g in enumerate(groups):
        for i, t in enumerate(truth):
            if jacc(g, t) >= 0.5: match[i].add(j)
    hit = tot = 0; mm = 0
    for u, cs in multi.items():
        got = [i for i in cs if any(u in groups[j] for j in match[i])]
        hit += len(got); tot += len(cs); mm += len(got) >= 2
    return hit / tot, mm / len(multi), len(multi)


def eval_users(scores, spam_users, all_users, k):
    y = np.array([u in spam_users for u in all_users]); s = np.array([scores.get(u, 0) for u in all_users])
    top = np.argsort(-s)[:k]
    return y[top].mean(), y[top].sum() / y.sum(), average_precision_score(y, s)


def recall_by_ncamp(scores, camps, all_users, k):
    """Recall in the top-k users, split by how many campaigns a spammer belongs to."""
    cnt = collections.Counter(u for c in camps for u in c["members"])
    s = np.array([scores.get(u, 0) for u in all_users])
    top = set(np.array(all_users)[np.argsort(-s)[:k]])
    out = {}
    for n in (1, 2, 3):
        us = [u for u, m in cnt.items() if m == n]
        out[f"R_c{n}"] = np.mean([u in top for u in us]) if us else np.nan
    return out


def user_if_baseline(df, seed):
    item_mean = df.groupby("item").rating.mean()
    d = df.assign(dev=(df.rating - df.item.map(item_mean)).abs())
    first = df.groupby("item").t.transform(lambda x: x.rank(pct=True))
    d["early"] = first < 0.1
    def burst(t):
        t = np.sort(t.values); return np.mean(np.diff(t) < 2) if len(t) > 1 else 0
    f = d.groupby("user").agg(n=("item", "size"), dev=("dev", "mean"), ext=("rating", lambda r: np.mean((r == 5) | (r == 1))),
                              early=("early", "mean"))
    f["burst"] = d.groupby("user").t.apply(burst)
    s = -IsolationForest(n_estimators=300, random_state=seed).fit(f.values).score_samples(f.values)
    return dict(zip(f.index, pd.Series(s).rank(pct=True).values))


# ---------------------------------------------------------------- run
ALGOS = [("Louvain (disjoint)", louvain), ("SLPA", slpa), ("DEMON", demon)]
W_GRID = [0.05, 0.3, 0.6, 0.9]


def make_data(seed, c):
    rng = np.random.default_rng(seed)
    bg, pop, q = make_background(rng, c)
    df, camps = inject(rng, bg, pop, q, c)
    return df, camps, rng


def stage1(name, fn, G, seed):
    t0 = time.time(); groups = [g for g in fn(G, seed=seed) if 3 <= len(g) <= 200]
    return groups, time.time() - t0


def tune(c, seed=100):
    """Pick w_min per algorithm on a separate validation injection (seed 100)."""
    df, camps, rng = make_data(seed, c); rows = []
    for w in W_GRID:
        G = build_graph(df, dict(c, w_min=w), rng)
        for name, fn in ALGOS:
            groups, ts = stage1(name, fn, G, seed)
            sg = score_groups(group_features(groups, df, G), seed) if groups else np.array([])
            gp, gr, gf, gra = eval_groups(groups, sg, camps, c["k_groups"]) if groups else (0, 0, 0, 0)
            mr, mmr, _ = eval_membership(groups, camps) if groups else (0, 0, 0)
            rows.append(dict(w_min=w, method=name, edges=G.number_of_edges(), n_groups=len(groups),
                             gF1=gf, gRecAll=gra, MR=mr))
            print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    best = {m: float(t[t.method == m].sort_values(["gF1", "MR"], ascending=False).iloc[0].w_min) for m, _ in ALGOS}
    return t, best


def run(seed, c, best_w):
    df, camps, rng = make_data(seed, c)
    spam_users = set().union(*[set(x["members"]) for x in camps])
    users = sorted(df.user.unique()); k_users = len(spam_users)
    stats = dict(seed=seed, users=len(users), items=df.item.nunique(), reviews=len(df),
                 injected=int(df.spam.sum()), spammers=len(spam_users),
                 multi=sum(1 for u in spam_users if sum(u in x["members"] for x in camps) >= 2))
    graphs = {}
    for w in sorted(set(best_w.values())):
        t0 = time.time(); graphs[w] = build_graph(df, dict(c, w_min=w), rng)
        stats[f"edges_w{w}"] = graphs[w].number_of_edges(); stats[f"t_graph_w{w}"] = time.time() - t0
    res = []
    ub = user_if_baseline(df, seed)
    up, ur, ua = eval_users(ub, spam_users, users, k_users)
    res.append(dict(seed=seed, method="Behavioural IF (no graph)", prop="--", w_min=np.nan, n_groups=np.nan,
                    gP=np.nan, gR=np.nan, gF1=np.nan, gRecAll=np.nan, MR=np.nan, MMR=np.nan,
                    uP=up, uR=ur, uAP=ua, t_s1=np.nan, **recall_by_ncamp(ub, camps, users, k_users)))
    for name, fn in ALGOS:
        G = graphs[best_w[name]]
        groups, ts = stage1(name, fn, G, seed)
        X = group_features(groups, df, G); sg = score_groups(X, seed)
        gp, gr, gf, gra = eval_groups(groups, sg, camps, c["k_groups"])
        mr, mmr, _ = eval_membership(groups, camps)
        props = ["single"] if name.startswith("Louvain") else ["noisy_or", "noisy_or_w", "max", "mean"]
        for how in props:
            us = (propagate_weighted(groups, sg) if how == "noisy_or_w"
                  else propagate(groups, sg, "max" if how == "single" else how))
            up, ur, ua = eval_users(us, spam_users, users, k_users)
            res.append(dict(seed=seed, method=name, prop=how, w_min=best_w[name], n_groups=len(groups),
                            gP=gp, gR=gr, gF1=gf, gRecAll=gra, MR=mr, MMR=mmr, uP=up, uR=ur, uAP=ua, t_s1=ts,
                            **recall_by_ncamp(us, camps, users, k_users)))
    return stats, res, camps


if __name__ == "__main__":
    out = sys.argv[1]; import os; os.makedirs(out, exist_ok=True)
    allstats, allres = [], []
    tun, best_w = tune(CFG); tun.to_csv(f"{out}/validation_wmin_sweep.csv", index=False)
    print("best w_min:", best_w, flush=True)
    for seed in range(5):
        st, rs, camps = run(seed, CFG, best_w); allstats.append(st); allres += rs
        print(st, flush=True)
        for r in rs: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        if seed == 0: json.dump(camps, open(f"{out}/campaigns_seed0.json", "w"), indent=1)
    pd.DataFrame(allstats).to_csv(f"{out}/dataset_stats.csv", index=False)
    pd.DataFrame(allres).to_csv(f"{out}/results_raw.csv", index=False)
    json.dump(dict(CFG, best_w_min=best_w), open(f"{out}/config.json", "w"), indent=1)
