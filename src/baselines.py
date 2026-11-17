"""Simplified re-implementations ("-lite") of the baselines in the end-sem report.

  SC-Com-lite   collusiveness graph -> Louvain -> community features -> Random Forest
                (SUPERVISED: trained on the validation injection's labels)
  TKCCL-lite    per-item consecutive-reviewer vectors [dt, dr], cut where ||.|| > delta,
                segments merged greedily into groups, score = distance from the mean
  OSG-behav.    Ego-Splitting overlapping groups + behavioural group features only +
                the same Stage 3 (DRL-OSG-style: overlap without text)
  Behavioural IF  Isolation Forest on user features only (no graph, no text)
  Text-only     Stage 3 on phi_text and the user text vectors only (runs in pipeline.py)
"""
import collections
import numpy as np, pandas as pd, networkx as nx
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from .evaluate import jacc


def _entropy(x, bins):
    h, _ = np.histogram(x, bins=bins)
    p = h[h > 0] / h.sum() if h.sum() else np.array([1.0])
    return float(-(p * np.log(p)).sum())


# ------------------------------------------------------------------ Behavioural IF
def behavioural_if(ufeat, seed=0):
    X = StandardScaler().fit_transform(ufeat.values)
    s = -IsolationForest(n_estimators=300, random_state=seed).fit(X).score_samples(X)
    return dict(zip(ufeat.index, pd.Series(s).rank(pct=True).values))


# ------------------------------------------------------------------ SC-Com-lite
def sccom_features(groups, G, df):
    by_user = {u: g for u, g in df.groupby("user")}
    F = []
    for g in groups:
        sub = G.subgraph(g); n = len(g); m = sub.number_of_edges()
        rv = pd.concat([by_user[u] for u in g if u in by_user])
        F.append([n, 2 * m / (n * (n - 1)) if n > 1 else 0, 2 * m / n,
                  _entropy(rv.t.values, 20), _entropy(rv.rating.values, [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
                  np.mean([d["weight"] for *_, d in sub.edges(data=True)]) if m else 0])
    return np.array(F).reshape(len(groups), 6)


def sccom_label(groups, camps, thr=0.5):
    truth = [set(c["members"]) for c in camps]
    return np.array([max(jacc(g, t) for t in truth) >= thr for g in groups], int)


def sccom_fit(groups, G, df, camps, seed=0):
    X = sccom_features(groups, G, df); y = sccom_label(groups, camps)
    clf = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced")
    clf.fit(X, y) if len(set(y)) > 1 else None
    return clf if len(set(y)) > 1 else None


def sccom_score(clf, groups, G, df):
    if not groups:
        return np.zeros(0)
    X = sccom_features(groups, G, df)
    return clf.predict_proba(X)[:, 1] if clf is not None else np.zeros(len(groups))


# ------------------------------------------------------------------ TKCCL-lite
def tkccl_lite(df, delta=1.0, t_scale=7.0, min_size=3, merge_j=0.5, seed=0):
    """Temporal consistency: on each item sort reviews by time; consecutive reviews
    form vector [dt / t_scale, dr]; cut the sequence where the vector norm > delta.
    Segments (>= min_size users) are candidate local groups; segments are merged
    greedily across items when their user-set Jaccard >= merge_j (greedy cover).
    Group embedding = [mean dt, mean dr, log size, #items]; score = distance from
    the mean embedding. Disjoint: each user is kept only in its best-scored group."""
    segs = []
    for it, g in df.sort_values("t").groupby("item", sort=False):
        if len(g) < min_size:
            continue
        u, t, r = g.user.values, g.t.values, g.rating.values
        v = np.sqrt((np.diff(t) / t_scale) ** 2 + np.diff(r) ** 2)
        cut = np.flatnonzero(v > delta) + 1
        for s, e in zip(np.r_[0, cut], np.r_[cut, len(g)]):
            us = set(u[s:e])
            if len(us) >= min_size:
                dt = np.diff(t[s:e]).mean() if e - s > 1 else 0; dr = np.abs(np.diff(r[s:e])).mean() if e - s > 1 else 0
                segs.append([us, {it}, [dt], [dr]])
    segs.sort(key=lambda x: -len(x[0]))
    merged, index = [], collections.defaultdict(set)
    for us, its, dts, drs in segs:
        hit = None
        for j in set().union(*(index[x] for x in us)):
            if jacc(us, merged[j][0]) >= merge_j:
                hit = j; break
        if hit is None:
            merged.append([set(us), set(its), list(dts), list(drs)]); j = len(merged) - 1
        else:
            m = merged[hit]; m[0] |= us; m[1] |= its; m[2] += dts; m[3] += drs; j = hit
        for x in merged[j][0]:
            index[x].add(j)
    merged = [m for m in merged if len(m[1]) >= 2 and len(m[0]) <= 100]  # coordinated on >= 2 items
    if not merged:
        return [], np.zeros(0)
    E = np.array([[np.mean(m[2]), np.mean(m[3]), np.log(len(m[0])), len(m[1])] for m in merged])
    Z = StandardScaler().fit_transform(E)
    d = np.linalg.norm(Z - Z.mean(0), axis=1)  # distance from the mean embedding
    order = np.argsort(-d)
    used, groups, S = set(), [], []
    for i in order:  # disjoint greedy cover
        g = merged[i][0] - used
        if len(g) >= min_size:
            groups.append(g); S.append(d[i]); used |= g
    S = pd.Series(S).rank(pct=True).values
    return groups, S
