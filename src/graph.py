"""Co-review graph (ColluEagle similarity), vectorised.

sigma_ijp = 4 * Phi(-|t_ip - t_jp| / sigma_t) * Phi(-|r_ip - r_jp| / sigma_r)
w_ij      = max_p sigma_ijp ; keep (i,j) iff w_ij >= w_min and the pair has
            >= min_common items with sigma_ijp >= w_min.
"""
import numpy as np, pandas as pd, networkx as nx
from scipy.special import ndtr


def pair_table(df, sigma_t, sigma_r, cap, seed=0, floor=0.05):
    """All same-item reviewer pairs with similarity >= floor.
    Returns DataFrame(u, v, w, item) with integer user codes u < v, plus the user index."""
    users, uid = np.unique(df.user.values, return_inverse=True)
    d = pd.DataFrame(dict(u=uid, item=df.item.values, t=df.t.values.astype(float), r=df.rating.values.astype(float)))
    rng = np.random.default_rng(seed)
    parts = []
    for it, g in d.groupby("item", sort=False):
        n = len(g)
        if n < 2:
            continue
        if n > cap:
            g = g.iloc[rng.choice(n, cap, replace=False)]; n = cap
        a, b = np.triu_indices(n, 1)
        u, t, r = g.u.values, g.t.values, g.r.values
        s = 4 * ndtr(-np.abs(t[a] - t[b]) / sigma_t) * ndtr(-np.abs(r[a] - r[b]) / sigma_r)
        k = s >= floor
        if not k.any():
            continue
        x, y = u[a[k]], u[b[k]]
        parts.append(pd.DataFrame(dict(u=np.minimum(x, y), v=np.maximum(x, y), w=s[k], item=it)))
    pt = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["u", "v", "w", "item"])
    pt = pt[pt.u != pt.v]
    return pt, users


def graph_from_pairs(pt, users, w_min, min_common=1):
    p = pt[pt.w >= w_min]
    agg = p.groupby(["u", "v"]).w.agg(["max", "size"]).reset_index()
    agg = agg[agg["size"] >= min_common]
    G = nx.Graph()
    G.add_weighted_edges_from(zip(users[agg.u.values], users[agg.v.values], agg["max"].values))
    return G


def build_graph(df, gcfg, seed=0, w_min=None, pt=None):
    pt_, users = pair_table(df, gcfg["sigma_t"], gcfg["sigma_r"], gcfg["cap"], seed) if pt is None else pt
    return graph_from_pairs(pt_, users, gcfg["w_min"] if w_min is None else w_min, gcfg["min_common"])
