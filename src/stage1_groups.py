"""Stage 1: (overlapping) candidate-group discovery on the co-review graph.

Algorithms: BigCLAM, DEMON, SLPA (overlapping, ours), Ego-Splitting (overlapping,
DRL-OSG-style baseline) and Louvain (disjoint control). All are implemented here
in numpy / networkx / torch so the code needs no cdlib and runs on Windows.
"""
import collections, time
import numpy as np, pandas as pd, networkx as nx


# ------------------------------------------------------------------ algorithms
def louvain(G, seed=0, **_):
    return [set(c) for c in nx.community.louvain_communities(G, weight="weight", seed=seed)]


def slpa(G, seed=0, T=21, r=0.1, **_):
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes()); mem = {n: [n] for n in nodes}
    nbrs = {n: list(G.neighbors(n)) for n in nodes}
    wts = {n: np.array([G[n][m]["weight"] for m in nbrs[n]]) for n in nodes}
    for _ in range(T):
        for i in rng.permutation(len(nodes)):
            n = nodes[i]
            if not nbrs[n]:
                continue
            heard = collections.defaultdict(float)
            for m, w in zip(nbrs[n], wts[n]):
                heard[mem[m][rng.integers(len(mem[m]))]] += w
            mem[n].append(max(heard.items(), key=lambda kv: kv[1])[0])
    comms = collections.defaultdict(set)
    for n in nodes:
        cnt = collections.Counter(mem[n]); L = len(mem[n])
        for lab, f in cnt.items():
            if f / L >= r:
                comms[lab].add(n)
    return list(comms.values())


def demon(G, seed=0, eps=0.25, min_size=3, **_):
    """DEMON: label propagation in each ego-network (ego removed), add the ego back,
    then merge communities whose overlap (|A∩B| / |smaller|) is >= 1 - eps."""
    comms = []
    for n in G.nodes():
        ego = G.subgraph(G.neighbors(n))
        if ego.number_of_nodes() < 2:
            continue
        for c in nx.community.asyn_lpa_communities(ego, weight="weight", seed=seed):
            c = set(c) | {n}
            if len(c) >= min_size:
                comms.append(c)
    comms.sort(key=len, reverse=True); merged = []; index = collections.defaultdict(set)
    for c in comms:
        cand = set().union(*(index[x] for x in c))
        hit = None
        for j in sorted(cand):
            d = merged[j]
            if len(c & d) / min(len(c), len(d)) >= 1 - eps:
                hit = j; break
        if hit is None:
            merged.append(set(c)); j = len(merged) - 1
        else:
            merged[hit] |= c; j = hit
        for x in merged[j]:
            index[x].add(j)
    return merged


def ego_splitting(G, seed=0, **_):
    """Epasto et al. 2017: split every node into personas (one per local cluster
    of its ego-net; local clustering = weighted label propagation), connect
    personas along original edges, run a non-overlapping clustering (Louvain)
    on the persona graph, map personas back to nodes."""
    persona_of = {}  # (node, neighbour) -> persona id
    owner = []
    for n in G.nodes():
        ego = G.subgraph(G.neighbors(n))
        parts = (nx.community.asyn_lpa_communities(ego, weight="weight", seed=seed)
                 if ego.number_of_edges() else [{m} for m in ego.nodes()])
        for comp in parts:
            pid = len(owner); owner.append(n)
            for m in comp:
                persona_of[(n, m)] = pid
    P = nx.Graph()
    P.add_nodes_from(range(len(owner)))
    for a, b, d in G.edges(data=True):
        P.add_edge(persona_of[(a, b)], persona_of[(b, a)], weight=d["weight"])
    out = []
    for c in nx.community.louvain_communities(P, weight="weight", seed=seed):
        out.append({owner[p] for p in c})
    return out


def bigclam(G, seed=0, dims=150, iters=300, lr=0.05, **_):
    """BigCLAM (Yang & Leskovec 2013) by projected gradient ascent (torch, CPU).
    u in community c iff F_uc >= delta, delta = sqrt(-log(1 - eps)), eps = edge density."""
    import torch
    torch.manual_seed(seed); torch.set_num_threads(min(2, torch.get_num_threads()))
    nodes = list(G.nodes()); n = len(nodes)
    if n == 0:
        return []
    idx = {u: i for i, u in enumerate(nodes)}
    e = np.array([(idx[a], idx[b]) for a, b in G.edges()])
    src = torch.tensor(np.r_[e[:, 0], e[:, 1]]); dst = torch.tensor(np.r_[e[:, 1], e[:, 0]])
    k = min(dims, max(2, n // 3))
    # init (as in the paper): ego-nets of nodes with locally minimal conductance
    cond = {}
    vol_all = 2 * G.number_of_edges()
    for u in nodes:
        S = set(G.neighbors(u)) | {u}
        vol = sum(G.degree(x) for x in S); cut = nx.cut_size(G, S)
        cond[u] = cut / max(1, min(vol, vol_all - vol))
    seeds = [u for u in nodes if all(cond[u] <= cond[v] for v in G.neighbors(u))]
    seeds = sorted(seeds, key=lambda u: (cond[u], -G.degree(u)))[:k]
    F0 = np.random.default_rng(seed).random((n, k)) * 0.01
    for c, u in enumerate(seeds):
        for x in list(G.neighbors(u)) + [u]:
            F0[idx[x], c] = 1.0
    F = torch.tensor(F0, dtype=torch.float32).requires_grad_(True)
    opt = torch.optim.Adam([F], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        Fp = F.clamp(min=0)
        dot = (Fp[src] * Fp[dst]).sum(1)
        ll_edges = torch.log(-torch.expm1(-dot.clamp(min=1e-8))).sum() / 2
        s = Fp.sum(0)
        all_pairs = ((s * s).sum() - (Fp * Fp).sum()) / 2  # sum over u<v of F_u.F_v
        ll_non = all_pairs - dot.sum() / 2
        loss = -(ll_edges - ll_non)
        loss.backward(); opt.step()
        with torch.no_grad():
            F.clamp_(min=0)
    eps = 2 * G.number_of_edges() / (n * (n - 1)) if n > 1 else 1e-8
    delta = np.sqrt(-np.log(1 - eps))
    Fn = F.detach().numpy()
    out = []
    for c in range(k):
        mem = np.flatnonzero(Fn[:, c] >= delta)
        if len(mem):
            out.append({nodes[i] for i in mem})
    return out


ALGOS = {"Louvain": louvain, "SLPA": slpa, "DEMON": demon, "EgoSplit": ego_splitting, "BigCLAM": bigclam}
OVERLAPPING = {"SLPA", "DEMON", "EgoSplit", "BigCLAM"}


# ------------------------------------------------------------------ post-processing
def postprocess(groups, reviews_by_user, s1cfg):
    """Keep 3 <= |g| <= S_max, require >= min_shared_items items reviewed by >= 50% of
    members, de-duplicate groups with Jaccard > dedup_jaccard."""
    keep = []
    for g in groups:
        if not (s1cfg["min_size"] <= len(g) <= s1cfg["max_size"]):
            continue
        cnt = collections.Counter(it for u in g for it in reviews_by_user.get(u, ()))
        shared = sum(1 for v in cnt.values() if v >= max(2, 0.5 * len(g)))
        if shared >= s1cfg["min_shared_items"]:
            keep.append(frozenset(g))
    keep = sorted(set(keep), key=len, reverse=True)
    out, index = [], collections.defaultdict(list)
    for g in keep:
        dup = False
        for j in set(j for u in g for j in index[u]):
            h = out[j]
            if len(g & h) / len(g | h) > s1cfg["dedup_jaccard"]:
                dup = True; break
        if not dup:
            out.append(g)
            for u in g:
                index[u].append(len(out) - 1)
    return [set(g) for g in out]


def discover(name, G, reviews_by_user, cfg, seed=0):
    s1 = cfg["stage1"]
    params = {"SLPA": s1["slpa"], "DEMON": s1["demon"], "BigCLAM": s1["bigclam"]}.get(name, {})
    t0 = time.time()
    raw = ALGOS[name](G, seed=seed, **params) if G.number_of_edges() else []
    groups = postprocess(raw, reviews_by_user, s1)
    return groups, time.time() - t0
