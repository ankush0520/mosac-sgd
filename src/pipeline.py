"""One benchmark instance: host + seed + injection settings -> every method's scores.

Used by src/run.py (experiments E1-E5). Expensive, text-independent parts (pair
table, graphs, Stage-1 groups, user features) are computed once per instance and
reused across text conditions and ablations.
"""
import time, collections
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from . import graph as GR, stage1_groups as S1, stage2_embed as S2, stage3_detect as S3, evaluate as EV
from . import baselines as BL, inject as INJ
from .data_prep import load_host


class Host:
    """Host data + LLM pool + embedder, loaded once per host."""

    def __init__(self, cfg, host):
        self.name = host
        self.df, self.pool = load_host(cfg, host)
        self.llm = INJ.load_llm_pool(cfg, host)
        self.briefs = INJ.briefs_from_pool(self.llm)
        self.emb = S2.Embedder(cfg["stage2"]["sbert"], f"{cfg['paths']['cache']}/sbert_{host}.npz")
        self.sent = S2.sentiment_axis(self.emb)


class Instance:
    def __init__(self, host, cfg, seed, subsample=None):
        self.host, self.cfg, self.seed = host, cfg, seed
        df = host.df
        if subsample:  # E5 scalability: random subset of items
            rng = np.random.default_rng(seed); items = df.item.unique()
            df = df[df.item.isin(rng.choice(items, int(len(items) * subsample), replace=False))]
        t0 = time.time()
        self.struct, self.camps, self.info = INJ.inject_structure(df, host.pool, host.briefs, cfg, seed)
        self.spam = set().union(*[set(c["members"]) for c in self.camps])
        self.users = sorted(self.struct.user.unique())
        self.by_user = {u: set(g.item) for u, g in self.struct.groupby("user")}
        self.timing = dict(inject=time.time() - t0)
        t0 = time.time()
        g = cfg["graph"]
        self.pt = GR.pair_table(self.struct, g["sigma_t"], g["sigma_r"], g["cap"], seed, floor=min(g["w_grid"] + [g["w_min"]]))
        self.timing["pairs"] = time.time() - t0
        self._graphs, self._groups, self._ufeat = {}, {}, {}

    # ---------------------------------------------------------------- cached pieces
    def graph(self, w):
        if w not in self._graphs:
            t0 = time.time()
            self._graphs[w] = GR.graph_from_pairs(*self.pt, w, self.cfg["graph"]["min_common"])
            self.timing[f"graph_w{w}"] = time.time() - t0 + self.timing["pairs"]
        return self._graphs[w]

    def groups(self, algo, w):
        key = (algo, w)
        if key not in self._groups:
            self._groups[key] = S1.discover(algo, self.graph(w), self.by_user, self.cfg, self.seed)
        return self._groups[key]

    def ufeat(self, w):
        if w not in self._ufeat:
            self._ufeat[w] = S2.user_features(self.struct, self.graph(w))
        return self._ufeat[w]

    def texts(self, text=None, control=None):
        df, n_ctrl = INJ.apply_text(self.struct, self.camps, self.host.llm, self.cfg, self.seed, self.host.name, text, control)
        E = self.host.emb.encode(df.text.tolist())
        return df, E, n_ctrl

    # ---------------------------------------------------------------- evaluation helpers
    def eval_groups(self, groups, S):
        k = self.cfg["eval"]["k_groups"]
        r = EV.group_metrics(groups, S, self.camps, k, self.cfg["eval"]["jaccard"])
        r.update(EV.membership_metrics(groups, self.camps, self.cfg["eval"]["jaccard"]))
        return r

    def eval_users(self, scores):
        k = len(self.spam)
        r = EV.user_metrics(scores, self.spam, self.users, k)
        r.update(EV.recall_by_ncamp(scores, self.camps, self.users, k))
        return r

    def stage1_row(self, algo, w):
        groups, ts = self.groups(algo, w)
        truth = [set(c["members"]) for c in self.camps]
        r = dict(algo=algo, w_min=w, n_groups=len(groups), t_stage1=ts,
                 ONMI=EV.onmi_lfk(truth, groups, set(self.users)) if groups else 0.0,
                 gRecAll=EV.group_metrics(groups, np.zeros(len(groups)), self.camps, 1)["gRecAll"] if groups else 0.0)
        r.update(EV.membership_metrics(groups, self.camps))
        return r


# ------------------------------------------------------------------ MOSAC-SGD and variants
def group_vectors(groups, users_idx, U, phi_parts, pca_dims, seed=0):
    """h_g = [PCA(mean_{u in g} U_u) ; phi parts...]. Returns (H, signs)."""
    if not groups:
        return np.zeros((0, 1)), np.zeros(1)
    parts = []
    if U is not None:
        M = np.array([U[[users_idx[u] for u in g]].mean(0) for g in groups])
        if M.shape[1] > pca_dims and len(groups) > pca_dims:
            M = PCA(pca_dims, random_state=seed).fit_transform(M)
        parts.append(M)
    signs = [np.zeros(parts[0].shape[1])] if parts else []
    for p in phi_parts:
        parts.append(p)
        signs.append(np.array(S2.GroupFeaturizer.BEH_SIGN if p.shape[1] == len(S2.GroupFeaturizer.BEH)
                              else S2.GroupFeaturizer.TXT_SIGN, float))
    return np.hstack(parts), np.concatenate(signs)


def run_mosac(inst, df, E, groups, cfg, variant="full", seed=0, w=None, need_encoders=True):
    """Return (S(g), p_u, extra) for MOSAC-SGD or one of its ablation / single-view variants.
    variants: full | behaviour_only | text_only | no_semantic | no_phi_text | concat |
              no_behaviour | svd | minmax | no_propagation"""
    s2, s3 = dict(cfg["stage2"]), dict(cfg["stage3"])
    if variant == "svd":
        s2["encoder"] = "svd"
    G = inst.graph(w)
    uf = inst.ufeat(w)
    gusers = sorted(set().union(*groups)) if groups else []
    users_idx = {u: i for i, u in enumerate(gusers)}
    # one featurizer per text condition (cached on df identity)
    key = (id(df), w)
    if getattr(inst, "_fz_key", None) != key:
        inst._fz = S2.GroupFeaturizer(df, G, E, inst.host.sent, inst.host.name, seed)
        inst._fz_key = key; inst._phi = {}
    gk = tuple(sorted(map(tuple, (sorted(g) for g in groups))))
    if gk not in inst._phi:
        inst._phi[gk] = inst._fz.all(groups)
    PB, PT = inst._phi[gk]
    extra = {}
    X = StandardScaler().fit_transform(uf.loc[gusers].values) if gusers else np.zeros((0, uf.shape[1]))
    # text vector s_u = mean review embedding
    rows = df.reset_index(drop=True).groupby("user").indices
    Su = np.array([E[rows[u]].mean(0) for u in gusers]) if gusers else np.zeros((0, E.shape[1]))
    if variant in ("behaviour_only",):
        H = group_vectors(groups, users_idx, X, [PB], s2["pca_dims"], seed)
    elif variant in ("text_only", "no_behaviour"):
        if variant == "text_only":
            H = group_vectors(groups, users_idx, Su, [PT], s2["pca_dims"], seed)
        else:
            zb, zs, info = S2.train_encoders(X, Su, G, gusers, s2, seed); extra.update(info)
            H = group_vectors(groups, users_idx, zs, [PT], s2["pca_dims"], seed)
    elif variant == "concat":
        H = group_vectors(groups, users_idx, np.hstack([X, Su]), [PB, PT], s2["pca_dims"], seed)
    else:
        t0 = time.time()
        zb, zs, info = S2.train_encoders(X, Su, G, gusers, s2, seed) if gusers else (np.zeros((0, 1)),) * 2 + ({},)
        extra.update(info); extra["t_encoder"] = time.time() - t0
        if variant == "no_semantic":
            H = group_vectors(groups, users_idx, zb, [PB], s2["pca_dims"], seed)
        elif variant == "no_phi_text":
            H = group_vectors(groups, users_idx, np.hstack([zb, zs]), [PB], s2["pca_dims"], seed)
        else:
            H = group_vectors(groups, users_idx, np.hstack([zb, zs]), [PB, PT], s2["pca_dims"], seed)
    H, signs = H
    S = S3.score_groups(H, s3["scorer"], rank_norm=(variant != "minmax"), seed=seed, signs=signs)
    how = "max" if variant == "no_propagation" else s3["propagation"]
    p = S3.propagate(groups, S, how, s3["tau_g"] if how != "max" else 0.0)
    extra["PT"] = PT; extra["PB"] = PB
    return S, p, extra
