"""Stage 2: behavioural branch, semantic branch, label-free fusion, group features.

  * SBERT (all-MiniLM-L6-v2) review embeddings, cached on disk by text hash.
  * user behavioural features x_u, GraphSAGE encoder (pure torch, CPU) or SVD fallback.
  * user text vector s_u = mean review embedding, MLP encoder.
  * symmetric InfoNCE between the two views (no labels).
  * group features phi_beh(g) and phi_text(g) = (CS, AO, SP).
"""
import hashlib, os, collections, itertools
import numpy as np, pandas as pd
from scipy import sparse
from . import aspects as A


# ------------------------------------------------------------------ text embeddings
class Embedder:
    """SBERT with a persistent cache keyed by sha1(text)."""

    def __init__(self, model_name, cache_path):
        self.model_name, self.path = model_name, cache_path
        self.keys, self.vecs, self.index = [], np.zeros((0, 384), np.float32), {}
        if os.path.exists(cache_path):
            z = np.load(cache_path, allow_pickle=False)
            self.keys = list(z["keys"]); self.vecs = z["vecs"]
            self.index = {k: i for i, k in enumerate(self.keys)}
        self.model = None
        self.dirty = False

    @staticmethod
    def h(t):
        return hashlib.sha1(t.encode("utf-8")).hexdigest()

    def encode(self, texts):
        hs = [self.h(t) for t in texts]
        missing = {}
        for h, t in zip(hs, texts):
            if h not in self.index and h not in missing:
                missing[h] = t
        if missing:
            if self.model is None:
                os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name, device="cpu")
            mt = list(missing.values())
            print(f"    SBERT: embedding {len(mt)} new texts", flush=True)
            v = self.model.encode(mt, batch_size=128, normalize_embeddings=True, show_progress_bar=len(mt) > 5000)
            base = len(self.keys)
            self.keys += list(missing); self.vecs = np.vstack([self.vecs, v.astype(np.float32)])
            for i, h in enumerate(missing):
                self.index[h] = base + i
            self.dirty = True
        return self.vecs[[self.index[h] for h in hs]]

    def save(self):
        if self.dirty:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            np.savez(self.path, keys=np.array(self.keys), vecs=self.vecs)
            self.dirty = False


def sentiment_axis(emb):
    pos = emb.encode(["This is excellent, I love it, highly recommended.", "Great quality, works perfectly, very happy."])
    neg = emb.encode(["This is terrible, I hate it, do not buy.", "Poor quality, broke immediately, very disappointed."])
    a = pos.mean(0) - neg.mean(0)
    return a / np.linalg.norm(a)


# ------------------------------------------------------------------ user features
def burstiness(t):
    t = np.sort(t)
    if len(t) < 3:
        return 0.0
    g = np.diff(t); m, s = g.mean(), g.std()
    return (s - m) / (s + m) if s + m > 0 else 0.0


def user_features(df, G):
    item_mean = df.groupby("item").rating.transform("mean")
    rank = df.groupby("item").t.rank(pct=True)
    d = df.assign(dev=(df.rating - item_mean).abs(), early=(rank <= 0.1).astype(float),
                  ext=df.rating.isin([1, 5]).astype(float))
    f = d.groupby("user").agg(n=("item", "size"), ext=("ext", "mean"), dev=("dev", "mean"), early=("early", "mean"))
    f["burst"] = d.groupby("user").t.apply(lambda x: burstiness(x.values))
    f["n"] = np.log1p(f.n)
    mw = {u: np.mean([G[u][v]["weight"] for v in G.neighbors(u)]) if G.degree(u) else 0.0 for u in G.nodes()}
    f["coweight"] = f.index.map(lambda u: mw.get(u, 0.0))
    return f


# ------------------------------------------------------------------ encoders
def _norm_adj(G, users):
    idx = {u: i for i, u in enumerate(users)}
    rows, cols = [], []
    for a, b in G.edges():
        if a in idx and b in idx:
            rows += [idx[a], idx[b]]; cols += [idx[b], idx[a]]
    n = len(users)
    A_ = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))
    deg = np.asarray(A_.sum(1)).ravel(); deg[deg == 0] = 1
    return sparse.diags(1 / deg) @ A_, A_


def _to_torch(m):
    import torch
    m = m.tocoo()
    return torch.sparse_coo_tensor(np.vstack([m.row, m.col]), m.data.astype(np.float32), m.shape).coalesce()


def train_encoders(X, S, G, users, s2, seed=0):
    """Returns z_b (n,d), z_s (n,d), info dict. X: standardised user features, S: s_u."""
    import torch, torch.nn as nn, torch.nn.functional as F
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(min(2, torch.get_num_threads()))  # small graphs: threading overhead dominates
    n = len(users); d, hdim = s2["dim"], s2["hidden"]
    Anorm, Araw = _norm_adj(G, users)
    if s2["encoder"] == "svd":
        from sklearn.decomposition import TruncatedSVD
        k = min(d, max(2, n - 1))
        W = Araw + sparse.eye(n) * 1e-3
        U = TruncatedSVD(k, random_state=seed).fit_transform(W)
        zb_fixed = torch.tensor(np.hstack([U, X]), dtype=torch.float32)
    At = _to_torch(Anorm)
    Xt = torch.tensor(X, dtype=torch.float32); St = torch.tensor(S, dtype=torch.float32)

    class SAGE(nn.Module):
        def __init__(s, i):
            super().__init__()
            s.l1s, s.l1n = nn.Linear(i, hdim), nn.Linear(i, hdim, bias=False)
            s.l2s, s.l2n = nn.Linear(hdim, d), nn.Linear(hdim, d, bias=False)

        def forward(s, x):
            h = F.relu(s.l1s(x) + s.l1n(torch.sparse.mm(At, x)))
            return s.l2s(h) + s.l2n(torch.sparse.mm(At, h))

    if s2["encoder"] == "svd":
        fb = nn.Sequential(nn.Linear(zb_fixed.shape[1], hdim), nn.ReLU(), nn.Linear(hdim, d))
        enc_b = lambda: fb(zb_fixed)
        params_b = list(fb.parameters())
    else:
        fb = SAGE(X.shape[1]); enc_b = lambda: fb(Xt); params_b = list(fb.parameters())
    fs = nn.Sequential(nn.Linear(S.shape[1], hdim), nn.ReLU(), nn.Linear(hdim, d))
    if s2["fusion"] == "concat":  # no contrastive training: random-init projections
        with torch.no_grad():
            return enc_b().numpy(), fs(St).numpy(), dict(loss=float("nan"), epochs=0)
    opt = torch.optim.Adam(params_b + list(fs.parameters()), lr=s2["lr"])
    B = min(s2["batch"], n); loss_v = float("nan")
    for ep in range(s2["epochs"]):
        opt.zero_grad()
        zb = F.normalize(enc_b(), dim=1); zs = F.normalize(fs(St), dim=1)
        bi = torch.randperm(n)[:B]
        logits = zb[bi] @ zs[bi].T / s2["tau"]
        lab = torch.arange(len(bi))
        loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))
        loss.backward(); opt.step(); loss_v = float(loss)
    with torch.no_grad():
        return enc_b().numpy(), fs(St).numpy(), dict(loss=loss_v, epochs=s2["epochs"])


# ------------------------------------------------------------------ group features
class GroupFeaturizer:
    BEH = ["log_size", "density", "mean_w", "n_shared", "log_span", "rating_var", "extremity", "target_conc"]
    TXT = ["CS", "AO", "SP"]
    # a-priori suspicious direction (used only by the "directional" scorer; no labels)
    # A-priori suspicious direction, used only by the "directional" scorer (no labels).
    # Signed features follow the group-spam indicators of Mukherjee et al. (WWW 2012):
    # GS size (+), GSUP support / shared items (+), GTW time window (-), GD rating
    # extremity (+), GCS content similarity (+; here CS, AO, SP). Density, mean edge
    # weight, rating variance and target concentration have no agreed direction
    # (camouflage reverses them), so they are left unsigned (0) and enter through
    # the Mahalanobis term together with the embedding dimensions.
    BEH_SIGN = [1, 0, 0, 1, -1, 0, 1, 0]
    TXT_SIGN = [1, 1, 1]

    def __init__(self, df, G, E, sent, category, seed=0):
        """df reviews with row-aligned embeddings E (n_reviews x 384)."""
        self.df, self.G, self.E = df.reset_index(drop=True), G, E
        self.sent = E @ sent
        self.category, self.seed = category, seed
        self.texts = self.df.text.values
        self._asp = {}
        self.by_user = {u: g for u, g in self.df.groupby("user").indices.items()}
        self.item_of = self.df.item.values; self.t = self.df.t.values; self.r = self.df.rating.values
        self.item_rows = self.df.groupby("item").indices
        self.mu, self.sd = {}, {}
        # global fallback for the same-item cosine baseline
        rng = np.random.default_rng(seed)
        its = list(self.item_rows); smp = rng.choice(len(its), min(200, len(its)), replace=False)
        for j in smp:
            self._baseline(its[j])
        self.mu0 = float(np.mean(list(self.mu.values()))) if self.mu else 0.3
        self.sd0 = float(np.mean(list(self.sd.values()))) if self.sd else 0.15

    def _baseline(self, it):
        """Mean/std of cosine similarity between random review pairs of item `it`
        (lazy; used to z-score the coordinated-similarity feature CS)."""
        if it not in self.mu:
            idx = self.item_rows.get(it, [])
            if len(idx) > 60:
                idx = np.random.default_rng(self.seed).choice(idx, 60, replace=False)
            if len(idx) >= 3:
                S = self.E[idx] @ self.E[idx].T; iu = np.triu_indices(len(idx), 1)
                self.mu[it], self.sd[it] = S[iu].mean(), S[iu].std() + 1e-3
            else:
                self.mu[it], self.sd[it] = None, None
        return self.mu[it], self.sd[it]

    def asp(self, i):
        if i not in self._asp:
            self._asp[i] = A.extract(self.texts[i], self.category)
        return self._asp[i]

    def shared(self, g):
        rows = np.concatenate([self.by_user[u] for u in g if u in self.by_user])
        cnt = collections.Counter(self.item_of[rows])
        thr = max(2, 0.5 * len(g))
        return rows, [it for it, c in cnt.items() if c >= thr], cnt

    def features(self, g):
        g = list(g); n = len(g)
        rows, sh, cnt = self.shared(g)
        sub = self.G.subgraph(g); m = sub.number_of_edges()
        dens = 2 * m / (n * (n - 1)); mw = np.mean([d["weight"] for *_, d in sub.edges(data=True)]) if m else 0.0
        shs = set(sh)
        srows = rows[np.isin(self.item_of[rows], list(shs))] if sh else rows[:0]
        if len(srows):
            per = pd.DataFrame(dict(item=self.item_of[srows], t=self.t[srows], r=self.r[srows]))
            span = per.groupby("item").t.std().fillna(0).mean(); rv = per.groupby("item").r.var().fillna(0).mean()
            ext = np.abs(per.r - 3).mean()
        else:
            span, rv, ext = 365.0, 2.0, 0.0
        top = sum(c for _, c in cnt.most_common(max(1, len(sh))))
        beh = [np.log(n), dens, mw, len(sh), np.log1p(span), rv, ext, top / len(rows)]
        # ---- text coordination on shared items
        cs, ao = [], []
        for it in sh:
            ri = srows[self.item_of[srows] == it]
            if len(ri) < 2:
                continue
            S = self.E[ri] @ self.E[ri].T; iu = np.triu_indices(len(ri), 1)
            mu, sd = self._baseline(it)
            mu, sd = (self.mu0, self.sd0) if mu is None else (mu, sd)
            cs.append(((S[iu] - mu) / sd).mean())
            asp = [self.asp(i) for i in ri]
            ao += [len(a & b) / len(a | b) if a | b else 0.0 for a, b in itertools.combinations(asp, 2)]
        CS = float(np.mean(cs)) if cs else 0.0
        AO = float(np.mean(ao)) if ao else 0.0
        SP = 1.0 - float(np.std(self.sent[srows])) if len(srows) > 1 else 0.0
        return np.array(beh, float), np.array([CS, AO, SP], float)

    def all(self, groups):
        B, T = [], []
        for g in groups:
            b, t = self.features(g); B.append(b); T.append(t)
        return np.array(B).reshape(len(groups), len(self.BEH)), np.array(T).reshape(len(groups), 3)
