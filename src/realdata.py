"""E6: agreement with REAL spam labels (public CARE-GNN releases of YelpChi and Amazon).

These releases contain no raw text and no timestamps, only hand-crafted node
features, relation graphs and labels, so only the behavioural part of
MOSAC-SGD can run on them (Stage 1 -> group scoring -> overlap propagation).

  YelpChi.mat  45,954 reviews (Yelp filter labels, Rayana & Akoglu 2015).
               Relations: R-U-R (same user), R-T-R (same product, same month),
               R-S-R (same product, same star). We recover users as connected
               components of R-U-R, label a user spam if any of their reviews is
               filtered, and connect two users when one review of each is linked
               by BOTH R-T-R and R-S-R (same product, same month, same star:
               the co-review rule of the injected benchmark at monthly granularity).
  Amazon.mat   11,944 users (Musical Instruments; label from helpful votes,
               Zhang et al. 2020). Co-review graph = U-P-U AND U-S-U.
"""
import os, collections, time
import numpy as np, pandas as pd, networkx as nx
import scipy.io as sio
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from . import stage1_groups as S1, stage3_detect as S3


def load_yelpchi(path):
    m = sio.loadmat(path)
    rur = m["net_rur"].tocsr(); both = m["net_rtr"].multiply(m["net_rsr"]).tocsr()
    lab = m["label"].ravel().astype(int); X = m["features"].toarray()
    n_u, uid = connected_components(rur, directed=False)
    # user-level features: mean of review features + review count; label: any filtered review
    U = pd.DataFrame(X).groupby(uid).mean()
    U["n"] = np.log1p(np.bincount(uid))
    y = pd.Series(lab).groupby(uid).max().values
    both = sparse.triu(both, 1).tocoo()
    a, b = uid[both.row], uid[both.col]; k = a != b
    E = pd.DataFrame(dict(a=np.minimum(a[k], b[k]), b=np.maximum(a[k], b[k]))).value_counts().reset_index()
    E.columns = ["a", "b", "c"]
    return U.values, y, E


def load_amazon(path):
    m = sio.loadmat(path)
    both = m["net_upu"].multiply(m["net_usu"]).tocsr()
    X = m["features"].toarray(); y = m["label"].ravel().astype(int)
    both = sparse.triu(both, 1).tocoo()
    E = pd.DataFrame(dict(a=both.row, b=both.col, c=1))
    return X, y, E


def run_one(name, X, y, E, cfg, seed=0, algos=("Louvain", "DEMON", "SLPA")):
    rows = []
    n = len(y)
    Xs = StandardScaler().fit_transform(np.nan_to_num(X))
    # user-level floor: Isolation Forest on user features
    s_if = -IsolationForest(n_estimators=300, random_state=seed).fit(Xs).score_samples(Xs)
    base = dict(dataset=name, users=n, spam_rate=float(y.mean()), graph_edges=len(E))
    rows.append(dict(base, method="Behavioural IF (no graph)", n_groups=np.nan, covered=1.0,
                     AUC_PR=average_precision_score(y, s_if), AUC_ROC=roc_auc_score(y, s_if)))
    G = nx.Graph()
    G.add_weighted_edges_from(zip(E.a, E.b, np.minimum(1.0, E.c / 2.0)))
    # degree-only reference
    deg = np.zeros(n); dd = dict(G.degree()); deg[list(dd)] = list(dd.values())
    rows.append(dict(base, method="Co-review degree", n_groups=np.nan, covered=float((deg > 0).mean()),
                     AUC_PR=average_precision_score(y, deg), AUC_ROC=roc_auc_score(y, deg)))
    nbr_items = {u: set(G.neighbors(u)) | {u} for u in G.nodes()}  # proxy "items" for the shared-item filter
    s1 = dict(cfg["stage1"], min_shared_items=0, max_size=200)
    for algo in algos:
        t0 = time.time()
        params = {"SLPA": s1["slpa"], "DEMON": s1["demon"]}.get(algo, {})
        raw = S1.ALGOS[algo](G, seed=seed, **params)
        groups = S1.postprocess(raw, nbr_items, s1)
        if not groups:
            continue
        H = []
        for g in groups:
            g = list(g); sub = G.subgraph(g); mm = sub.number_of_edges(); k = len(g)
            H.append(np.r_[np.log(k), 2 * mm / (k * (k - 1)), np.mean([d["weight"] for *_, d in sub.edges(data=True)]) if mm else 0,
                           Xs[g].mean(0)])
        H = np.array(H)
        H = np.hstack([H[:, :3], PCA(min(8, H.shape[1] - 3, len(H) - 1), random_state=seed).fit_transform(H[:, 3:])]) if len(H) > 9 else H
        S = S3.score_groups(H, cfg["stage3"]["scorer"], True, seed)
        for how in (["single"] if algo == "Louvain" else ["noisy_or", "max", "mean"]):
            p = S3.propagate(groups, S, "max" if how == "single" else how, 0.0)
            # users outside every group: rank them below grouped users, ordered by the IF score
            sc = np.array([p.get(u, -1.0) for u in range(n)])
            sc = np.where(sc < 0, -1 + (pd.Series(s_if).rank(pct=True).values) * 1e-3, sc)
            rows.append(dict(base, method=f"MOSAC-SGD behav. ({algo}, {how})", n_groups=len(groups),
                             covered=float(np.mean([u in p for u in range(n)])),
                             AUC_PR=average_precision_score(y, sc), AUC_ROC=roc_auc_score(y, sc),
                             t_s=time.time() - t0))
    return rows


def main(cfg, out_dir, seed=0):
    root = os.path.join(cfg["paths"]["raw"], "caregnn")
    rows = []
    for name, loader, f in (("YelpChi", load_yelpchi, "YelpChi.mat"), ("Amazon (CARE-GNN)", load_amazon, "Amazon.mat")):
        p = os.path.join(root, f)
        if not os.path.exists(p):
            print(f"  skip {name}: {p} missing"); continue
        print(f"[E6] {name}", flush=True)
        X, y, E = loader(p)
        rows += run_one(name, X, y, E, cfg, seed)
        print(pd.DataFrame(rows).tail(6).to_string(), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "e6_real_labels.csv"), index=False)
    return df


# ------------------------------------------------------------------ OpSpam text sanity check
def opspam_check(cfg, out_dir, seed=0):
    """Text-branch sanity check: can SBERT embeddings separate crowd-sourced deceptive
    hotel reviews from truthful ones (5-fold CV logistic regression)?"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from .stage2_embed import Embedder
    root = os.path.join(cfg["paths"]["raw"], "opspam", "op_spam_v1.4")
    if not os.path.isdir(root):
        print("  skip OpSpam: not downloaded"); return None
    texts, y, pol = [], [], []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".txt"):
                texts.append(open(os.path.join(dirpath, f), encoding="utf-8", errors="ignore").read().strip())
                y.append(int("deceptive" in dirpath)); pol.append("positive" if "positive" in dirpath else "negative")
    emb = Embedder(cfg["stage2"]["sbert"], os.path.join(cfg["paths"]["cache"], "sbert_opspam.npz"))
    E = emb.encode(texts); emb.save()
    y = np.array(y)
    acc = cross_val_score(LogisticRegression(max_iter=2000, C=1.0), E, y, cv=5, scoring="accuracy")
    res = pd.DataFrame([dict(dataset="OpSpam v1.4", reviews=len(y), deceptive=int(y.sum()),
                             cv_accuracy_mean=acc.mean(), cv_accuracy_std=acc.std())])
    res.to_csv(os.path.join(out_dir, "opspam_text_sanity.csv"), index=False)
    print(res.to_string(), flush=True)
    return res
