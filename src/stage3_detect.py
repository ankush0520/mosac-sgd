"""Stage 3: unsupervised group anomaly score S(g) and overlap propagation p_u."""
import collections
import numpy as np, pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def score_groups(H, scorer="mahalanobis", rank_norm=True, seed=0, signs=None):
    """H: group vectors (n_groups x d). Returns S in (0,1].
    scorer: mahalanobis (report default), iforest, or directional: only deviations
    in the a-priori suspicious direction of each signed feature count (signs: +1/-1,
    0 = unsigned, e.g. embedding dims, which enter through a Mahalanobis term)."""
    n = len(H)
    if n == 0:
        return np.zeros(0)
    if n < 5:
        return np.full(n, 0.5)
    Z = StandardScaler().fit_transform(np.nan_to_num(H))
    if scorer == "directional" and signs is not None:
        sg = np.asarray(signs, float)
        pos = np.clip(Z[:, sg != 0] * sg[sg != 0], 0, None)
        d2 = (pos ** 2).sum(1)
        U = Z[:, sg == 0]
        if U.shape[1] and n > U.shape[1]:
            d2 = d2 + LedoitWolf().fit(U).mahalanobis(U) / U.shape[1]
        d = np.sqrt(d2)
    elif scorer == "iforest":
        d = -IsolationForest(n_estimators=300, random_state=seed).fit(Z).score_samples(Z)
    else:  # d(g) = || Sigma^{-1/2} (h_g - mu) ||, shrinkage covariance for stability
        lw = LedoitWolf().fit(Z)
        d = np.sqrt(np.maximum(lw.mahalanobis(Z), 0))
    if rank_norm:
        return pd.Series(d).rank(pct=True).values
    return (d - d.min()) / (d.max() - d.min() + 1e-12)


def propagate(groups, S, how="noisy_or", tau_g=0.0):
    """p_u from all groups containing u with S(g) >= tau_g (users whose groups are all
    below the floor get max(S) * tau_g, i.e. ranked below every propagated user)."""
    mem = collections.defaultdict(list); best = collections.defaultdict(float)
    for g, s in zip(groups, S):
        for u in g:
            best[u] = max(best[u], s)
            if s >= tau_g:
                mem[u].append(s)
    out = {}
    for u in best:
        s = np.array(mem[u])
        if len(s) == 0:
            out[u] = best[u] * tau_g if tau_g > 0 else best[u]
        elif how == "noisy_or":
            out[u] = 1 - np.prod(1 - s)
        elif how in ("max", "single"):
            out[u] = s.max()
        elif how == "mean":
            out[u] = s.mean()
        else:
            raise ValueError(how)
    return out
