"""Metrics: group P/R/F1@k (Jaccard >= 0.5), user P/R/F1@k and AUC-PR, membership
recall MR and multi-membership recall MMR on multi-campaign spammers, recall by
|C(u)|, and overlapping NMI (LFK, Lancichinetti et al. 2009)."""
import collections
import numpy as np
from sklearn.metrics import average_precision_score


def jacc(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def group_metrics(groups, S, camps, k, thr=0.5):
    truth = [set(c["members"]) for c in camps]
    if not groups:
        return dict(gP=0.0, gR=0.0, gF1=0.0, gRecAll=0.0)
    order = np.argsort(-np.asarray(S), kind="stable")[:k]
    top = [groups[i] for i in order]
    best = [max(jacc(g, t) for t in truth) for g in top]
    P = float(np.mean([b >= thr for b in best]))
    R = float(np.mean([any(jacc(g, t) >= thr for g in top) for t in truth]))
    RA = float(np.mean([any(jacc(g, t) >= thr for g in groups) for t in truth]))
    return dict(gP=P, gR=R, gF1=2 * P * R / (P + R) if P + R else 0.0, gRecAll=RA)


def precision_curve(groups, S, camps, ks, thr=0.5):
    truth = [set(c["members"]) for c in camps]
    order = np.argsort(-np.asarray(S), kind="stable")
    hits = [max(jacc(groups[i], t) for t in truth) >= thr for i in order]
    return {k: float(np.mean(hits[:k])) if hits else 0.0 for k in ks}


def membership_metrics(groups, camps, thr=0.5):
    truth = [set(c["members"]) for c in camps]
    cu = collections.defaultdict(list)
    for i, t in enumerate(truth):
        for u in t:
            cu[u].append(i)
    multi = {u: cs for u, cs in cu.items() if len(cs) >= 2}
    match = collections.defaultdict(set)
    for j, g in enumerate(groups):
        for i, t in enumerate(truth):
            if jacc(g, t) >= thr:
                match[i].add(j)
    hit = tot = mm = 0
    for u, cs in multi.items():
        got = [i for i in cs if any(u in groups[j] for j in match[i])]
        hit += len(got); tot += len(cs); mm += len(got) >= 2
    return dict(MR=hit / tot if tot else np.nan, MMR=mm / len(multi) if multi else np.nan)


def user_metrics(scores, spam_users, all_users, k=None):
    y = np.array([u in spam_users for u in all_users])
    s = np.array([scores.get(u, 0.0) for u in all_users], float)
    k = k or int(y.sum())
    top = np.argsort(-s, kind="stable")[:k]
    P = y[top].mean(); R = y[top].sum() / y.sum()
    return dict(uP=float(P), uR=float(R), uF1=float(2 * P * R / (P + R)) if P + R else 0.0,
                uAP=float(average_precision_score(y, s)))


def recall_by_ncamp(scores, camps, all_users, k):
    cnt = collections.Counter(u for c in camps for u in c["members"])
    s = np.array([scores.get(u, 0.0) for u in all_users])
    top = set(np.asarray(all_users)[np.argsort(-s, kind="stable")[:k]])
    out = {}
    for n in (1, 2, 3):
        us = [u for u, m in cnt.items() if m == n]
        out[f"R_c{n}"] = float(np.mean([u in top for u in us])) if us else np.nan
    multi = [u for u, m in cnt.items() if m >= 2]
    out["R_multi"] = float(np.mean([u in top for u in multi])) if multi else np.nan
    return out


# ---------------------------------------------------------------- ONMI (LFK)
def _h(p):
    return -p * np.log2(p) if p > 0 else 0.0


def onmi_lfk(X, Y, universe):
    """Overlapping NMI of Lancichinetti, Fortunato & Kertesz (2009).
    X, Y: lists of sets (covers) over node set `universe`."""
    N = len(universe)
    X = [x & universe for x in X if x & universe]; Y = [y & universe for y in Y if y & universe]
    if not X or not Y:
        return 0.0

    def H(c):
        p = len(c) / N
        return _h(p) + _h(1 - p)

    def cond(A, B):
        tot = 0.0
        for a in A:
            best = None
            for b in B:
                n11 = len(a & b); n10 = len(a - b); n01 = len(b - a); n00 = N - n11 - n10 - n01
                h11, h00, h10, h01 = (_h(v / N) for v in (n11, n00, n10, n01))
                if h11 + h00 >= h01 + h10:
                    hab = h11 + h00 + h01 + h10 - H(b)
                    best = hab if best is None else min(best, hab)
            Ha = H(a)
            tot += (best / Ha if best is not None and Ha > 0 else 1.0)
        return tot / len(A)

    return float(1 - 0.5 * (cond(X, Y) + cond(Y, X)))
