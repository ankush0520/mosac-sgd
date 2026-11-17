"""Synthetic campaign injection (planted groups with exact ground truth).

Protocol (end-sem report, "Synthetic LLM-Campaign Injection"):
  * K campaigns, 3-8 target items, 5-30 accounts, a 1-10 day window, mostly 5-star.
  * Overlap: a fraction of spammer accounts join 2 or 3 campaigns.
  * Camouflage: every spammer also posts 5-15 genuine-looking reviews (real human
    text taken from host reviews outside the kept sample).
  * Text: T0 templated | T1 LLM-plain | T2 LLM-diverse, from the cached LLM pool
    (scripts/generate_llm_texts.py). All members of one campaign share one brief
    (aspect list), as a real campaign brief would.
  * Benign-LLM control: ordinary background reviews are re-written with benign
    LLM text of the same rating, so "LLM-sounding text" is not a spam shortcut.

The injection is split in two so that the text condition can be varied while
the behavioural structure (who reviewed what, when, with which rating) stays
byte-identical:  inject_structure()  then  apply_text().
"""
import json, os
import numpy as np, pandas as pd
from . import aspects as A

TEMPLATES = [
    "Great {product}! The {a0} is excellent and the {a1} exceeded my expectations. {A2} is also top notch. Highly recommend.",
    "I love this {product}. Amazing {a0}, solid {a1}, and the {a2} is perfect. Five stars!",
    "Best purchase this year. The {a0} and {a1} are fantastic, and {a2} is great too. Buy it!",
    "Excellent {product}. {A0} is superb, {a1} is impressive and {a2} is just right. Would buy again.",
    "This {product} is wonderful: great {a0}, great {a1}, great {a2}. Totally worth it.",
]
NEG_TEMPLATES = [
    "Terrible {product}. The {a0} is awful and the {a1} is a joke. {A2} was disappointing too. Avoid.",
    "Do not buy this {product}. Poor {a0}, bad {a1}, and the {a2} is the worst. One star.",
    "Waste of money. The {a0} and {a1} are horrible and {a2} is no better. Returning it.",
]


def load_llm_pool(cfg, host):
    path = os.path.join(cfg["paths"]["llm_cache"], f"{host}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing. It is committed to the repo; regenerate with "
                                f"python scripts/generate_llm_texts.py --category {host}")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    df = pd.DataFrame([r for r in rows if len(r["text"].split()) >= 8])
    df["brief"] = df.brief.astype("Int64")
    return df


def briefs_from_pool(llm):
    """brief id -> (aspects, promote)"""
    out = {}
    t2 = set(llm[llm.pool == "T2"].brief.dropna().astype(int))
    for b, g in llm[llm.pool == "T1"].groupby("brief"):
        if int(b) not in t2:  # only briefs with texts for every LLM condition
            continue
        out[int(b)] = (list(g.aspects.iloc[0]), bool(g.rating.mean() >= 3))
    return out


def _template_text(rng, product, asp, rating):
    tpl = TEMPLATES if rating >= 3 else NEG_TEMPLATES
    t = tpl[rng.integers(len(tpl))]
    a = list(asp); rng.shuffle(a)
    return t.format(product=product, a0=a[0], a1=a[1], a2=a[2], A0=a[0].capitalize(), A2=a[2].capitalize())


def inject_structure(df, pool, briefs, cfg, seed):
    """Behavioural part of the injection. df: host reviews (user,item,rating,t,text).
    pool: human texts for camouflage. briefs: {id: (aspects, promote)}.
    Returns (reviews with columns spam/llm/camp, campaigns, info)."""
    c = cfg["inject"]; rng = np.random.default_rng(seed)
    df = df.copy(); df["spam"] = 0; df["llm"] = 0; df["camp"] = -1
    items = df.item.unique(); horizon = float(df.t.max())
    pop = df.item.value_counts().reindex(items).values.astype(float); pop /= pop.sum()
    K = c["K"]
    sizes = rng.integers(c["members"][0], c["members"][1] + 1, K)
    spammers, members = [], [set() for _ in range(K)]
    for k in range(K):
        for _ in range(sizes[k]):
            s = f"spam_{seed}_{len(spammers)}"; spammers.append(s); members[k].add(s)
    if c["overlap"] > 0:
        for s in rng.choice(spammers, int(round(c["overlap"] * len(spammers))), replace=False):
            own = next(k for k in range(K) if s in members[k])
            others = [k for k in range(K) if k != own]
            for k in rng.choice(others, rng.integers(1, 3), replace=False):
                members[k].add(s)
    bids = sorted(briefs)
    rows, camps = [], []
    for k in range(K):
        targets = rng.choice(items, rng.integers(c["targets"][0], c["targets"][1] + 1), replace=False)
        start = rng.uniform(0, max(horizon - 10, 1)); win = rng.uniform(*c["window"])
        b = int(bids[rng.integers(len(bids))]); asp, promote = briefs[b]
        for s in sorted(members[k]):
            for it in targets:
                if rng.random() >= c["hit_prob"]:
                    continue
                r = (5 if promote else 1) if rng.random() > 0.15 else (4 if promote else 2)
                rows.append((s, it, r, start + rng.uniform(0, win), "", 1, 0, k))
        camps.append(dict(id=k, members=sorted(members[k]), items=[str(x) for x in targets],
                          start=float(start), window=float(win), promote=bool(promote), brief=b,
                          aspects=list(asp)))
    n_camp = len(rows)
    for s in spammers:  # camouflage
        n = rng.integers(c["camouflage"][0], c["camouflage"][1] + 1)
        for it in rng.choice(items, n, replace=False, p=pop):
            pr = pool.iloc[rng.integers(len(pool))]
            rows.append((s, it, int(pr.rating), rng.uniform(0, horizon), pr.text, 0, 0, -1))
    inj = pd.DataFrame(rows, columns=["user", "item", "rating", "t", "text", "spam", "llm", "camp"])
    # a spammer can hit the same target twice when it is in two campaigns: keep the first
    out = pd.concat([df, inj], ignore_index=True).drop_duplicates(["user", "item"], keep="first")
    out = out.reset_index(drop=True)
    info = dict(spammers=len(spammers), campaign_reviews=int(out.spam.sum()),
                camouflage_reviews=int((out.user.str.startswith("spam_") & out.spam.eq(0)).sum()),
                multi=sum(1 for s in spammers if sum(s in m for m in members) >= 2))
    return out, camps, info


def apply_text(reviews, camps, llm, cfg, seed, category, text=None, control=None):
    """Fill campaign review text for condition `text` (T0/T1/T2) and apply the
    benign-LLM control (on/off). Uses its own RNG so structure is untouched."""
    c = cfg["inject"]; text = text or c["text"]; control = c["control"] if control is None else control
    rng = np.random.default_rng(seed * 1000 + {"T0": 0, "T1": 1, "T2": 2}[text] + 10 * int(control))
    spec = A.get(category)
    df = reviews.copy()
    ti, li = df.columns.get_loc("text"), df.columns.get_loc("llm")
    texts_col = df.text.values.copy(); llm_col = df.llm.values.copy()
    spam_idx = np.flatnonzero(df.spam.values == 1)
    users = df.user.values; ratings = df.rating.values; campcol = df.camp.values
    for cp in camps:
        if text in ("T1", "T2"):
            pool = llm[(llm.pool == text) & (llm.brief == cp["brief"])]
            pos = pool[pool.rating >= 3].text.values; neg = pool[pool.rating < 3].text.values
            pos = pos if len(pos) else pool.text.values; neg = neg if len(neg) else pool.text.values
            pos = pos[rng.permutation(len(pos))]; neg = neg[rng.permutation(len(neg))]
        idx = spam_idx[campcol[spam_idx] == cp["id"]]
        mem = {u: j for j, u in enumerate(cp["members"])}
        for i in idx:
            j = mem[users[i]]
            if text == "T0":
                texts_col[i] = _template_text(rng, spec["product"], cp["aspects"], int(ratings[i])); llm_col[i] = 0
            else:  # each member keeps one LLM text for the whole campaign
                arr = pos if ratings[i] >= 3 else neg
                texts_col[i] = arr[j % len(arr)]; llm_col[i] = 1
    n_ctrl = 0
    if control:
        ben = llm[llm.pool == "benign"].reset_index(drop=True)
        bg = np.flatnonzero(~pd.Series(users).str.startswith("spam_").values)
        n_ctrl = min(int(c["control_ratio"] * len(spam_idx)), len(bg))
        idx = rng.choice(bg, n_ctrl, replace=False)
        by_r = {r: ben.text.values[ben.rating.values == r] for r in range(1, 6)}
        for i in idx:
            cand = by_r.get(int(ratings[i]))
            cand = cand if cand is not None and len(cand) else ben.text.values
            texts_col[i] = cand[rng.integers(len(cand))]; llm_col[i] = 1
    df["text"] = texts_col; df["llm"] = llm_col
    return df, n_ctrl
