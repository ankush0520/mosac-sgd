"""Load -> normalise -> filter a host review dataset.

Output (data/processed/):
  <host>.parquet       columns user, item, rating, t (days since window start), text
  <host>_pool.parquet  human-written texts from OUTSIDE the kept sample (rating, text);
                       used as camouflage-review text for injected spammer accounts.

    python -m src.data_prep --config configs/default.yaml
"""
import argparse, json, os
import numpy as np, pandas as pd
from .config import load_config


def _read_amazon(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append((d["user_id"], d["parent_asin"], float(d["rating"]), d["timestamp"] / 1000.0,
                         (d.get("title") or "").strip(), (d.get("text") or "").strip()))
    df = pd.DataFrame(rows, columns=["user", "item", "rating", "ts", "title", "text"])
    df["text"] = np.where(df.title.str.len() > 0, df.title + ". " + df.text, df.text)
    return df.drop(columns="title")


def _read_yelp(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append((d["user_id"], d["business_id"], float(d["stars"]),
                         pd.Timestamp(d["date"]).timestamp(), (d.get("text") or "").strip()))
    return pd.DataFrame(rows, columns=["user", "item", "rating", "ts", "text"])


def kcore(df, min_user, min_item, max_iter=20):
    for _ in range(max_iter):
        n = len(df)
        u = df.user.value_counts(); df = df[df.user.isin(u[u >= min_user].index)]
        i = df.item.value_counts(); df = df[df.item.isin(i[i >= min_item].index)]
        if len(df) == n:
            break
    return df


def prepare(host, hc, raw_root, out_root, seed=0):
    path = os.path.join(raw_root, hc["file"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run `python -m src.download` first")
    print(f"[{host}] reading {path}", flush=True)
    df = _read_yelp(path) if hc.get("format") == "yelp" else _read_amazon(path)
    n_raw = len(df)
    df = df[df.text.str.len() > 0]
    df = df.drop_duplicates(["user", "item", "text"])
    df["year"] = pd.to_datetime(df.ts, unit="s").dt.year
    inwin = df.year.between(*hc["years"]) if hc.get("years") else pd.Series(True, index=df.index)
    sample = kcore(df[inwin], hc["min_user"], hc["min_item"])
    if hc.get("max_items") and sample.item.nunique() > hc["max_items"]:
        keep = sample.item.value_counts().index[: hc["max_items"]]
        sample = kcore(sample[sample.item.isin(keep)], hc["min_user"], hc["min_item"])
    rest = df.drop(index=sample.index)
    pool = rest.sample(min(len(rest), 30000), random_state=seed)[["rating", "text"]]
    out = sample[["user", "item", "rating", "ts", "text"]].copy()
    out["t"] = (out.ts - out.ts.min()) / 86400.0
    out = out.drop(columns="ts").reset_index(drop=True)
    out["rating"] = out.rating.round().astype(int)
    pool["rating"] = pool.rating.round().astype(int)
    os.makedirs(out_root, exist_ok=True)
    out.to_parquet(os.path.join(out_root, f"{host}.parquet"), index=False)
    pool.reset_index(drop=True).to_parquet(os.path.join(out_root, f"{host}_pool.parquet"), index=False)
    st = dict(host=host, raw_reviews=n_raw, reviews=len(out), users=out.user.nunique(), items=out.item.nunique(),
              days=round(out.t.max(), 1), pool_texts=len(pool))
    print(st, flush=True)
    return st


def load_host(cfg, host):
    p = os.path.join(cfg["paths"]["processed"], f"{host}.parquet")
    if not os.path.exists(p):
        prepare(host, cfg["hosts"][host], cfg["paths"]["raw"], cfg["paths"]["processed"])
    return (pd.read_parquet(p),
            pd.read_parquet(os.path.join(cfg["paths"]["processed"], f"{host}_pool.parquet")))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    stats = [prepare(h, cfg["hosts"][h], cfg["paths"]["raw"], cfg["paths"]["processed"]) for h in cfg["run"]["hosts"]]
    os.makedirs(cfg["paths"]["results"], exist_ok=True)
    pd.DataFrame(stats).to_csv(os.path.join(cfg["paths"]["results"], "host_stats.csv"), index=False)


if __name__ == "__main__":
    main()
