"""Download every public dataset used by the project into data/raw/.

    python -m src.download                # everything the default config needs
    python -m src.download --only amazon  # one group: amazon | caregnn | opspam

All sources are public, no login needed:

  amazon   Amazon Reviews 2023 (McAuley Lab), raw review JSONL per category, from
           https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
  caregnn  YelpChi.mat and Amazon.mat graph-fraud benchmarks (real spam labels),
           from https://github.com/YingtongDou/CARE-GNN (data/ folder)
  opspam   Deceptive Opinion Spam corpus v1.4 (Ott et al.), from https://myleott.com/op_spam_v1.4.zip

The Yelp Open Dataset is NOT downloaded: Yelp requires accepting its dataset
terms in a browser. If you have it, put yelp_academic_dataset_review.json in
data/raw/yelp/ and set host: Yelp in the config (see README).
"""
import argparse, os, sys, zipfile, shutil
import requests

HF = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/{}.jsonl"
SOURCES = {
    "amazon": [(HF.format(c), f"amazon2023/{c}.jsonl") for c in ("Musical_Instruments", "All_Beauty")],
    "caregnn": [("https://raw.githubusercontent.com/YingtongDou/CARE-GNN/master/data/YelpChi.zip", "caregnn/YelpChi.zip"),
                ("https://raw.githubusercontent.com/YingtongDou/CARE-GNN/master/data/Amazon.zip", "caregnn/Amazon.zip")],
    "opspam": [("https://myleott.com/op_spam_v1.4.zip", "opspam/op_spam_v1.4.zip")],
}


def fetch(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  have {dest}"); return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    print(f"  GET {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)); got = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk); got += len(chunk)
                if total:
                    sys.stdout.write(f"\r    {got / 1e6:8.1f} / {total / 1e6:.1f} MB"); sys.stdout.flush()
    print()
    os.replace(tmp, dest)
    if dest.endswith(".zip"):
        with zipfile.ZipFile(dest) as z:
            z.extractall(os.path.dirname(dest))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--only", choices=list(SOURCES), nargs="*")
    ap.add_argument("--category", nargs="*", help="extra Amazon-2023 categories to fetch")
    a = ap.parse_args(argv)
    groups = a.only or list(SOURCES)
    for g in groups:
        print(f"[{g}]")
        items = list(SOURCES[g])
        if g == "amazon" and a.category:
            items += [(HF.format(c), f"amazon2023/{c}.jsonl") for c in a.category]
        for url, rel in items:
            fetch(url, os.path.join(a.root, rel))
    print("done")


if __name__ == "__main__":
    main()
