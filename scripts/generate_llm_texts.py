"""Generate the cached pool of LLM-written reviews used by the injection.

Runs a small open instruction model (default Qwen/Qwen2.5-0.5B-Instruct,
Apache-2.0) on CPU. Output: data/llm_cache/<category>.jsonl, one JSON per line:
  {"pool": "T1"|"T2"|"benign", "brief": int|null, "rating": int, "persona": str,
   "aspects": [...], "length": int, "temperature": float, "text": str}

The cache is committed to the repository so that reproducing the experiments
does NOT require running the LLM (generation took ~2-3 h on a 4-core CPU).
Every text in it is synthetic and marked as such by its file location.

Usage: python scripts/generate_llm_texts.py --category Musical_Instruments
"""
import argparse, json, os, sys, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import aspects as A

PERSONAS = ["enthusiastic hobbyist", "busy parent", "professional who uses it daily",
            "casual beginner", "retired teacher", "college student on a budget",
            "detail-oriented reviewer", "gift buyer", "skeptical first-time buyer",
            "long-time fan of the brand"]
LENGTHS = [30, 50, 70, 100]
PROMPT = ("You are a customer who recently used {item} ({category}). Write a {length}-word "
          "review giving {rating} stars. Mention these aspects naturally: {aspects}. "
          "Tone: {persona}. Output only the review text.")


def jobs(category, n_briefs, per_brief, n_benign, seed):
    rnd = random.Random(seed)
    spec = A.get(category); names = list(spec["aspects"])
    out = []
    for b in range(n_briefs):
        asp = rnd.sample(names, 3)
        promote = rnd.random() < 0.8
        for pool in ("T1", "T2"):
            for j in range(per_brief):
                rating = (5 if promote else 1) if rnd.random() > 0.15 else (4 if promote else 2)
                if pool == "T1":   # one prompt, one persona, fixed length & temperature
                    per, ln, temp = "satisfied customer" if promote else "disappointed customer", 50, 0.7
                else:              # varied persona / aspect order / length / temperature
                    per, ln, temp = rnd.choice(PERSONAS), rnd.choice(LENGTHS), rnd.choice([0.7, 0.9, 1.1])
                    asp_j = asp[:]; rnd.shuffle(asp_j)
                out.append(dict(pool=pool, brief=b, rating=rating, persona=per,
                                aspects=asp if pool == "T1" else asp_j, length=ln, temperature=temp))
    for j in range(n_benign):  # benign-LLM control: ordinary opinions, any rating, random aspects
        out.append(dict(pool="benign", brief=None, rating=rnd.choice([1, 2, 3, 4, 4, 5, 5, 5]),
                        persona=rnd.choice(PERSONAS), aspects=rnd.sample(names, rnd.randint(1, 3)),
                        length=rnd.choice(LENGTHS), temperature=rnd.choice([0.7, 0.9, 1.1])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Musical_Instruments")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--briefs", type=int, default=40)
    ap.add_argument("--per-brief", type=int, default=24)
    ap.add_argument("--benign", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.manual_seed(a.seed)
    out = a.out or f"data/llm_cache/{a.category}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = sum(1 for _ in open(out)) if os.path.exists(out) else 0
    alljobs = sorted(jobs(a.category, a.briefs, a.per_brief, a.benign, a.seed), key=lambda j: j["temperature"])
    todo = alljobs[done:]  # sorted by temperature so batches share one temperature
    print(f"{done} cached, {len(todo)} to generate", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    spec = A.get(a.category); t0 = time.time()
    # group by temperature so one generate() call uses one temperature
    with open(out, "a") as f:
        i = 0
        while i < len(todo):
            temp = todo[i]["temperature"]; batch = []
            while i < len(todo) and len(batch) < a.batch and todo[i]["temperature"] == temp:
                batch.append(todo[i]); i += 1
            prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(
                item="this " + spec["product"], category=a.category.replace("_", " "), length=j["length"],
                rating=j["rating"], aspects=", ".join(j["aspects"]), persona=j["persona"])}],
                tokenize=False, add_generation_prompt=True) for j in batch]
            x = tok(prompts, return_tensors="pt", padding=True)
            mx = int(max(j["length"] for j in batch) * 1.6) + 20
            with torch.no_grad():
                y = model.generate(**x, max_new_tokens=mx, do_sample=True, temperature=temp, top_p=0.95,
                                   pad_token_id=tok.eos_token_id)
            for j, t in zip(batch, tok.batch_decode(y[:, x["input_ids"].shape[1]:], skip_special_tokens=True)):
                j = dict(j, text=t.strip().strip('"'), generator=a.model)
                f.write(json.dumps(j) + "\n")
            f.flush()
            print(f"{done + i}/{done + len(todo)}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
