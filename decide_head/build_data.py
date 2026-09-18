"""Build the fly-decider training set from CWT2 selected corpus.

Task: document routing — given a text snippet, choose which of 8 expert
domains should handle it (matches Jev's routing use-case).
Labels come free from CWT2's domain field. Output: decide_data/*.bin + vocab.
"""
import json, glob, os, random, re
import numpy as np

random.seed(5)
SRC = r"D:\user\corpus\cwt2\selected"
OUT = r"D:\user\flypoet\decide_data"
os.makedirs(OUT, exist_ok=True)

DOMAINS = ["general", "news", "encyclopedia", "technology", "law",
           "education", "dialogue", "finance"]
D2I = {d: i for i, d in enumerate(DOMAINS)}

samples = []
for fp in sorted(glob.glob(os.path.join(SRC, "selected-000[1-5]*.jsonl"))):
    with open(fp, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            dominfo = d.get("domain") or {}
            dom = dominfo.get("single_label")
            multi = dominfo.get("multi_label") or []
            # 去噪: 只保留"单标签且 multi 一致"的清晰样本
            if dom not in D2I or (multi and multi != [dom]):
                continue
            text = (d.get("text") or "").strip()
            # truncate to first 200 chars (routing decision uses the head)
            if len(text) < 50:
                continue
            samples.append((text[:200], D2I[dom]))
            if len(samples) >= 120000:
                break
    if len(samples) >= 120000:
        break

random.shuffle(samples)
# balance classes roughly: cap per class
from collections import Counter
cnt = Counter(l for _, l in samples)
cap = min(cnt.values()) * 2
balanced = []
seen = Counter()
for t, l in samples:
    if seen[l] < cap:
        balanced.append((t, l))
        seen[l] += 1
samples = balanced
random.shuffle(samples)
print("class balance:", dict(Counter(l for _, l in samples)))
print("total:", len(samples))

# char vocab (reuse flypoet main vocab for the trunk; head data only needs ids)
vj = json.load(open(r"D:\user\flypoet\data_v2\vocab.json", encoding="utf-8"))
stoi = vj["stoi"]

split = int(len(samples) * 0.9)
splits = {"train": samples[:split], "val": samples[split:]}
for name, rows in splits.items():
    with open(os.path.join(OUT, f"{name}.jsonl"), "w", encoding="utf-8") as f:
        for text, label in rows:
            ids = [stoi.get(c, 1) for c in text][:200]
            f.write(json.dumps({"ids": ids, "label": label}) + "\n")
json.dump({"domains": DOMAINS}, open(os.path.join(OUT, "labels.json"), "w"))
print(f"written: train={len(splits['train'])}, val={len(splits['val'])}")
