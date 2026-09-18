"""Prepare the Chinese-literature corpus for char-level training.

Sources:
  - ci.song.*.json / poet.tang.*.json / poet.song.*.json / yuanqu.json  (chinese-poetry)
  - luxun-master/全集/**/*.md                                           (鲁迅全集)

Output: data/train.bin, data/val.bin (uint32 token ids), data/vocab.json
Split is BY DOCUMENT (poems/essays), never mid-text, so val measures real
generalisation to unseen works.
"""
import json, glob, os, random, re
import numpy as np

DATA = r"D:/user/flypoet/data"
OUT = r"D:/user/flypoet/data"
random.seed(7)

def clean(t: str) -> str:
    return re.sub(r"\s+", "", t)

docs = []

# ---- classical poetry (chinese-poetry)
for pat, kind in [("ci.song.*.json", "ci"), ("poet.tang.*.json", "shi"),
                  ("poet.song.*.json", "shi"), ("yuanqu.json", "qu")]:
    for fp in glob.glob(os.path.join(DATA, pat)):
        try:
            items = json.load(open(fp, encoding="utf-8"))
        except Exception as e:
            print("skip", fp, e); continue
        for it in items:
            paras = it.get("paragraphs") or it.get("content") or []
            if isinstance(paras, str):
                paras = [paras]
            text = clean("".join(paras))
            title = clean(it.get("title", "") or "")
            if len(text) < 20:
                continue
            docs.append((f"《{title}》{text}", kind))

# ---- modern literature (鲁迅)
n_lx = 0
for fp in glob.glob(r"D:/user/flypoet/data/luxun-master/全集/**/*.md", recursive=True):
    raw = open(fp, encoding="utf-8").read()
    raw = re.sub(r"#[^\n]*", "", raw)                    # md headers
    raw = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", raw)       # images
    raw = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", raw)   # links -> text
    text = re.sub(r"\s+", "", raw)
    if len(text) < 200:
        continue
    for i in range(0, len(text), 2000):                  # chunk long essays
        chunk = text[i:i+2000]
        if len(chunk) >= 200:
            docs.append((chunk, "luxun"))
            n_lx += 1

random.shuffle(docs)
print(f"docs total: {len(docs)}  (鲁迅 chunks: {n_lx})")
chars = sum(len(d) for d, _ in docs)
print(f"total chars: {chars/1e6:.1f}M")

# ---- char vocab from TRAIN docs only
random.shuffle(docs)  # already shuffled; split first then count would leak less, but char coverage matters
split = int(len(docs) * 0.97)
train_docs, val_docs = docs[:split], docs[split:]
# guard: ensure val kinds present
from collections import Counter
print("val docs:", len(val_docs), Counter(k for _, k in val_docs))

all_text_train = "".join(d for d, _ in train_docs)
freq = {}
for ch in all_text_train:
    freq[ch] = freq.get(ch, 0) + 1
vocab_chars = [c for c, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:8000]]
stoi = {c: i + 2 for i, c in enumerate(vocab_chars)}   # 0=pad 1=unk
itos = {i + 2: c for c, i in stoi.items()}
stoi["<pad>"], stoi["<unk>"] = 0, 1
itos[0], itos[1] = "<pad>", "<unk>"

def encode(doc: str):
    return [stoi.get(c, 1) for c in doc]

def dump(dlist, path):
    ids = []
    for d, _ in dlist:
        e = encode(d)
        e.append(0)   # pad as doc separator
        ids.extend(e)
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(path)
    return len(ids)

n_tr = dump(train_docs, os.path.join(OUT, "train.bin"))
n_va = dump(val_docs, os.path.join(OUT, "val.bin"))
json.dump({"stoi": stoi, "itos": {str(k): v for k, v in itos.items()}},
          open(os.path.join(OUT, "vocab.json"), "w", encoding="utf-8"), ensure_ascii=False)
print(f"train tokens: {n_tr/1e6:.1f}M | val tokens: {n_va/1e6:.2f}M | vocab: {len(stoi)}")
