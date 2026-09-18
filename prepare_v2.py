"""v2 corpus pipeline: stratified mix of c4 modern Chinese + classical poetry.

Sources:
  - D:/user/corpus/c4/*.jsonl.zst  (96 shards; sample subset for 25M chars)
  - chinese-poetry jsons (30M chars target, all of what we have)
Output: data_v2/train.bin, val.bin, vocab.json  (doc-level split)
"""
import glob, gzip, json, os, random, re, sys, time
import numpy as np

random.seed(11)
CJK = re.compile(r"[\u4e00-\u9fff]")

def clean(t): return re.sub(r"\s+", "", t)

docs = []   # (text, kind)

# ---- poetry (chinese-poetry, all downloaded shards)
DATA = r"D:/user/flypoet/data"
for pat, kind in [("ci.song.*.json", "ci"), ("poet.tang.*.json", "shi"),
                  ("poet.song.*.json", "shi"), ("yuanqu.json", "qu")]:
    for fp in glob.glob(os.path.join(DATA, pat)):
        try:
            items = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(items, list):
            continue
        for it in items:
            paras = it.get("paragraphs") or it.get("content") or []
            if isinstance(paras, str):
                paras = [paras]
            text = clean("".join(paras))
            title = clean(it.get("title", "") or "")
            if len(text) >= 20:
                docs.append((f"《{title}》{text}", kind))
n_poetry = len(docs)
chars_poetry = sum(len(d) for d, _ in docs)
print(f"poetry docs: {n_poetry}, {chars_poetry/1e6:.1f} chars", flush=True)

# ---- modern literature (鲁迅全集 md)
n_lx = 0
for fp in glob.glob(r"D:/user/flypoet/data/luxun-master/全集/**/*.md", recursive=True):
    raw = open(fp, encoding="utf-8").read()
    raw = re.sub(r"#[^\n]*", "", raw)
    raw = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", raw)
    raw = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", raw)
    text = re.sub(r"\s+", "", raw)
    if len(text) < 200:
        continue
    for i in range(0, len(text), 2000):          # chunk long essays (non-overlap)
        chunk = text[i:i + 2000]
        if len(chunk) >= 200:
            docs.append((chunk, "luxun"))
            n_lx += 1
print(f"luxun chunks: {n_lx}", flush=True)

# ---- c4 zst shards (sample for 30M chars)
import zstandard as zstd
C4 = r"D:/user/corpus/c4"
TARGET_C4 = 30_000_000
shards = sorted(glob.glob(os.path.join(C4, "*.jsonl.zst")))
random.shuffle(shards)
got = 0
dctx = zstd.ZstdDecompressor()
for sp in shards:
    if got >= TARGET_C4:
        break
    try:
        with open(sp, "rb") as fh:
            reader = dctx.stream_reader(fh)
            text_stream = reader.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"skip corrupt shard {os.path.basename(sp)}: {str(e)[:50]}", flush=True)
        continue
    for line in text_stream.split("\n"):
        if got >= TARGET_C4:
            break
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "zho" not in (d.get("content_language") or ""):
            continue
        text = (d.get("text") or "").strip()
        if not (120 <= len(text) <= 3000):
            continue
        if len(CJK.findall(text)) / len(text) < 0.85:
            continue
        docs.append((clean(text), "c4"))
        got += len(clean(text))
    print(f"c4 shard {os.path.basename(sp)[:16]}: cumulative {got/1e6:.1f}M chars", flush=True)

print(f"total docs: {len(docs)}")
random.shuffle(docs)
split = int(len(docs) * 0.97)
train_docs, val_docs = docs[:split], docs[split:]

all_train = "".join(d for d, _ in train_docs)
freq = {}
for ch in all_train:
    freq[ch] = freq.get(ch, 0) + 1
vocab_chars = [c for c, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:10000]]
stoi = {c: i + 2 for i, c in enumerate(vocab_chars)}
stoi["<pad>"], stoi["<unk>"] = 0, 1
itos = {i + 2: c for c, i in stoi.items()}; itos[0] = "<pad>"; itos[1] = "<unk>"

def dump(dl_, path):
    ids = []
    for d, _ in dl_:
        e = [stoi.get(c, 1) for c in d]
        e.append(0)
        ids.extend(e)
    np.array(ids, dtype=np.uint16).tofile(path)
    return len(ids)

OUT = r"D:/user/flypoet/data_v2"
os.makedirs(OUT, exist_ok=True)
# 域标签元数据（第二阶段多域训练/校准评估用）
meta = [{"kind": k, "len": len(d)} for d, k in train_docs]
json.dump(meta, open(os.path.join(OUT, "train_meta.json"), "w", encoding="utf-8"))
n_tr = dump(train_docs, os.path.join(OUT, "train.bin"))
n_va = dump(val_docs, os.path.join(OUT, "val.bin"))
json.dump({"stoi": stoi, "itos": {str(k): v for k, v in itos.items()}},
          open(os.path.join(OUT, "vocab.json"), "w", encoding="utf-8"), ensure_ascii=False)
print(f"train {n_tr/1e6:.1f}M | val {n_va/1e6:.2f}M | vocab {len(stoi)}")
