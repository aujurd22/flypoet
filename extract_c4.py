"""Stream-extract ~25M chars of clean modern Chinese from chinese-c4.

Filters: language tag contains 'zho', length 120..5000, no leading ASCII,
require >85% CJK chars. Output: data/c4_modern.txt (one doc per line-ish).
"""
import re, time
from datasets import load_dataset

TARGET = 25_000_000
CJK = re.compile(r"[\u4e00-\u9fff]")
out = open(r"D:/user/flypoet/data/c4_modern.txt", "w", encoding="utf-8")
ds = load_dataset("shjwudp/chinese-c4", split="train", streaming=True)
it = iter(ds)
t0 = time.time()
got = 0
seen = 0
while got < TARGET:
    try:
        row = next(it)
    except StopIteration:
        print("stream exhausted"); break
    seen += 1
    lang = row.get("content_language") or ""
    if "zho" not in lang:
        continue
    text = (row.get("text") or "").strip()
    if not (120 <= len(text) <= 5000):
        continue
    cjk = len(CJK.findall(text))
    if cjk / len(text) < 0.85:
        continue
    clean = re.sub(r"\s+", "", text)
    out.write(clean + "\n")
    got += len(clean)
    if seen % 5000 == 0:
        print(f"scanned {seen}, got {got/1e6:.1f}M chars, {time.time()-t0:.0f}s", flush=True)
out.close()
print(f"DONE: {got/1e6:.1f}M chars from {seen} rows, {time.time()-t0:.0f}s")
