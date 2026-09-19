"""Collect all runs into one table: sweep points, seeds, scale pair, sleep.
Reads logs_v2/*_final.json + last probe of each run. Prints and saves a
markdown summary to logs_v2/ALL_RUNS.md."""
import glob, json, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(ROOT, "logs_v2")


def last_probe(tag):
    fp = os.path.join(LOG, f"{tag}_probes.jsonl")
    if not os.path.exists(fp):
        return None
    rows = [json.loads(l) for l in open(fp, encoding="utf-8")]
    # 追加式文件：只取最后一次运行（step 发生回退处分段）
    runs, prev = [], -1
    for d in rows:
        if d["step"] < prev:
            runs.append([])
        prev = d["step"]
        if not runs:
            runs.append([])
        runs[-1].append(d)
    return runs[-1][-1] if runs else None


def main():
    rows = []
    for fp in sorted(glob.glob(os.path.join(LOG, "*_final.json"))):
        meta = json.load(open(fp))
        tag = os.path.basename(fp)[:-len("_final.json")]
        p = last_probe(tag)
        rows.append({"tag": tag, "arm": meta.get("arm"), "k": meta.get("k_frac"),
                     "impl": meta.get("impl"), "d": meta.get("d"),
                     "seed": meta.get("seed"), "steps": meta.get("steps"),
                     "params_M": round(meta.get("params_M", 0), 1),
                     "val": p.get("val") if p else None,
                     "erank": p.get("erank") if p else None,
                     "d3": p.get("distinct3") if p else None,
                     "keep": p.get("keep") if p else None})
    rows.sort(key=lambda r: (r["steps"] or 0, r["params_M"], r["tag"]))
    lines = ["| tag | arm | k | impl | d | seed | steps | params_M | val | erank | d3 | keep |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in
                     ["tag", "arm", "k", "impl", "d", "seed", "steps",
                      "params_M", "val", "erank", "d3", "keep"]) + " |")
    out = "\n".join(lines)
    print(out)
    with open(os.path.join(LOG, "ALL_RUNS.md"), "w", encoding="utf-8") as f:
        f.write("# All runs\n\n" + out + "\n")


if __name__ == "__main__":
    main()
