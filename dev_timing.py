"""Developmental timing: does the ADDRESS system freeze before the SEMANTIC
system? For each snapshot of a k25 training run:
  stability(step) = Jaccard between codes of the SAME windows at step t and t-1
  semantic(step)  = Spearman(code-Hamming similarity, char PMI)
Plot/answer: which curve saturates first?
"""
import json, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import train_v2 as T
from code_address import codes_for, spearman

DEV = "cuda"
N_BIGRAMS = 60
OCC = 4


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "flynetS_k25_snap"
    corpus = T.Corpus()
    import re
    snaps = sorted(int(re.search(r"snap(\d+)\.pt$", f).group(1))
                   for f in os.listdir(os.path.join(ROOT, "logs_v2"))
                   if f.startswith(f"{tag}_snap") and f.endswith(".pt"))
    if not snaps:
        print("no snapshots found"); return
    print("snapshots:", snaps, flush=True)

    arr = corpus.train[: 2_000_000]
    a, b = arr[:-1], arr[1:]
    pairs, cnts = np.unique(np.stack([a[m], b[m]], 1), axis=0, return_counts=True) \
        if (m := (a != b)).any() else (None, None)
    keep = cnts >= 30
    pairs, cnts = pairs[keep], cnts[keep]
    order = np.argsort(cnts)[::-1]
    rng = np.random.default_rng(3)
    picked = []
    for i in order:
        x, y = int(pairs[i][0]), int(pairs[i][1])
        hits = np.where((arr[:-1] == x) & (arr[1:] == y))[0]
        hits = hits[(hits > 40) & (hits < len(arr) - 40)]
        if len(hits) < OCC:
            continue
        picked.append((x, y, rng.choice(hits, size=OCC, replace=False)))
        if len(picked) >= N_BIGRAMS:
            break
    chars = sorted({c for x, y, _ in picked for c in (x, y)})
    char_pos, owner = [], []
    for ci, c in enumerate(chars):
        hits = np.where(arr == c)[0]
        hits = hits[(hits > 40) & (hits < len(arr) - 40)]
        for p in rng.choice(hits, size=min(OCC, len(hits)), replace=False):
            char_pos.append(int(p))
            owner.append(ci)

    # fixed query windows for stability (val stream)
    qpos = rng.integers(40, len(corpus.val) - 2, 300)

    prev_code = None
    out = {"steps": [], "stability": [], "spearman_pmi": []}
    model = T.GPT(corpus.V, kwta_opts={"impl": "torch", "k_frac": 0.25}).to(DEV)
    for st in snaps:
        sd = torch.load(os.path.join(ROOT, "logs_v2", f"{tag}_snap{st}.pt"),
                        map_location=DEV, weights_only=True)
        model.load_state_dict(sd)
        model.eval()
        C_bi, _ = codes_for(model, corpus,
                            np.array([int(p) + 1 for x, y, hs in picked for p in hs]))
        C_ch, _ = codes_for(model, corpus, np.array(char_pos))
        C_q, _ = codes_for(model, corpus, qpos)
        J = jaccard_matrix(C_q).numpy()
        n = len(qpos)
        iu = np.triu_indices(n, 1)
        stability = float(np.mean(J[iu]))
        # semantic: per-bigram majority codes vs char PMI
        ch_code = {}
        for ci, c in enumerate(chars):
            sel = [k for k, o in enumerate(owner) if o == ci]
            ch_code[c] = C_ch[sel].float().mean(0) > 0.5
        reps = list(range(len(picked)))
        Db = jaccard_matrix(C_bi[: len(picked)]).numpy()
        pm = np.random.rand(len(reps), len(reps))  # placeholder replaced below
        big = arr[: 1_500_000]
        chars_l = [c for x, y, _ in picked for c in (x, y)]
        chars_u = sorted(set(chars_l))
        co = np.zeros((len(chars_u), len(chars_u)))
        c2i = {c: i for i, c in enumerate(chars_u)}
        for off in (-10, -5, 5, 10):
            aa, bb = big[:-off], big[off:]
            mm = np.isin(aa, chars_u) & np.isin(bb, chars_u)
            for xx, yy in zip(aa[mm], bb[mm]):
                co[c2i[int(xx)], c2i[int(yy)]] += 1
        cnt = np.array([np.sum(arr == c) for c in chars_u], dtype=float)
        pp = co / max(co.sum(), 1)
        indep = np.outer(cnt, cnt) / max((cnt.sum() ** 2), 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            PMI = np.log(pp / indep)
        PMI[~np.isfinite(PMI)] = 0.0
        # map bigram -> its char pair indices (only pairs in chars_u)
        idx_map = [(c2i.get(x, -1), c2i.get(y, -1)) for x, y, _ in picked]
        sel = [i for i, (ix, iy) in enumerate(idx_map) if ix >= 0 and iy >= 0]
        if len(sel) > 10:
            Ds = Db[np.ix_(sel, sel)]
            P = PMI[np.ix_([idx_map[i][0] for i in sel],
                           [idx_map[i][1] for i in sel])]
            iu2 = np.triu_indices(len(sel), 1)
            sp = spearman(Ds[iu2], P[iu2])
        else:
            sp = float("nan")
        if prev_code is not None:
            out["steps"].append(st)
            out["stability"].append(round(stability, 4))
            out["spearman_pmi"].append(round(sp, 4))
            print(f"step {st}: stability={stability:.3f} spearman={sp:.3f}", flush=True)
        prev_code = C_q
    with open(os.path.join(ROOT, "logs_v2", "dev_timing_result.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved logs_v2/dev_timing_result.json", flush=True)


if __name__ == "__main__":
    from code_address import jaccard_matrix
    main()
