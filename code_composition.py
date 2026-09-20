"""Experiment: is the sparse address COMPOSITIONAL?

code_address showed codes are stable per char. The open question (gap in
2025-26 literature): does the code of a bigram xy relate to the codes of its
parts — i.e., is code(x then y) ≈ something built from code(x) and code(y)?

Metrics (top-25% codes at the position right after seeing the unit):
  m_self_x  Jaccard(code_xy, code_x)   x's own occurrence, one token earlier
  m_self_y  Jaccard(code_xy, code_y)   y alone (from other occurrences)
  m_union   Jaccard(code_xy, code_x ∪ code_y)
  m_rand    Jaccard(code_xy, union of two frequency-matched random char codes)
Compositionality ratio = m_union / m_rand  (>1 means binding structure).
"""
import json, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import train_v2 as T
from code_address import codes_for, jaccard_matrix

DEV = "cuda"
N_BIGRAMS = 80
OCC = 5


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "flynetS_k25_24k"
    torch.manual_seed(7)
    corpus = T.Corpus()
    model = T.GPT(corpus.V, kwta_opts={"impl": "torch", "k_frac": 0.25}).to(DEV)
    sd = torch.load(os.path.join(ROOT, "logs_v2", f"{which}_model.pt"),
                    map_location=DEV, weights_only=True)
    model.load_state_dict(sd)
    model.eval()

    arr = corpus.train[: 2_000_000]
    uni = np.bincount(arr, minlength=corpus.V)

    # frequent bigrams with distinct chars
    a, b = arr[:-1], arr[1:]
    m = (a != b)
    pairs, cnts = np.unique(np.stack([a[m], b[m]], 1), axis=0, return_counts=True)
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
    print(f"bigrams picked: {len(picked)}", flush=True)

    # collect codes: for each picked bigram occurrence we need
    #   code after y (pos p+1), code after x (pos p) — same occurrence —
    # plus separate occurrence codes for x and for y alone.
    pos_by, own_by = [], []
    for x, y, hits in picked:
        for p in hits:
            pos_by.append(int(p) + 1)          # right after y
            own_by.append(int(p))              # right after x (same window)
    C_by, _ = codes_for(model, corpus, pos_by)
    C_x, _ = codes_for(model, corpus, own_by)

    # y-alone and x-alone codes from other occurrences of each char
    chars = sorted({c for x, y, _ in picked for c in (x, y)})
    char_pos = {c: [] for c in chars}
    for c in chars:
        hits = np.where(arr == c)[0]
        hits = hits[(hits > 40) & (hits < len(arr) - 40)]
        take = rng.choice(hits, size=min(OCC, len(hits)), replace=False)
        for p in take:
            char_pos[c].append(int(p))
    all_pos, owner = [], []
    for ci, c in enumerate(chars):
        for p in char_pos[c]:
            all_pos.append(int(p))
            owner.append(ci)
    C_ch, _ = codes_for(model, corpus, np.array(all_pos))
    ch_code = {}
    for ci, c in enumerate(chars):
        ch_code[c] = C_ch[[k for k, o in enumerate(owner) if o == ci]].float().mean(0) > 0.5

    # assemble
    m_self_x, m_self_y, m_union, m_rand = [], [], [], []
    freq_rank = {int(c): r for r, c in enumerate(np.argsort(uni)[::-1])}
    for bi, (x, y, hits) in enumerate(picked):
        sl = slice(bi * OCC, (bi + 1) * OCC)
        B = C_by[sl].float().mean(0) > 0.5           # majority code after xy
        Xc = ch_code[x]
        Yc = ch_code[y]
        U = Xc | Yc
        m_self_x.append(float((B & Xc).sum() / max((B | Xc).sum(), 1)))
        m_self_y.append(float((B & Yc).sum() / max((B | Yc).sum(), 1)))
        m_union.append(float((B & U).sum() / max((B | U).sum(), 1)))
        # random frequent pair from the same char pool
        for _ in range(10):
            ri = rng.choice(len(chars), 2, replace=False)
            Ru = ch_code[chars[int(ri[0])]] | ch_code[chars[int(ri[1])]]
            m_rand.append(float((B & Ru).sum() / max((B | Ru).sum(), 1)))

    m_u, m_r = float(np.mean(m_union)), float(np.mean(m_rand))
    out = {"m_self_x": round(float(np.mean(m_self_x)), 4),
           "m_self_y": round(float(np.mean(m_self_y)), 4),
           "m_union": round(m_u, 4), "m_rand": round(m_r, 4),
           "compositionality_ratio": round(m_u / max(m_r, 1e-6), 3),
           "n_bigrams": len(picked)}
    print(json.dumps(out, ensure_ascii=False), flush=True)
    with open(os.path.join(ROOT, "logs_v2", "code_composition_result.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved logs_v2/code_composition_result.json", flush=True)


if __name__ == "__main__":
    main()
