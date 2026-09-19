"""Experiment: is the k-WTA sparse code a memory address?

Fly claim: Kenyon-cell sparse patterns ARE associative-memory addresses —
stable per concept, mutually separated, completable from partial cues.

Tests on the k25-24k checkpoint (char-level, top-25% of final hidden):
  A. address stability  — Jaccard overlap of active sets for the SAME char
     across independent contexts vs matched DIFFERENT-char pairs.
  B. semantic metric    — Spearman(code-Hamming similarity, char PMI) vs
     Spearman(dense cosine, PMI): which representation tracks co-occurrence.
  C. code completion    — Jaccard(code@half-context, code@full-context) of the
     same window vs the random-pair baseline (does the code fill itself in).
"""
import json, os, sys, itertools
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import train_v2 as T

DEV = "cuda"
WIN = 64
N_CONCEPTS = 150
OCC_PER = 5


@torch.no_grad()
def codes_for(model, corpus, positions, ctx_len=32):
    """Sparse (top-25%) and dense code at the position RIGHT AFTER seeing the
    center char, for windows centered on each position."""
    xs = []
    for p in positions:
        seg = corpus.train[max(0, p - ctx_len + 1): p + 1]
        if len(seg) < ctx_len:
            seg = np.concatenate([np.zeros(ctx_len - len(seg), dtype=np.int64), seg])
        xs.append(torch.from_numpy(np.array(seg[-WIN:], dtype=np.int64)))
    X = torch.stack(xs).to(DEV)
    out_codes, out_dense = [], []
    for i in range(0, len(X), 64):
        xb = X[i:i + 64]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(xb, collect_final=True)
        h = model._last_hidden[:, -1].float().cpu()      # code at last position
        thr = torch.kthvalue(h, int(h.shape[-1] * 0.75), dim=-1, keepdim=True).values
        out_codes.append((h >= thr))
        out_dense.append(h)
    return torch.cat(out_codes), torch.cat(out_dense)


def jaccard_matrix(C):
    """Pairwise Jaccard of boolean active sets (n, d) -> (n, n)."""
    C = C.float()
    inter = C @ C.T
    sums = C.sum(1, keepdim=True)
    union = sums + sums.T - inter
    return inter / union.clamp(min=1)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "flynetS_k25_24k"
    torch.manual_seed(7)
    corpus = T.Corpus()
    model = T.GPT(corpus.V, kwta_opts={"impl": "torch", "k_frac": 0.25}).to(DEV)
    sd = torch.load(os.path.join(ROOT, "logs_v2", f"{which}_model.pt"),
                    map_location=DEV, weights_only=True)
    model.load_state_dict(sd)
    model.eval()

    # pick frequent chars as concepts
    arr = corpus.train[: 2_000_000]
    counts = np.bincount(arr, minlength=corpus.V)
    cand = np.argsort(counts)[::-1]
    cand = [c for c in cand if counts[c] > 300][:N_CONCEPTS]

    rng = np.random.default_rng(5)
    positions, owner = [], []
    for ci, c in enumerate(cand):
        hits = np.where(arr == c)[0]
        hits = hits[(hits > WIN) & (hits < len(arr) - WIN)]
        if len(hits) < OCC_PER:
            continue
        take = rng.choice(hits, size=OCC_PER, replace=False)
        for p in take:
            positions.append(int(p) + WIN - 1)
            owner.append(ci)
    positions = np.array(positions[: N_CONCEPTS * OCC_PER])
    owner = np.array(owner[: len(positions)])
    print(f"concepts={len(set(owner.tolist()))} windows={len(positions)}", flush=True)

    C, H = codes_for(model, corpus, positions)
    J = jaccard_matrix(C).numpy()
    same, diff = [], []
    n = len(owner)
    for i in range(n):
        for j in range(i + 1, n):
            if owner[i] == owner[j]:
                same.append(J[i, j])
    rng2 = np.random.default_rng(0)
    while len(diff) < len(same):
        i, j = rng2.integers(0, n, 2)
        if owner[i] != owner[j]:
            diff.append(J[i, j])
    same_m, diff_m = float(np.mean(same)), float(np.mean(diff))
    print(f"stability: same-char Jaccard {same_m:.3f} vs diff-char {diff_m:.3f} "
          f"(lift {same_m / max(diff_m, 1e-6):.2f}x)", flush=True)

    # B: one code per concept (first occurrence) vs PMI
    rep = {}
    for k, ci in enumerate(owner.tolist()):
        rep.setdefault(ci, k)
    reps = sorted(rep.keys())
    idx = np.array([rep[c] for c in reps])
    Cc, Hc = C[idx], H[idx]
    chars = [cand[c] for c in reps]

    big = arr[: 1_500_000]
    pmi_n = len(chars)
    ch2i = {ch: i for i, ch in enumerate(chars)}
    co = np.zeros((pmi_n, pmi_n))
    cnt = np.array([counts[c] for c in chars], dtype=float)
    for off in range(-10, 11):
        if off == 0:
            continue
        a, b = big[:-off], big[off:]
        m = np.isin(a, chars) & np.isin(b, chars)
        for x_, y_ in zip(a[m], b[m]):
            co[ch2i[x_], ch2i[y_]] += 1
    p = co / co.sum()
    indep = np.outer(cnt / cnt.sum(), cnt / cnt.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        PMI = np.log(p / indep)
    PMI[~np.isfinite(PMI)] = 0.0

    Dc = jaccard_matrix(Cc).numpy()
    Hn = Hc / Hc.norm(dim=1, keepdim=True).clamp(min=1e-6)
    cos = (Hn @ Hn.T).numpy()
    iu = np.triu_indices(pmi_n, 1)
    pm = PMI[iu]
    s_code = spearman(Dc[iu], pm)
    s_dense = spearman(cos[iu], pm)
    print(f"semantic: spearman(code,PMI)={s_code:.3f} vs spearman(dense,PMI)={s_dense:.3f}",
          flush=True)

    # C: completion — half-context code vs full-context code of same window
    pos2 = positions[rng2.choice(len(positions), size=200, replace=False)]
    C_half, _ = codes_for(model, corpus, pos2 - 16, ctx_len=16)
    C_full, _ = codes_for(model, corpus, pos2)
    Jc = jaccard_matrix(torch.cat([C_half, C_full])).numpy()
    nn = len(pos2)
    self_comp = np.mean([Jc[i, nn + i] for i in range(nn)])
    rnd = np.mean([Jc[i, nn + (i + 7) % nn] for i in range(nn)])
    print(f"completion: self {self_comp:.3f} vs mismatched {rnd:.3f}", flush=True)

    out = {"checkpoint": which, "stability_same": round(same_m, 4),
           "stability_diff": round(diff_m, 4), "lift": round(same_m / max(diff_m, 1e-6), 3),
           "spearman_code_pmi": round(s_code, 4), "spearman_dense_pmi": round(s_dense, 4),
           "completion_self": round(float(self_comp), 4),
           "completion_mismatch": round(float(rnd), 4)}
    with open(os.path.join(ROOT, "logs_v2", "code_address_result.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved logs_v2/code_address_result.json", flush=True)


if __name__ == "__main__":
    main()
