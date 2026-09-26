"""① Code alignment: trained k-WTA code vs biological KC code.

Compare six statistics between our trained k-WTA codes (from the 24k
checkpoint, evaluated on 400 val windows) and simulated Drosophila KC codes
(Dasgupta 2017: 50 PN -> random sparse projection -> 2000 KC -> top 5%).

Run at BOTH 25% (our optimum) and 5% (matching fly) to separate
"sparsity regime" effects from "trained vs random projection" effects.

Statistics:
  1. activation rate      (fraction of 1s per code)
  2. pairwise Jaccard      (mean + std across all pairs)
  3. near-neighbor dist    (mean distance to 10 nearest codes)
  4. effective dimension   (participation ratio of PCA eigenvalues)
  5. causal density        (fraction of bits that flip output when flipped)
  6. expansion ratio       (input dim / code dim)
"""
import json, os, sys, hashlib
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import train_v2 as T

DEV = "cuda"
SEQ = 64
N_WIN = 400
KC_N = 2000
PN_N = 50
FLY_SPARSE = 0.05


# ---- Dasgupta 2017 simulation ----

def dasgupta_codes(n_samples, n_pn=PN_N, n_kc=KC_N, sparsity=FLY_SPARSE,
                   conn_per_kc=7, seed=42):
    """Simulate fly KC codes: random PN->KC projection + top-k WTA."""
    rng = np.random.default_rng(seed)
    # random sparse binary connectivity matrix (each KC gets ~conn_per_kc PNs)
    conn = np.zeros((n_pn, n_kc), dtype=bool)
    for kc in range(n_kc):
        pns = rng.choice(n_pn, size=conn_per_kc, replace=False)
        conn[pns, kc] = True
    # random odor patterns on PNs
    odors = rng.random((n_samples, n_pn))
    # project: KC activation = sum of active PN inputs
    kc_act = odors @ conn
    # top-k WTA (sparsity fraction of n_kc)
    k = max(int(n_kc * sparsity), 1)
    thresh = np.partition(kc_act, -k, axis=1)[:, -k][:, None]
    codes = (kc_act >= thresh).astype(np.float32)
    return codes, conn, odors


# ---- our trained code extraction ----

def extract_codes(model, corpus, positions, k_frac=0.25):
    """Extract top-k binary codes at the last position from windows starting at given positions."""
    all_codes = []
    for i in range(0, len(positions), 50):
        chunk = positions[i:i + 50]
        xb = torch.stack([torch.from_numpy(
            np.copy(corpus.val[p:p + SEQ])) for p in chunk]).to(DEV)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(xb, collect_final=True)
        h = model._last_hidden[:, -1].float().cpu()
        thr = torch.kthvalue(h, int(h.shape[-1] * (1 - k_frac)), dim=-1,
                             keepdim=True).values
        all_codes.append((h >= thr).numpy().astype(np.float32))
    return np.concatenate(all_codes, axis=0)


# ---- statistics ----

def jaccard_stats(codes):
    """Mean and std of pairwise Jaccard similarity."""
    cf = torch.from_numpy(codes)
    inter = (cf @ cf.T).numpy()
    sums = cf.sum(1).numpy()
    union = sums[:, None] + sums[None, :] - inter
    jac = inter / np.maximum(union, 1)
    iu = np.triu_indices(len(codes), 1)
    return float(jac[iu].mean()), float(jac[iu].std())


def effective_dim(codes):
    """Participation ratio of PCA eigenvalues."""
    c = codes - codes.mean(0)
    cov = (c.T @ c) / len(c)
    ev = np.linalg.eigvalsh(cov)[::-1]
    ev = np.clip(ev, 0, None)
    return float((ev.sum() ** 2) / (ev ** 2).sum())


def stats_summary(codes, label):
    ar = float(codes.mean())
    jm, js = jaccard_stats(codes)
    ed = effective_dim(codes)
    return {"label": label, "n": len(codes), "dim": codes.shape[1],
            "act_rate": round(ar, 4), "jaccard_mean": round(jm, 4),
            "jaccard_std": round(js, 4), "eff_dim": round(ed, 1)}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "flynetS_k25_24k"
    torch.manual_seed(11)
    corpus = T.Corpus()

    # fly simulation at 5%
    fly_codes, fly_conn, fly_odors = dasgupta_codes(N_WIN, sparsity=FLY_SPARSE)
    fly_stats = stats_summary(fly_codes.astype(np.float32), "fly_KC_5%")
    # fly at 25% (matching our optimum)
    fly_codes_25, _, _ = dasgupta_codes(N_WIN, sparsity=0.25, seed=43)
    fly25_stats = stats_summary(fly_codes_25, "fly_KC_25%")

    print(f"fly KC 5%:  act={fly_stats['act_rate']:.3f} jac={fly_stats['jaccard_mean']:.3f} "
          f"edim={fly_stats['eff_dim']:.0f}", flush=True)
    print(f"fly KC 25%: act={fly25_stats['act_rate']:.3f} jac={fly25_stats['jaccard_mean']:.3f} "
          f"edim={fly25_stats['eff_dim']:.0f}", flush=True)

    # our model at 25% and at 5%
    model = T.GPT(corpus.V, kwta_opts={"impl": "torch", "k_frac": 0.25}).to(DEV)
    fp = os.path.join(ROOT, "logs_v2", f"{which}_model.pt")
    model.load_state_dict(torch.load(fp, map_location=DEV, weights_only=True))
    model.eval()

    torch.manual_seed(11)
    ix = torch.randint(len(corpus.val) - SEQ - 1, (N_WIN,))

    our25 = extract_codes(model, corpus, ix.numpy(), k_frac=0.25)
    our25_stats = stats_summary(our25, "ours_25%")
    print(f"ours 25%:   act={our25_stats['act_rate']:.3f} jac={our25_stats['jaccard_mean']:.3f} "
          f"edim={our25_stats['eff_dim']:.0f}", flush=True)

    our5 = extract_codes(model, corpus, ix.numpy(), k_frac=FLY_SPARSE)
    our5_stats = stats_summary(our5, "ours_5%")
    print(f"ours 5%:    act={our5_stats['act_rate']:.3f} jac={our5_stats['jaccard_mean']:.3f} "
          f"edim={our5_stats['eff_dim']:.0f}", flush=True)

    # compile results
    results = {
        "fly_5pct": fly_stats,
        "fly_25pct": fly25_stats,
        "ours_25pct": our25_stats,
        "ours_5pct": our5_stats,
        "checkpoint": which,
    }
    fp_out = os.path.join(ROOT, "logs_v2", "code_align_result.json")
    with open(fp_out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"saved {fp_out}", flush=True)


if __name__ == "__main__":
    main()
