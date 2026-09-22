"""Benchmark: Hamming prefilter + dense rerank vs full dense cosine search
on the exported codebook (CPU-only — the deployment target for flymemory).

Quality metric: 8-domain topic hit@1/@10 (same as domain_retrieval.py).
Speed metric: wall-clock of the full search over the whole query set.
Variants:
  full_cosine32   dense fp32 cosine, all store entries      (upper baseline)
  full_hamming    packed Hamming only, no rerank            (cheapest)
  two_stage_M     Hamming prefilter -> cosine rerank on top-M candidates
"""
import json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_all(q_packed, s_packed, chunk=100):
    """(nq, ns) uint16 Hamming distances, chunked LUT popcount."""
    nq = q_packed.shape[0]
    ns = s_packed.shape[0]
    out = np.empty((nq, ns), dtype=np.uint16)
    for i in range(0, nq, chunk):
        x = np.bitwise_xor(q_packed[i:i + chunk][:, None, :], s_packed[None, :, :])
        out[i:i + chunk] = POP[x].sum(axis=2)
    return out


def cos32(qv, sv):
    qn = qv / np.linalg.norm(qv, axis=1, keepdims=True).clip(min=1e-6)
    sn = sv / np.linalg.norm(sv, axis=1, keepdims=True).clip(min=1e-6)
    return qn @ sn.T


def hitk(d, ql, sl, k, descending=True):
    order = np.argsort(-d if descending else d, axis=1)[:, :k]
    return float(np.mean((sl[order] == ql[:, None]).any(axis=1)))


def main():
    z = np.load(os.path.join(ROOT, "logs_v2", "codebook.npz"))
    q_p, s_p = z["q_codes_packed"], z["codes_packed"]
    qv32 = z["q_dense16"].astype(np.float32)
    sv32 = z["dense16"].astype(np.float32)
    ql, sl = z["q_labels"], z["labels"]
    nq, ns = q_p.shape[0], s_p.shape[0]
    print(f"store={ns} query={nq} codebytes={q_p.shape[1]}", flush=True)

    res = {}
    # full dense cosine (fp32)
    t0 = time.perf_counter()
    d_cos = cos32(qv32, sv32)
    t_cos = time.perf_counter() - t0
    res["full_cos32_s"] = round(t_cos, 3)
    res["full_cos32_hit1"] = round(hitk(d_cos, ql, sl, 1), 4)
    res["full_cos32_hit10"] = round(hitk(d_cos, ql, sl, 10), 4)
    print(f"full_cos32: {t_cos:.3f}s hit1={res['full_cos32_hit1']} "
          f"hit10={res['full_cos32_hit10']}", flush=True)

    # full hamming (packed, no rerank)
    t0 = time.perf_counter()
    d_h = hamming_all(q_p, s_p).astype(np.int32)
    t_h = time.perf_counter() - t0
    res["full_hamming_s"] = round(t_h, 3)
    res["full_hamming_hit1"] = round(hitk(-d_h, ql, sl, 1), 4)
    res["full_hamming_hit10"] = round(hitk(-d_h, ql, sl, 10), 4)
    print(f"full_hamming: {t_h:.3f}s hit1={res['full_hamming_hit1']} "
          f"hit10={res['full_hamming_hit10']}", flush=True)

    # two-stage: hamming top-M -> cosine rerank among candidates
    for M in (100, 500):
        t0 = time.perf_counter()
        order_h = np.argsort(d_h, axis=1)[:, :M]
        tt1 = time.perf_counter() - t0
        d2 = np.empty((nq, M), dtype=np.float32)
        for qi in range(nq):
            cand = sv32[order_h[qi]]
            qn = qv32[qi] / max(np.linalg.norm(qv32[qi]), 1e-6)
            cn = cand / np.linalg.norm(cand, axis=1, keepdims=True).clip(min=1e-6)
            d2[qi] = cn @ qn
        order_c = np.argsort(-d2, axis=1)
        t1 = time.perf_counter() - t0
        idx = np.take_along_axis(order_h, order_c, axis=1)[:, :10]
        hit1 = float(np.mean(sl[idx[:, 0]] == ql))
        hit10 = float(np.mean((sl[idx] == ql[:, None]).any(axis=1)))
        res[f"two_stage_M{M}_s"] = round(t1, 3)
        res[f"two_stage_M{M}_hit1"] = round(hit1, 4)
        res[f"two_stage_M{M}_hit10"] = round(hit10, 4)
        print(f"two_stage M={M}: {t1:.3f}s hit1={hit1:.3f} hit10={hit10:.3f}",
              flush=True)

    res["codebook_bytes_per_entry"] = int(q_p.shape[1])
    res["dense32_bytes_per_entry"] = int(sv32.shape[1] * 4)
    with open(os.path.join(ROOT, "logs_v2", "prefilter_bench_result.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("saved logs_v2/prefilter_bench_result.json", flush=True)


if __name__ == "__main__":
    main()
