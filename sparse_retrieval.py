"""Experiment: sparse code as a retrieval address.

code_address.py showed k-WTA codes are stable and completable but weak on
PMI semantics. This test asks the practical follow-up: can a Hamming lookup
on those codes retrieve useful continuations, compared against dense-cosine
lookup on the same stored windows?

Address book: 6000 train windows (64 tok), keyed by the top-25% code (and by
the dense vector) at the window's last position; payload = the next char.
Queries: 1000 val windows, same construction, held-out.
Metrics: next-char hit@1 / hit@10 for Hamming-NN vs cosine-NN, plus
frequency-matched baselines.
"""
import json, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import train_v2 as T

DEV = "cuda"
SEQ = 64
N_STORE = 6000
N_QUERY = 1000


@torch.no_grad()
def build(model, arr, positions):
    codes, dense, nxt = [], [], []
    for i in range(0, len(positions), 64):
        ps = positions[i:i + 64]
        X = []
        for p in ps:
            seg = arr[max(0, p - SEQ + 1): p + 1]
            if len(seg) < SEQ:
                seg = np.concatenate([np.zeros(SEQ - len(seg), dtype=np.int64), seg])
            X.append(torch.from_numpy(np.array(seg[-SEQ:], dtype=np.int64)))
        xb = torch.stack(X).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(xb, collect_final=True)
        h = model._last_hidden[:, -1].float().cpu()
        thr = torch.kthvalue(h, int(h.shape[-1] * 0.75), dim=-1, keepdim=True).values
        codes.append(h >= thr)
        dense.append(h)
        nxt.extend(int(arr[p + 1]) for p in ps)
    return torch.cat(codes), torch.cat(dense), np.array(nxt)


def topk_hits(qm, sm, qv, sv, k_list):
    """Return hit@k lists for binary (Hamming) or float (cosine) matrices."""
    if qm.dtype == torch.bool:
        d = (qm.sum(1, keepdim=True) + sm.sum(1).unsqueeze(0)
             - 2.0 * (qm.float() @ sm.float().T))       # Hamming (q, s)
    else:
        qn = qv / qv.norm(dim=1, keepdim=True).clamp(min=1e-6)
        sn = sv / sv.norm(dim=1, keepdim=True).clamp(min=1e-6)
        d = 1.0 - qn @ sn.T                              # cosine distance
    hits = {}
    for k in k_list:
        idx = d.argsort(dim=1)[:, :k]                    # (nq, k) store idx
        hits[k] = idx
    return hits


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "flynetS_k25_24k"
    torch.manual_seed(7)
    corpus = T.Corpus()
    model = T.GPT(corpus.V, kwta_opts={"impl": "torch", "k_frac": 0.25}).to(DEV)
    sd = torch.load(os.path.join(ROOT, "logs_v2", f"{which}_model.pt"),
                    map_location=DEV, weights_only=True)
    model.load_state_dict(sd)
    model.eval()

    rng = np.random.default_rng(9)
    tr = corpus.train
    va = corpus.val
    store_pos = rng.integers(SEQ, len(tr) - 2, N_STORE)
    query_pos = rng.integers(SEQ, len(va) - 2, N_QUERY)

    print("building address book (store)...", flush=True)
    S_code, S_dense, S_next = build(model, tr, store_pos)
    print("building queries...", flush=True)
    Q_code, Q_dense, Q_next = build(model, va, query_pos)

    freq = np.bincount(S_next, minlength=corpus.V).astype(float)
    freq_char = int(freq.argmax())

    out = {"checkpoint": which, "n_store": N_STORE, "n_query": N_QUERY}
    for name, qm, sm, qv, sv, sign in [
            ("hamming", Q_code, S_code, None, None, 1),
            ("cosine", None, None, Q_dense, S_dense, -1)]:
        hits = topk_hits(qm, sm, qv, sv, [1, 10])
        for k in (1, 10):
            idx = hits[k].numpy()
            pred = S_next[idx]                       # (nq, k) retrieved chars
            true = Q_next[:, None]
            hit = (pred == true).any(axis=1).mean() if k > 1 else (pred[:, 0] == true[:, 0]).mean()
            out[f"{name}_hit{k}"] = round(float(hit), 4)
            print(f"{name} hit@{k}: {hit:.4f}", flush=True)
    # baselines
    out["freq_baseline_hit1"] = round(float((Q_next == freq_char).mean()), 4)
    rnd = float(np.mean([ (S_next[rng.integers(0, N_STORE, 10)] == q).any()
                          for q in Q_next ]))
    out["random_hit10"] = round(rnd, 4)
    print(f"freq-baseline hit@1: {out['freq_baseline_hit1']}", flush=True)
    print(f"random hit@10: {out['random_hit10']}", flush=True)

    with open(os.path.join(ROOT, "logs_v2", "sparse_retrieval_result.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved logs_v2/sparse_retrieval_result.json", flush=True)


if __name__ == "__main__":
    main()
