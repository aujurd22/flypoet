"""D1 experiment (bypass): trained sparse-projection memory table for
flymemory-style entries — k-WTA on momentum-averaged projections ("MA table").

Hypothesis under test (user's, from 5 days ago): a TRAINED sparse projection
(k-WTA on top of a learned linear map, as in FlyPoet blocks) beats the
hashing encoder on retrieval quality at the same sparsity (5%) and bit
budget (4096).

Bypass constraint: reads ONLY flymemory's exported JSONL (no imports of its
code, no writes to its directory); everything lives in flypoet.

Pipeline:
  1. load flymemory export (_fm_export.json if present, else synthesize
     domain-tagged text pairs from decide_data as a stand-in corpus)
  2. corpus = (context, response, label) triples
  3. baselines:
     a. hash encoder (flymemory's exact algorithm, reimplemented here)
     b. random projection + k-WTA 5% (untrained sparse, same budget)
  4. D1: linear 384->4096 trained with contrastive loss (InfoNCE) on
     context->response pairs; k-WTA 5% on the projected features ("MA table"
     when weights are an EMA of training — we also track the EMA variant)
  5. retrieval: query=context, key=response-code; hit@1/@10; plus
     interference probe (near-duplicate discrimination)
Pre-registered verdicts:
  D1-hit@1 > hash+0.05  -> trained projection wins, adopt
  |D1-hash| <= 0.05     -> parity, hashing stays (simpler)
  D1 < hash             -> training does not transfer to this data scale
"""
import json, os, sys, hashlib
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_BITS = 4096
SPARSITY = 0.05
DIM_IN = 384
N_PAIRS = 4000
N_TEST = 500
EPOCHS = 12
EMA_DECAY = 0.999


def hash_encode(text, n_bits=N_BITS, sparsity=SPARSITY, n_hashes=3):
    """flymemory's exact hashing algorithm (reimplemented for the bypass)."""
    code = np.zeros(n_bits, dtype=np.float32)
    for word in text.lower().split():
        for h in range(n_hashes):
            hsh = int(hashlib.md5(f"{word}_{h}".encode()).hexdigest(), 16)
            code[hsh % n_bits] += 1.0
    k = max(int(n_bits * sparsity), 1)
    if k > 0 and code.max() > 0:
        thresh = np.partition(code, -k)[-k]
        code = np.where(code >= thresh, 1.0, 0.0)
    return code


def load_export():
    """Prefer flymemory's real export; else synthesize from decide_data."""
    fp = os.path.join(ROOT, "_fm_export.json")
    if os.path.exists(fp):
        data = json.load(open(fp, encoding="utf-8"))
        triples = [(d["context"], d["response"], d.get("label", 0))
                   for d in data if d.get("context") and d.get("response")]
        if len(triples) >= N_PAIRS:
            return triples[: N_PAIRS + N_TEST]
    # stand-in: decide_data rows give (text, label); context=first 100 chars,
    # response=next 100 chars of the same doc — co-reference is the task
    rows = []
    fp = os.path.join(ROOT, "decide_data", "train.jsonl")
    with open(fp, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ids = d["ids"]
            if len(ids) < 200:
                continue
            rows.append((ids[:100], ids[100:200], d["label"]))
    return rows


def ids_to_text(ids):
    """Decode char ids back to text via vocab (for the hashing baseline)."""
    import json as J
    vj = J.load(open(os.path.join(ROOT, "data_v2", "vocab.json"),
                     encoding="utf-8"))
    itos = vj["itos"]
    return "".join(itos.get(str(i), "") for i in ids)


class Proj(torch.nn.Module):
    def __init__(self, dim_in=DIM_IN, n_bits=N_BITS):
        super().__init__()
        self.w = torch.nn.Linear(dim_in, n_bits, bias=False)

    def forward(self, x):
        return self.w(x)


def kwta(x, k):
    thr = torch.kthvalue(x, x.shape[-1] - k + 1, dim=-1, keepdim=True).values
    return x * (x >= thr)


def main():
    triples = load_export()
    print(f"corpus triples: {len(triples)}", flush=True)
    tr = triples[:N_PAIRS]
    te = triples[N_PAIRS:N_PAIRS + N_TEST]

    # context/response embeddings: hashed char 8-gram bag (fixed "frozen" side)
    def bag(ids):
        v = np.zeros(DIM_IN, dtype=np.float32)
        for i in range(0, len(ids) - 7, 4):
            hsh = int(hashlib.md5(str(tuple(ids[i:i + 8])).encode()).hexdigest(), 16)
            v[hsh % DIM_IN] += 1.0
        return v / (np.linalg.norm(v) + 1e-9)

    Xc_tr = np.stack([bag(t[0]) for t in tr])
    Xr_tr = np.stack([bag(t[1]) for t in tr])
    y_tr = np.array([t[2] for t in tr])
    Xc_te = np.stack([bag(t[0]) for t in te])
    Xr_te = np.stack([bag(t[1]) for t in te])
    y_te = np.array([t[2] for t in te])

    k = max(int(N_BITS * SPARSITY), 1)

    # ---- baseline a: hash encoder on decoded text
    vj = json.load(open(os.path.join(ROOT, "data_v2", "vocab.json"),
                        encoding="utf-8"))
    itos = vj["itos"]
    H_r_tr = np.stack([hash_encode(ids_to_text(t[1])) for t in tr])
    H_r_te = np.stack([hash_encode(ids_to_text(t[1])) for t in te])

    # ---- baseline b: random projection + kwta
    g = np.random.default_rng(7)
    Rp = g.normal(0, 1 / np.sqrt(DIM_IN), (DIM_IN, N_BITS)).astype(np.float32)
    Rp_codes_tr = kwta(torch.from_numpy(Xc_tr @ Rp), k).numpy()
    Rp_codes_te = kwta(torch.from_numpy(Xc_te @ Rp), k).numpy()

    # ---- D1: trained projection, InfoNCE context->response
    torch.manual_seed(7)
    proj = Proj().to(DEV)
    ema_w = proj.w.weight.detach().clone()
    opt = torch.optim.AdamW(proj.parameters(), lr=3e-4, weight_decay=1e-4)
    Xc_t = torch.from_numpy(Xc_tr).to(DEV)
    Xr_t = torch.from_numpy(Xr_tr).to(DEV)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xc_t), device=DEV)
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            c = proj(Xc_t[idx])
            r = proj(Xr_t[idx])
            c = kwta(F.relu(c), k)
            r = kwta(F.relu(r), k)
            cn = F.normalize(c, dim=-1)
            rn = F.normalize(r, dim=-1)
            logits = cn @ rn.T / 0.07
            tgt = torch.arange(len(idx), device=DEV)
            loss = F.cross_entropy(logits, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            ema_w.mul_(EMA_DECAY).add_(proj.w.weight.detach(), alpha=1 - EMA_DECAY)
        if (ep + 1) % 4 == 0:
            print(f"epoch {ep+1}: loss {loss.item():.4f}", flush=True)

    # codes: current weights and EMA weights (the "MA table")
    def encode_codes(X, weights):
        with torch.no_grad():
            z = kwta(F.relu(X @ weights.T), k)
        return z.cpu().numpy()

    D1_tr = encode_codes(Xc_t, proj.w.weight.detach())
    D1_te = encode_codes(torch.from_numpy(Xc_te).to(DEV), proj.w.weight.detach())
    EMA_tr = encode_codes(Xc_t, ema_w)
    EMA_te = encode_codes(torch.from_numpy(Xc_te).to(DEV), ema_w)

    def hits(qc, sc):
        out = {}
        for kk in (1, 10):
            d = -(qc @ sc.T)
            idx = np.argsort(d, axis=1)[:, :kk]
            hit = float(np.mean([np.isin(y_te[i], y_tr[idx[i]]).any()
                                 for i in range(len(qc))]))
            out[f"hit{kk}"] = round(hit, 4)
        return out

    # retrieval: query code -> nearest store codes -> any same-label = hit
    res = {}
    res["hash"] = hits(H_r_te, H_r_tr)
    res["random_kwta"] = hits(Rp_codes_te, Rp_codes_tr)
    res["D1_trained"] = hits(D1_te, D1_tr)
    res["D1_ema"] = hits(EMA_te, EMA_tr)
    res["chance"] = round(float(np.mean([np.isin(y, y_tr).any() / len(np.unique(y_tr))
                                         for y in y_te])), 4)
    for name, r in res.items():
        print(f"{name}: {r}", flush=True)

    with open(os.path.join(ROOT, "logs_v2", "ma_table_d1_result.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("saved logs_v2/ma_table_d1_result.json", flush=True)


if __name__ == "__main__":
    main()
