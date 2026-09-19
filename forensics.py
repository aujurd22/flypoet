"""Experiment: single-window contamination forensics via feature ensemble.

fam_nll showed per-window NLL alone separates seen/unseen weakly (AUC ~0.59).
Features that might each catch a different trace of memorization:
  f1 mean token NLL (std model)          — loss-based trace
  f2 mean energy keep-rate (e90 model)   — sparsity-based trace
  f3 max token NLL                       — local memorization spikes
  f4 bigram surrogate                    — n-gram overlap trace
Trained: logistic on 3000 seen + 3000 unseen; tested on disjoint 1000+1000.
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import train_v2 as T
from fam_heads import windows_from_stream, auc
from fam_nll import score as nll_score
from calibration_eval import build_bigram_model

DEV = "cuda"
SEQ = 192


@torch.no_grad()
def keep_scores(model, X):
    """Mean per-window energy keep-rate across blocks."""
    feats = []
    for i in range(0, len(X), 64):
        xb = X[i:i + 64].to(DEV)
        store = []

        def hook(mod, inp, out):
            x = inp[0].detach().float()
            mag = x.abs()
            tot = mag.sum(-1, keepdim=True) + 1e-8
            s, _ = torch.sort(mag, dim=-1, descending=True)
            cs = s.cumsum(-1)
            k = (cs < 0.90 * tot).sum(-1, keepdim=True).clamp(min=1)
            store.append((k.squeeze(-1).float() / x.shape[-1]).mean(dim=1).cpu())

        hs = [b.kwta.register_forward_hook(hook) for b in model.blocks]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(xb)
        for h in hs:
            h.remove()
        feats.append(torch.cat(store).mean(0))
    return torch.cat(feats).numpy()


@torch.no_grad()
def bigram_scores(logp, X):
    out = []
    for i in range(0, len(X), 256):
        xb = X[i:i + 256].numpy()
        for row in xb:
            lls = [logp.get((a, b), -12.0) for a, b in zip(row[:-1], row[1:])]
            out.append(float(np.mean(lls)))
    return np.array(out)


def main():
    corpus = T.Corpus()
    X_seen = windows_from_stream(corpus.train, 4000, seed=41)
    X_unseen = windows_from_stream(corpus.val, 4000, seed=42)

    print("loading models...", flush=True)
    std = T.GPT(corpus.V).to(DEV)
    std.load_state_dict(torch.load(os.path.join(ROOT, "logs_v2", "std_model.pt"),
                                   map_location=DEV, weights_only=True))
    std.eval()
    e90 = T.GPT(corpus.V, kwta_opts={"impl": "energy", "e_frac": 0.90}).to(DEV)
    e90.load_state_dict(torch.load(os.path.join(ROOT, "logs_v2", "flynetS_e90_model.pt"),
                                   map_location=DEV, weights_only=True))
    e90.eval()

    print("feature 1/4: mean NLL...", flush=True)
    f1_s = np.array(nll_score(std, corpus, X_seen))
    f1_u = np.array(nll_score(std, corpus, X_unseen))
    print("feature 2/4: keep-rate...", flush=True)
    f2_s = keep_scores(e90, X_seen)
    f2_u = keep_scores(e90, X_unseen)
    print("feature 3/4: max NLL...", flush=True)

    @torch.no_grad()
    def maxnll(X):
        out = []
        for i in range(0, len(X), 64):
            xb = X[i:i + 64].to(DEV)
            yb = xb.clone()
            yb[:, :-1] = xb[:, 1:]
            yb[:, -1] = xb[:, 0]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = std(xb)
            ce = F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                                 yb.view(-1), reduction="none")
            out.append(ce.view(xb.shape[0], -1).max(dim=1).values.cpu())
        return torch.cat(out).numpy()

    f3_s = maxnll(X_seen)
    f3_u = maxnll(X_unseen)
    print("feature 4/4: bigram...", flush=True)
    big, uni, Vb = build_bigram_model(corpus.train[: 3_000_000])
    import math
    logp = {(a, b): math.log((c + 0.1) / (uni[a] + 0.1 * Vb))
            for (a, b), c in big.items()}
    f4_s = bigram_scores(logp, X_seen)
    f4_u = bigram_scores(logp, X_unseen)

    def split(f):
        return f[:3000], f[3000:]

    feats = {"nll": (f1_s, f1_u, -1), "keep": (f2_s, f2_u, 1),
             "maxnll": (f3_s, f3_u, -1), "bigram": (f4_s, f4_u, -1)}
    print("--- single features (test AUC) ---", flush=True)
    test_scores = {}
    for name, (fs, fu, sign) in feats.items():
        a = auc(sign * fs[3000:], sign * fu[3000:])
        test_scores[name] = round(a, 4)
        print(f"{name}: {a:.4f}", flush=True)

    # logistic ensemble on standardized features
    def mat(f_list):
        M = np.stack(f_list, 1)
        return (M - M[:3000].mean(0)) / (M[:3000].std(0) + 1e-6)

    Ms, Mu = mat([feats[n][:2][0] for n in feats]), mat([feats[n][:2][1] for n in feats])
    Xtr = torch.tensor(np.concatenate([Ms[:3000], Mu[:3000]]), dtype=torch.float32).to(DEV)
    ytr = torch.cat([torch.ones(3000), torch.zeros(3000)]).to(DEV)
    Xte = torch.tensor(np.concatenate([Ms[3000:], Mu[3000:]]), dtype=torch.float32).to(DEV)
    yte = torch.cat([torch.ones(1000), torch.zeros(1000)]).to(DEV)
    torch.manual_seed(0)
    w = torch.zeros(Xtr.shape[1], device=DEV, requires_grad=True)
    b = torch.zeros(1, device=DEV, requires_grad=True)
    opt = torch.optim.AdamW([w, b], lr=0.05, weight_decay=1e-2)
    for _ in range(800):
        idx = torch.randint(0, len(Xtr), (1024,), device=DEV)
        out = (Xtr[idx] @ w + b).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(out, ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        s_te = (Xte @ w + b).squeeze(-1).cpu().numpy()
        s_tr = (Xtr @ w + b).squeeze(-1).cpu().numpy()
    a_te = auc(s_te[yte == 1], s_te[yte == 0])
    a_tr = auc(s_tr[:3000], s_tr[3000:])
    print(f"ensemble: train-AUC {a_tr:.4f} test-AUC {a_te:.4f}", flush=True)
    test_scores["ensemble_test"] = round(a_te, 4)

    with open(os.path.join(ROOT, "logs_v2", "forensics_result.json"), "w") as f:
        json.dump(test_scores, f, indent=1)
    print("saved logs_v2/forensics_result.json", flush=True)


if __name__ == "__main__":
    main()
