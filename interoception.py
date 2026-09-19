"""Experiment: is per-token activation rate a free self-monitoring signal?

The energy k-WTA model picks, per token and block, how many channels to keep
(smallest set holding 90% of |x| energy). Question: does that chosen rate
carry information the softmax confidence does not?

Metrics (per token, rate = mean keep fraction across blocks):
  1. Spearman(rate, token NLL) on train and val streams
  2. AUC(rate) for predicting "token in top-quartile NLL" — vs the same AUC
     using softmax confidence (lower conf should predict high loss too)
  3. OOD: AUC(rate) separating train tokens from val tokens — vs conf AUC
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import train_v2 as T

DEV = "cuda"
SEQ = 128
N_WIN = 1500
E_FRAC = 0.90


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def auc(pos, neg):
    s = np.concatenate([pos, neg])
    order = np.argsort(np.argsort(s)).astype(float)
    r_pos = order[: len(pos)].sum()
    n_p, n_n = len(pos), len(neg)
    return float((r_pos - n_p * (n_p - 1) / 2) / (n_p * n_n))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "flynetS_e90"
    corpus = T.Corpus()
    model = T.GPT(corpus.V, kwta_opts={"impl": "energy", "e_frac": E_FRAC}).to(DEV)
    sd = torch.load(os.path.join(ROOT, "logs_v2", f"{which}_model.pt"),
                    map_location=DEV, weights_only=True)
    model.load_state_dict(sd)
    model.eval()

    rates_store = {}

    def mk_hook(key):
        def hook(mod, inp, out):
            x = inp[0].detach().float()
            mag = x.abs()
            tot = mag.sum(-1, keepdim=True) + 1e-8
            s, _ = torch.sort(mag, dim=-1, descending=True)
            cs = s.cumsum(-1)
            k = (cs < E_FRAC * tot).sum(-1, keepdim=True).clamp(min=1)
            rates_store.setdefault(key, []).append(
                (k.squeeze(-1).float() / x.shape[-1]).cpu())
        return hook

    handles = [b.kwta.register_forward_hook(mk_hook(i))
               for i, b in enumerate(model.blocks)]

    def stream_tokens(arr, tag):
        torch.manual_seed(hash(tag) % 2**31)
        rates, nlls = [], []
        for i in range(0, N_WIN, 64):
            ix = torch.randint(len(arr) - SEQ - 1, (min(64, N_WIN - i),))
            x = torch.stack([torch.from_numpy(arr[p:p + SEQ]) for p in ix]).to(DEV)
            y = torch.stack([torch.from_numpy(arr[p + 1:p + 1 + SEQ]) for p in ix]).to(DEV)
            rates_store.clear()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
            ce = F.cross_entropy(logits.float().transpose(1, 2), y, reduction="none")
            rate = torch.stack([torch.cat(v) for v in rates_store.values()]).mean(0)
            rates.append(rate.reshape(-1))
            nlls.append(ce.reshape(-1).cpu())
        return torch.cat(rates).numpy(), torch.cat(nlls).numpy()

    r_tr, n_tr = stream_tokens(corpus.train, "train")
    r_va, n_va = stream_tokens(corpus.val, "val")

    def topq_auc(r, n):
        thr = np.quantile(n, 0.75)
        hi = n >= thr
        return auc(-r[hi], -r[~hi])  # low rate should predict high loss

    res = {
        "spearman_rate_nll_train": round(spearman(r_tr, n_tr), 4),
        "spearman_rate_nll_val": round(spearman(r_va, n_va), 4),
        "auc_rate_highloss_train": round(topq_auc(r_tr, n_tr), 4),
        "auc_rate_highloss_val": round(topq_auc(r_va, n_va), 4),
        "auc_rate_ood": round(auc(r_tr, r_va), 4),
        "mean_rate_train": round(float(r_tr.mean()), 4),
        "mean_rate_val": round(float(r_va.mean()), 4),
    }
    for k, v in res.items():
        print(f"{k}: {v}", flush=True)
    with open(os.path.join(ROOT, "logs_v2", "interoception_result.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("saved logs_v2/interoception_result.json", flush=True)


if __name__ == "__main__":
    main()
