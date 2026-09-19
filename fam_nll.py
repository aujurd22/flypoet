"""Familiarity via direct NLL: score = -mean per-token loss of the window.
If trunks memorized training text, seen windows score lower loss than unseen.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\user\flypoet")
import train_v2 as T
from fam_heads import windows_from_stream, auc

DEV = "cuda"
SEQ = 192


@torch.no_grad()
def score(model, corpus, X):
    """Per-WINDOW mean NLL (not per-batch) — contamination detection needs
    a score for every window."""
    out = []
    for i in range(0, len(X), 64):
        xb = X[i:i + 64].to(DEV)
        yb = xb.clone()
        yb[:, :-1] = xb[:, 1:]
        yb[:, -1] = xb[:, 0]  # last target unknown within window; rare noise
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(xb)
        ce = F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                             yb.view(-1), reduction="none")
        out.extend(ce.view(xb.shape[0], -1).mean(dim=1).tolist())
    return out


def main():
    corpus = T.Corpus()
    results = {}
    for arm in ["std", "flynetS"]:
        model = T.GPT(corpus.V, kwta_opts={"impl": "torch"} if arm == "flynetS" else None).to(DEV)
        sd = torch.load(rf"D:\user\flypoet\logs_v2\{arm}_model.pt",
                        map_location=DEV, weights_only=True)
        model.load_state_dict(sd)
        model.eval()
        X_seen = windows_from_stream(corpus.train, 1500, seed=31)
        X_unseen = windows_from_stream(corpus.val, 1500, seed=32)
        s_seen = torch.tensor(score(model, corpus, X_seen))
        s_unseen = torch.tensor(score(model, corpus, X_unseen))
        print(f"[{arm}] train-NLL {s_seen.mean():.4f}±{s_seen.std():.4f} | "
              f"val-NLL {s_unseen.mean():.4f}±{s_unseen.std():.4f}", flush=True)
        a = auc(-s_seen, -s_unseen)  # familiarity score = -NLL (low loss = familiar); pos=seen
        print(f"[{arm}] NLL-based familiarity AUC: {a:.4f}", flush=True)
        det = {}
        for p in [0.05, 0.10, 0.20, 0.50]:
            n_contam = int(p * 1000)
            Xc = torch.cat([X_seen[1000:1000 + n_contam], X_unseen[1000:1000 + 1000]])
            sc = torch.tensor(score(model, corpus, Xc))
            kth = sc.flatten().kthvalue(n_contam).values  # n_contam-th smallest NLL
            recall = float((sc[:n_contam] <= kth).float().mean())  # low NLL = familiar
            det[str(p)] = round(recall, 3)
            print(f"[{arm}] contamination p={p:.2f} recall@top-p: {recall:.3f}", flush=True)
        results[arm] = {"train_nll": round(float(s_seen.mean()), 4),
                        "val_nll": round(float(s_unseen.mean()), 4),
                        "auc": round(a, 4), "contamination_recall": det}
        del model
        torch.cuda.empty_cache()
    with open(r"D:\user\flypoet\logs_v2\familiarity_nll_result.json", "w") as f:
        json.dump(results, f, indent=1)
    print("saved logs_v2/familiarity_nll_result.json", flush=True)


if __name__ == "__main__":
    main()
