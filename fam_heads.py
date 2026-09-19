"""Experiment 2: fly familiarity circuit as a contamination / memorization probe.

Fly story: mushroom-body ON/OFF novelty detectors answer "have I seen this?"
without recalling content. ML analogue: a tiny logistic head on frozen trunk
hidden states that scores how familiar a window is to the trunk.

Uses:
 1. Memorization signature: can we separate windows the trunk TRAINED on
    (train.bin) from held-out windows (val.bin)? AUC near 0.5 = no signature;
    high AUC = the trunk's hidden states leak "I've seen this".
 2. Contamination detector demo: mix p of train windows into a val eval set;
    report how many land in the top-p% familiarity scores.

Run for both std and flynetS trunks (24k checkpoints).
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\user\flypoet")
import train_v2 as T

DEV = "cuda"
SEQ = 192
N_PER = 4000


def windows_from_stream(arr, n, seed):
    torch.manual_seed(seed)
    ix = torch.randint(len(arr) - SEQ - 1, (n,))
    return torch.stack([torch.from_numpy(arr[i:i + SEQ]) for i in ix])


@torch.no_grad()
def extract(model, corpus, X):
    hs = []
    for i in range(0, len(X), 64):
        xb = X[i:i + 64].to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(xb, collect_final=True)
        hs.append(model._last_hidden[:, -1].float().cpu())
    return torch.cat(hs)


def auc(pos, neg):
    """Mann-Whitney AUC: P(score_pos > score_neg) + 0.5*ties."""
    s = torch.cat([pos, neg])
    order = s.argsort().argsort().float()          # ranks, 0-based
    r_pos = order[:len(pos)].sum()
    n_p, n_n = len(pos), len(neg)
    return float((r_pos - n_p * (n_p - 1) / 2) / (n_p * n_n))


def train_head(Htr, ytr, d=768, steps=600):
    torch.manual_seed(0)
    head = torch.nn.Linear(d, 1).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-3)
    X, y = Htr.to(DEV), ytr.to(DEV)
    for _ in range(steps):
        idx = torch.randint(0, len(X), (512,))
        out = head(X[idx]).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(out, y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return head


def main():
    corpus = T.Corpus()
    results = {}
    for arm in ["std", "flynetS"]:
        model = T.GPT(corpus.V, kwta_opts={"impl": "torch"} if arm == "flynetS" else None).to(DEV)
        sd = torch.load(rf"D:\user\flypoet\logs_v2\{arm}_model.pt",
                        map_location=DEV, weights_only=True)
        model.load_state_dict(sd)
        model.eval()

        X_seen = windows_from_stream(corpus.train, N_PER, seed=11)
        X_unseen = windows_from_stream(corpus.val, N_PER, seed=12)
        H_seen, H_unseen = extract(model, corpus, X_seen), extract(model, corpus, X_unseen)

        # head: 3000/class train, 1000/class test
        y = torch.cat([torch.ones(3000), torch.zeros(3000)])
        Htr = torch.cat([H_seen[:3000], H_unseen[:3000]])
        head = train_head(Htr, y)
        with torch.no_grad():
            s_seen = head(H_seen[3000:].to(DEV)).squeeze(-1).cpu()
            s_unseen = head(H_unseen[3000:].to(DEV)).squeeze(-1).cpu()
        a = auc(s_seen, s_unseen)
        print(f"[{arm}] familiarity AUC (seen vs unseen): {a:.4f}", flush=True)

        # contamination demo: 1000 val windows + p*1000 train windows mixed in
        det = {}
        for p in [0.05, 0.10, 0.20, 0.50]:
            torch.manual_seed(21)
            n_contam = int(p * 1000)
            mix_seen = X_seen[3000:3000 + n_contam]
            mix_val = X_unseen[3000:3000 + 1000]
            Hm = extract(model, corpus, torch.cat([mix_seen, mix_val]))
            with torch.no_grad():
                scores = head(Hm.to(DEV)).squeeze(-1).cpu()
            kth = scores.argsort(descending=True)[max(0, n_contam - 1)]
            recall = float((scores[:n_contam] >= kth).float().mean())
            det[str(p)] = round(recall, 3)
            print(f"[{arm}] contamination p={p:.2f} recall@top-p: {recall:.3f}", flush=True)

        results[arm] = {"auc": round(a, 4), "contamination_recall": det}
        del model
        torch.cuda.empty_cache()

    with open(r"D:\user\flypoet\logs_v2\familiarity_result.json", "w") as f:
        json.dump(results, f, indent=1)
    print("saved logs_v2/familiarity_result.json", flush=True)


if __name__ == "__main__":
    main()
