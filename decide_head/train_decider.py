"""Train the fly decision head on a frozen FlyNet-Dopa trunk.

Head: input = trunk final hidden state (after k-WTA gate), output =
8-domain distribution + 1 abstain logit (9-way). Abstain predictions are
excluded from accuracy but counted separately (Jev-critique fix).
Frozen trunk => only the head trains (~6M params), fast.
"""
import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r"D:\user\flypoet")
import train_v2 as T

torch.manual_seed(7)
DEV = "cuda"
DATA = r"D:\user\flypoet\decide_data"
SEQ = 200


class FlyDecider(nn.Module):
    def __init__(self, trunk, hidden=768, n_classes=8):
        super().__init__()
        self.trunk = trunk
        for p in self.trunk.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(hidden, 256), nn.GELU(),
            nn.Linear(256, n_classes + 1))      # +1 = abstain

    def forward(self, idx):
        with torch.no_grad():
            self.trunk(idx, collect_final=True)
            h = self.trunk._last_hidden[:, -1]   # last position hidden
        return self.head(h.float())


def load_rows(split):
    rows = []
    with open(os.path.join(DATA, f"{split}.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append((d["ids"], d["label"]))
    return rows


def batches(rows, batch, stoi_pad=0, seq=SEQ):
    idx = torch.randint(0, len(rows), (batch,))
    x = torch.zeros(batch, seq, dtype=torch.long)
    y = torch.zeros(batch, dtype=torch.long)
    for b, i in enumerate(idx.tolist()):
        ids = rows[i][0][:seq]
        x[b, :len(ids)] = torch.tensor(ids)
        y[b] = rows[i][1]
    return x.to(DEV), y.to(DEV)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunk", default="flynetS_adaptive")
    a = ap.parse_args()
    # load trunk with its trained compute graph (flynetS_adaptive: CUDA k-WTA)
    corpus = T.Corpus()
    kwta = {"impl": "cuda"} if a.trunk == "flynetS_adaptive" else None
    trunk = T.GPT(corpus.V, kwta_opts=kwta).to(DEV)
    sd = torch.load(rf"D:\user\flypoet\logs_v2\{a.trunk}_model.pt",
                    map_location=DEV, weights_only=True)
    trunk.load_state_dict(sd)
    trunk.eval()
    print(f"[trunk: {a.trunk}]", flush=True)

    model = FlyDecider(trunk).to(DEV)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"decision head trainable params: {n_train/1e6 if False else n_trainable/1e6:.2f}M", flush=True)

    train_rows = load_rows("train")
    val_rows = load_rows("val")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=3e-4, weight_decay=0.01)
    labels = json.load(open(os.path.join(DATA, "labels.json")))["domains"]

    t0 = time.time()
    step = 0
    best_acc = 0.0
    while step < 10000:
        x, y = batches(train_rows, 32)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        step += 1
        if step % 200 == 0:
            print(f"step {step}: loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
        if step % 1000 == 0 or step == 10000:
            model.eval()
            correct = abstain = total = 0
            confs, corr = [], []
            with torch.no_grad():
                for _ in range(140):
                    xb, yb = batches(val_rows, 32)
                    logits = model(xb)
                    probs = F.softmax(logits.float(), dim=-1)
                    conf, pred = probs.max(dim=-1)
                    for p_, c_, pr in zip(pred.tolist(), yb.tolist(), conf.tolist()):
                        total += 1
                        if p_ == 8:            # abstain
                            abstain += 1
                        elif p_ == c_:
                            correct += 1
                            confs.append(pr); corr.append(1)
                        else:
                            confs.append(pr); corr.append(0)
            acc = correct / max(total - abstain, 1)
            ab_rate = abstain / total
            print(f"    val acc(非弃权)={acc:.3f} | 弃权率={ab_rate:.1%}", flush=True)
            if acc > best_acc and step >= 1000:
                best_acc = acc
                torch.save(model.state_dict(), os.path.join(DATA, "best_decider.pt"))
            model.train()

    # calibration quick-check on non-abstain predictions
    model.eval()
    confs, corr = [], []
    with torch.no_grad():
        for _ in range(140):
            xb, yb = batches(val_rows, 32)
            probs = F.softmax(model(xb).float(), dim=-1)
            conf, pred = probs.max(dim=-1)
            for p_, c_, pr in zip(pred.tolist(), yb.tolist(), conf.tolist()):
                if p_ != 8:
                    confs.append(pr); corr.append(1.0 if p_ == c_ else 0.0)
    confs = np.array(confs); corr = np.array(corr)
    ece, edges = 0.0, np.linspace(0, 1, 11)
    for b in range(10):
        m = (confs > edges[b]) & (confs <= edges[b+1])
        if m.sum():
            ece += m.mean() * abs(corr[m].mean() - confs[m].mean())
    print(f"FINAL: acc={corr.mean():.3f} | ECE={ece:.4f} | "
          f"覆盖={len(confs)/ (len(confs)+ (total-abstain) * 0 + 0):.0%} of non-abstain", flush=True)
    json.dump({"acc_non_abstain": float(corr.mean()), "ECE": float(ece),
               "best_acc": best_acc}, open(os.path.join(DATA, f"decider_result_{a.trunk}.json"), "w"))

if __name__ == "__main__":
    main()
