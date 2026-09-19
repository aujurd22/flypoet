"""Temperature scaling (Guo et al. 2017) for the fly decision head.

Fits a single scalar T on a held-out split by minimizing NLL of the
temperature-scaled softmax. Standard post-hoc calibration; cost ~0.

Also evaluates ECE before/after so the fix is measurable.
"""
import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r"D:\user\flypoet")
import train_v2 as T

DEV = "cuda"
DATA = r"D:\user\flypoet\decide_data"
SEQ = 200


def load_rows(split):
    rows = []
    with open(os.path.join(DATA, f"{split}.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append((d["ids"], d["label"]))
    return rows


def batches(rows, batch=32):
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        x = torch.zeros(len(chunk), SEQ, dtype=torch.long)
        y = torch.zeros(len(chunk), dtype=torch.long)
        for b, (ids, lab) in enumerate(chunk):
            ids = ids[:SEQ]
            x[b, :len(ids)] = torch.tensor(ids)
            y[b] = lab
        yield x.to(DEV), y.to(DEV)


def collect_logits(model, rows):
    model.eval()
    logits_all, y_all = [], []
    for x, y in batches(rows):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        logits_all.append(logits.float().cpu())
        y_all.append(y.cpu())
    return torch.cat(logits_all), torch.cat(y_all)


def ece_eval(confs, corrects, bins=15):
    ece, edges = 0.0, np.linspace(0, 1, bins + 1)
    for b in range(bins):
        m = (confs > edges[b]) & (confs <= edges[b + 1])
        if m.sum():
            ece += m.mean() * abs(corrects[m].mean() - confs[m].mean())
    return float(ece)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunk", default="flynetS_adaptive")
    ap.add_argument("--ckpt", default="")
    a = ap.parse_args()

    # trunk + head
    sys.path.insert(0, r"D:\user\flypoet\decide_head")
    from train_decider import FlyDecider, load_rows
    corpus = T.Corpus()
    kwta = {"impl": "cuda"} if a.trunk == "flynetS_adaptive" else None
    trunk = T.GPT(corpus.V, kwta_opts=kwta).to(T.DEV)
    sd = torch.load(rf"D:\user\flypoet\logs_v2\{a.trunk}_model.pt",
                    map_location=T.DEV, weights_only=True)
    trunk.load_state_dict(sd)
    trunk.eval()
    head_ckpt = a.ckpt or os.path.join(DATA, f"best_decider_{a.trunk}.pt")
    if not os.path.exists(head_ckpt):
        head_ckpt = os.path.join(DATA, "best_decider.pt")  # legacy single-ckpt name
        print(f"[warn] per-trunk ckpt missing, falling back to {head_ckpt}")
    head_sd = torch.load(head_ckpt, map_location=T.DEV, weights_only=True)
    model = FlyDecider(trunk).to(T.DEV)
    model.load_state_dict(head_sd)
    print(f"[trunk={a.trunk}, head={os.path.basename(head_ckpt)}]", flush=True)

    # split val into calib(half) / test(half): T fitted on calib, evaluated on test
    rows = load_rows("val")
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(rows))
    calib_rows = [rows[i] for i in idx[: len(rows) // 2]]
    test_rows = [rows[i] for i in idx[len(rows) // 2:]]

    print("collecting logits...", flush=True)
    logits_c, y_c = collect_logits(model, calib_rows)
    logits_t, y_t = collect_logits(model, test_rows)

    # ---- before: raw softmax
    def metrics(logits, y, temperature=1.0):
        # 9th column is the abstain logit; score top-1 over the 8 real classes
        probs = F.softmax(logits[:, :8] / temperature, dim=-1)
        conf, pred = probs.max(dim=-1)
        correct = (pred == y).float()
        acc = float(correct.mean())
        ece = ece_eval(conf.detach().numpy(), correct.detach().numpy())
        brier = float(((conf - correct) ** 2).mean())
        return acc, ece, brier

    acc0, ece0, brier0 = metrics(logits_t, y_t, 1.0)
    print(f"BEFORE: acc={acc0:.3f} ECE={ece0:.4f} Brier={brier0:.4f}", flush=True)

    # ---- fit T on calib (minimize NLL)
    logT = torch.zeros(1, device=T.DEV, requires_grad=True)
    optT = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)
    logits_c_dev, y_c_dev = logits_c.to(T.DEV), y_c.to(T.DEV)
    def closure():
        optT.zero_grad()
        loss = F.cross_entropy(logits_c_dev / logT.exp(), y_c_dev)
        loss.backward()
        return loss
    optT.step(closure)
    T_star = float(logT.exp())
    print(f"fitted T = {T_star:.3f}", flush=True)

    acc1, ece1, brier1 = metrics(logits_t, y_t, T_star)
    print(f"AFTER:  acc={acc1:.3f} ECE={ece1:.4f} Brier={brier1:.4f}", flush=True)

    out = {"trunk": a.trunk, "T": T_star,
           "before": {"acc": acc0, "ECE": ece0, "Brier": brier0},
           "after": {"acc": acc1, "ECE": ece1, "Brier": brier1}}
    outp = rf"D:\user\flypoet\decide_data\temperature_{a.trunk}.json"
    json.dump(out, open(outp, "w"), indent=1)
    print(f"saved {outp}", flush=True)


if __name__ == "__main__":
    main()
