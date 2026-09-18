"""Calibration evaluation for trained arms: does the fly mechanism produce
calibrated confidence?

Method: on held-out text, measure the model's confidence in its top-1 next-char
prediction vs the actual frequency that prediction was correct.
Metrics: ECE (15-bin), MCE, Brier (proper scoring, binning-free).

Usage: python calibration_eval.py --arm std --model logs_v2/std_model.pt
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F

import train_v2 as T

def build_bigram_model(train):
    """Char-bigram log-prob table from the training stream (Laplace smoothed)."""
    import collections
    big = collections.Counter()
    uni = collections.Counter()
    for i in range(0, len(train) - 1, 7):          # strided sample for speed
        big[(train[i], train[i + 1])] += 1
        uni[train[i]] += 1
    V = len(uni) + 2
    return big, uni, V


def familiarity(arr, train, start, seq=256, big=None, uni=None, V=10002):
    """Familiarity proxy: mean bigram log-prob of the window (smoothed)."""
    import math
    lps = []
    for i in range(start, min(start + seq, len(arr) - 1)):
        a, b = int(arr[i]), int(arr[i + 1])
        cnt = big.get((a, b), 0) + 1
        tot = uni.get(a, 0) + V
        lps.append(math.log(cnt / tot))
    return float(np.mean(lps)) if lps else -20.0


@torch.no_grad()
def evaluate(arm, model_path, n_windows=400, bins=15, seed=0):
    corpus = T.Corpus()
    # 关键: 评估必须复现该臂训练时的 forward（flynetS 权重依赖 k-WTA 计算图）
    kwta = {"impl": "torch"} if arm == "flynetS" else ({"impl": "cuda"} if "adaptive" in arm else None)
    model = T.GPT(corpus.V, kwta_opts=kwta).to(T.DEV)
    # 注: cuda 惰性编译需 MSVC 环境; 无环境时该臂评估会失败(见 except 分支)
    sd = torch.load(model_path, map_location=T.DEV, weights_only=True)
    model.load_state_dict(sd)
    model.eval()

    confs, corrects, fams = [], [], []
    arr = corpus.val
    big, uni, _ = build_bigram_model(corpus.train)
    rng = np.random.default_rng(seed)
    starts = rng.choice(len(arr) - 257, size=n_windows, replace=False)
    for s in starts:
        fams.append(familiarity(arr, corpus.train, int(s), big=big, uni=uni))
        x = torch.tensor(arr[s:s + 256], device=T.DEV).unsqueeze(0)
        y_next = arr[s + 256]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
        probs = F.softmax(logits[0, -1].float(), dim=-1)
        top_p, top_i = probs.max(dim=-1)
        confs.append(float(top_p))
        corrects.append(1.0 if int(top_i) == y_next else 0.0)

    confs = np.array(confs); corrects = np.array(corrects)
    fams = np.array(fams)
    acc = corrects.mean()
    avg_conf = confs.mean()
    ece, mce = 0.0, 0.0
    edges = np.linspace(0, 1, bins + 1)
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        m = (confs > lo) & (confs <= hi)
        if m.sum() > 0:
            e = abs(corrects[m].mean() - confs[m].mean())
            ece += (m.sum() / len(confs)) * e
            mce = max(mce, e)
    brier = float(np.mean((confs - corrects) ** 2))
    # familiarity-stratified calibration (ternary split)
    q = np.quantile(fams, [1/3, 2/3])
    fam_strata = []
    for lo, hi in ((0, q[0]), (q[0], q[1]), (q[1], 1.01)):
        m = (fams >= lo) & (fams < hi)
        if m.sum() > 5:
            fam_strata.append({"fam_range": [float(lo), float(hi)],
                               "ece": float(np.abs(corrects[m] - confs[m]).mean()),
                               "acc": float(corrects[m].mean()),
                               "n": int(m.sum())})
    return {"arm": arm, "n_windows": n_windows, "accuracy": float(acc),
            "avg_confidence": float(avg_conf), "confidence_gap": float(avg_conf - acc),
            "ECE": float(ece), "MCE": float(mce), "Brier": brier,
            "familiarity_strata": fam_strata}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_windows", type=int, default=400)
    a = ap.parse_args()
    r = evaluate(a.arm, a.model, seed=a.seed, n_windows=a.n_windows)
    out = rf"D:\user\flypoet\logs_v2\calibration_{a.arm}_seed{a.seed}.json"
    json.dump(r, open(out, "w"), indent=1)
    print(json.dumps(r, indent=1))
