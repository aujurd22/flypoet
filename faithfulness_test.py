"""Faithfulness test: does trained-in stable binding produce more grounded output?

Simple test: sample 200 continuations from the same 200 seed contexts using
k25-24k and dense-24k (same temperature). Score each generation:
  1. NLL under the model itself (self-confidence in own output)
  2. distinct-3 (diversity — rules out mode collapse)
  3. NLL under the OTHER model (cross-model agreement)

Prediction (stable binding): k25 generations should have lower self-NLL
and/or lower cross-model NLL than dense, without diversity collapse.
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import train_v2 as T

DEV = "cuda"
SEQ = 64
GEN = 128
N = 200
TEMP = 0.7


@torch.no_grad()
def sample(model, corpus, starts, n_new=GEN, temp=TEMP):
    model.eval()
    gens = []
    for i in range(0, len(starts), 50):
        chunk = starts[i:i + 50]
        xb = torch.stack([torch.from_numpy(np.copy(corpus.val[p:p + SEQ]))
                          for p in chunk]).to(DEV)
        gen = xb.clone()
        for _ in range(n_new):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(gen)
            lg = logits[:, -1, :].float() / temp
            nxt = torch.multinomial(F.softmax(lg, dim=-1), 1)
            gen = torch.cat([gen, nxt], dim=1)
        gens.append(gen[:, -n_new:].cpu())
    return torch.cat(gens, dim=0)


@torch.no_grad()
def batch_nll(model, corpus, windows):
    """Per-window NLL."""
    out = []
    for i in range(0, len(windows), 50):
        xb = torch.stack([torch.from_numpy(np.copy(w)) for w in windows[i:i + 50]]).to(DEV)
        yb = xb.clone()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, l = model(xb[:, :-1], yb[:, 1:])
        out.extend([l.item()] * xb.shape[0])
    return np.array(out)


@torch.no_grad()
def batch_nll(model, corpus, windows):
    """Mean NLL for each window."""
    out = []
    for i in range(0, len(windows), 50):
        xb = torch.stack([torch.from_numpy(np.copy(w)) for w in windows[i:i+50]]).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, l = model(xb[:, :-1].contiguous(), xb[:, 1:].contiguous())
        out.extend([l.item()] * xb.shape[0])
    return np.array(out)


def distinct_n(gens, n=3):
    """distinct-n over token sequences."""
    grams = set()
    total = 0
    for g in gens:
        toks = g.numpy().tolist()
        for i in range(len(toks) - n + 1):
            grams.add(tuple(toks[i:i + n]))
            total += 1
    return len(grams) / max(total, 1)


def main():
    corpus = T.Corpus()
    torch.manual_seed(42)
    starts = torch.randint(len(corpus.val) - SEQ - GEN - 1, (N,)).numpy()

    models = {}
    for label, tag, kfrac in [("k25", "flynetS_k25_24k", 0.25),
                               ("dense", "std", None)]:
        kw = {"impl": "torch", "k_frac": kfrac} if kfrac else None
        m = T.GPT(corpus.V, kwta_opts=kw).to(DEV)
        fp = os.path.join(ROOT, "logs_v2", f"{tag}_model.pt")
        m.load_state_dict(torch.load(fp, map_location=DEV, weights_only=True))
        models[label] = m

    # generate from both models with the same seeds
    torch.manual_seed(999)
    gens = {}
    for label, model in models.items():
        model.eval()
        gens[label] = sample(model, corpus, starts)

    # score
    results = {}
    ref_model = models["dense"]  # reference for NLL

    for label in ["k25", "dense"]:
        g = gens[label]
        # 1. NLL under reference model (dense)
        nll = batch_nll(ref_model, corpus, g.numpy())
        # 2. distinct-3
        d3 = distinct_n([g[i] for i in range(len(g))])
        # 3. mean token entropy of generated distribution
        results[f"{label}_ref_nll"] = round(float(np.mean(nll)), 4)
        results[f"{label}_distinct3"] = round(d3, 4)

    # self-consistency: NLL of k25-gen under k25-model vs dense-gen under dense-model
    for label, model in [("k25", models["k25"]), ("dense", models["dense"])]:
        g = gens[label]
        nlls = []
        for i in range(0, len(g), 50):
            xb = g[i:i + 50].to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, l = model(xb[:, :-1].contiguous(), xb[:, 1:].contiguous())
            nlls.append(l.item())
        results[f"{label}_self_nll"] = round(float(np.mean(nlls)), 4)

    for k, v in sorted(results.items()):
        print(f"  {k}: {v}", flush=True)

    with open(os.path.join(ROOT, "logs_v2", "faithfulness_result.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("saved logs_v2/faithfulness_result.json", flush=True)


if __name__ == "__main__":
    main()
