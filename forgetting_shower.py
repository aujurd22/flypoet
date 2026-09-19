"""Experiment 3: fly active-forgetting shower on the overfit tail.

Berry et al. 2018: fruit flies have dedicated forgetting neurons — active
erasure is a feature. Hook: all three 24k arms show a synchronized val
rise in the OneCycle tail (std 3.610@20k -> 3.673@24k). Test whether a
data-free "forgetting shower" targeted at low-magnitude weights undoes
that rise.

Variants (each starts fresh from std-24k, 500 shower steps, no data/grad):
  decay30 / decay10 : multiply weights below the 30th/10th |w| percentile
                      by (1 - rate) each step
  noise30           : add Gaussian noise (sigma = 1e-4 * tensor std) to
                      weights below the 30th percentile
Metric: val loss every 50 steps (same protocol as train_v2 val_loss).
"""
import json, sys
import numpy as np
import torch

sys.path.insert(0, r"D:\user\flypoet")
import train_v2 as T

DEV = "cuda"
STEPS = 500


@torch.no_grad()
def val_loss(model, corpus):
    model.eval()
    ls = []
    for _ in range(24):
        x, y = corpus.batches(corpus.val, 16, 256)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, l = model(x, y)
        ls.append(l.item())
    model.train()
    return float(np.mean(ls))


def shower(model, variant, rate=1e-3):
    """One shower step, applied in-place to 2D weights."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim != 2:
                continue
            flat = p.data.abs().flatten()
            k = max(1, int(flat.numel() * (0.30 if "30" in variant else 0.10)))
            thr = flat.kthvalue(k).values
            low = p.data.abs() <= thr
            if variant.startswith("decay"):
                p.data[low] *= (1 - rate)
            elif variant == "noise30":
                p.data[low] += 1e-4 * p.data.std() * torch.randn_like(p.data[low])


def main():
    corpus = T.Corpus()
    results = {}
    for variant in ["decay30", "decay10", "noise30"]:
        model = T.GPT(corpus.V).to(DEV)
        sd = torch.load(r"D:\user\flypoet\logs_v2\std_model.pt",
                        map_location=DEV, weights_only=True)
        model.load_state_dict(sd)
        model.train()
        v0 = val_loss(model, corpus)
        traj = [{"step": 0, "val": round(v0, 4)}]
        print(f"[{variant}] start val={v0:.4f}", flush=True)
        for step in range(1, STEPS + 1):
            shower(model, variant)
            if step % 50 == 0:
                v = val_loss(model, corpus)
                traj.append({"step": step, "val": round(v, 4)})
                print(f"[{variant}] step{step} val={v:.4f}", flush=True)
        best = min(t["val"] for t in traj)
        results[variant] = {"start": round(v0, 4), "best": round(best, 4),
                            "final": traj[-1]["val"], "traj": traj}
        del model
        torch.cuda.empty_cache()
    with open(r"D:\user\flypoet\logs_v2\forgetting_shower_result.json", "w") as f:
        json.dump(results, f, indent=1)
    print("saved logs_v2/forgetting_shower_result.json", flush=True)


if __name__ == "__main__":
    main()
