"""Experiment: the elastic dial — a model TRAINED at k=25% evaluated with a
different inference-time k.

Dense-to-Dynamic-k (NeurIPS 2024) dials k for CONVERTED networks; untested
for trained-in fixed-k transformers. Sweep inference k over the 9-point
ladder on the k25-24k checkpoint; report val loss per point.

If the val(k) curve is U-shaped with its own sweet spot near 25%, one model
is a multi-gear deployable; if quality collapses off 25%, the weights are
co-adapted to exactly one sparsity.
"""
import json, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import train_v2 as T

DEV = "cuda"
LADDER = [0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 1.00]


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


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "flynetS_k25_24k"
    corpus = T.Corpus()
    results = {}
    for k in LADDER:
        kwta = None if k >= 1.0 else {"impl": "torch", "k_frac": k}
        model = T.GPT(corpus.V, kwta_opts=kwta).to(DEV)
        sd = torch.load(os.path.join(ROOT, "logs_v2", f"{which}_model.pt"),
                        map_location=DEV, weights_only=True)
        model.load_state_dict(sd)
        v = val_loss(model, corpus)
        results[str(k)] = round(v, 4)
        print(f"inference k={k:g}: val={v:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()
    with open(os.path.join(ROOT, "logs_v2", "elastic_inference_result.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("saved logs_v2/elastic_inference_result.json", flush=True)


if __name__ == "__main__":
    main()
