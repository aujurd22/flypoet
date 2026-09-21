"""Shower re-verification with fixed val windows and multiple base checkpoints.

ChatGPT round-6 critique: the original shower result (targeted decay10 helps,
final==best 3.588) rested on one checkpoint and a 24-random-batch eval.
This rerun:
  - FIXED val windows (seeded generator, 40 batches x 16 x 256)
  - 4 base checkpoints: std-24k, std-48k, k25-48k, adaptive-24k
  - variants: baseline (no shower) / decay10 / noise30, 500 steps each
Verdict format: per (base, variant) final val and delta vs that base's
baseline, on identical windows.
Note: the 48k schedules show NO overfit tail (val improves to the end), so
the "shower undoes the tail" hypothesis predicts ~no gain there — a real
boundary test, not a replica.
"""
import json, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import train_v2 as T

DEV = "cuda"
STEPS = 500
BASES = ["std", "std_48k", "flynetS_k25_48k", "flynetS_adaptive"]


@torch.no_grad()
def fixed_val(model, corpus):
    model.eval()
    gen = torch.Generator().manual_seed(1234)
    ls = []
    for _ in range(40):
        ix = torch.randint(len(corpus.val) - 256 - 1, (16,), generator=gen)
        x = torch.stack([torch.from_numpy(corpus.val[i:i + 256]) for i in ix]).to(DEV)
        y = torch.stack([torch.from_numpy(corpus.val[i + 1:i + 1 + 256]) for i in ix]).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, l = model(x, y)
        ls.append(l.item())
    model.train()
    return float(np.mean(ls))


@torch.no_grad()
def shower_step(model, variant):
    for p in model.parameters():
        if p.ndim != 2:
            continue
        flat = p.data.abs().flatten()
        frac = 0.10 if variant == "decay10" else 0.30
        k = max(1, int(flat.numel() * frac))
        thr = flat.kthvalue(k).values
        low = p.data.abs() <= thr
        if variant == "decay10":
            p.data[low] *= (1 - 1e-3)
        else:  # noise30
            p.data[low] += 1e-4 * p.data.std() * torch.randn_like(p.data[low])


def main():
    corpus = T.Corpus()
    out = {}
    for base in BASES:
        fp = os.path.join(ROOT, "logs_v2", f"{base}_model.pt")
        if not os.path.exists(fp):
            print(f"skip {base} (no ckpt)"); continue
        entry = {}
        for variant in ["baseline", "decay10", "noise30"]:
            model = T.GPT(corpus.V).to(DEV)
            sd = torch.load(fp, map_location=DEV, weights_only=True)
            model.load_state_dict(sd)
            model.train()
            v0 = fixed_val(model, corpus)
            best = v0
            if variant != "baseline":
                for step in range(1, STEPS + 1):
                    shower_step(model, variant)
                    if step % 50 == 0:
                        best = min(best, fixed_val(model, corpus))
            v = fixed_val(model, corpus)
            entry[variant] = {"start": round(v0, 4), "best": round(best, 4),
                              "final": round(v, 4),
                              "delta_final_vs_base0": round(v - v0, 4)}
            print(f"[{base}/{variant}] start={v0:.4f} final={v:.4f} "
                  f"delta={v - v0:+.4f}", flush=True)
            del model
            torch.cuda.empty_cache()
        out[base] = entry
    with open(os.path.join(ROOT, "logs_v2", "shower_verify_result.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved logs_v2/shower_verify_result.json", flush=True)


if __name__ == "__main__":
    main()
