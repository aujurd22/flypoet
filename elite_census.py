"""Experiment: elite channel census on trained-in k-WTA models.

For each k-WTA checkpoint: how often does each channel win (be in the top-k
active set) across tokens/layers/windows? Questions:
  1. Is the win-rate distribution concentrated (elite hubs) or flat (rotation)?
  2. Do different seeds pick the same elite set (Gini overlap)?
  3. Do the 3 scales (92M/216M/336M) differ in concentration — explaining
     the 216M inversion?
Output: per-layer win-rate arrays + Gini coefficients + top-channel overlap
between seeds, saved to logs_v2/elite_census_result.json
"""
import json, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import train_v2 as T

DEV = "cuda"
SEQ = 256
N_WIN = 200          # 200 windows x 256 tokens = 51k positions per layer


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def census(model, corpus, d, k):
    """Mean win-rate per channel per layer, over N_WIN val windows."""
    counts = np.zeros((model.layers_count, d), dtype=np.int64)
    total = 0
    with torch.no_grad():
        for wi in range(N_WIN):
            p = int(torch.randint(len(corpus.val) - SEQ - 1, (1,)))
            x = torch.from_numpy(corpus.val[p:p + SEQ]).unsqueeze(0).to(DEV)
            h = corpus_emb(model, x)
            for li, block_h in enumerate(h):        # block_h: (T, d)
                top = torch.topk(block_h, k, dim=-1).indices.reshape(-1)
                cnt = torch.bincount(top, minlength=d).cpu().numpy()
                counts[li] += cnt
                total += block_h.shape[0]
    return counts / total


def corpus_emb(model, x):
    """Run the trunk, capturing the k-WTA input (pre-mask attention output)
    per block: (T, d) per layer."""
    stores = [[] for _ in model.blocks]
    handles = []
    for i, b in enumerate(model.blocks):
        def mk2(i):
            def hook(mod, inp, out):
                stores[i].append(inp[0].detach().squeeze(0).float().cpu())
            return hook
        handles.append(b.kwta.register_forward_hook(mk2(i)))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(x, collect_final=False)
    for h in handles:
        h.remove()
    return [torch.cat(s) for s in stores]


def main():
    tags = sys.argv[1:] or ["flynetS_k25_s8", "flynetS_k25_s9", "flynetS_L2k25"]
    corpus = T.Corpus()
    results = {}
    models = {}
    for tag in tags:
        fin = os.path.join(ROOT, "logs_v2", f"{tag}_final.json")
        m = json.load(open(fin))
        d, layers = int(m["d"]), int(m["layers"])
        heads, ffn_h = int(m.get("heads", 12)), int(m.get("ffn_h", 2048))
        k = max(1, int(d * m["k_frac"]))
        kwta = None
        if m.get("impl") == "torch":
            kwta = {"impl": "torch", "k_frac": m["k_frac"]}
        elif m.get("impl") == "cuda":
            kwta = {"impl": "cuda", "k_frac": m["k_frac"]}
        model = T.GPT(corpus.V, d=d, layers=layers, heads=heads,
                      ffn_h=ffn_h, kwta_opts=kwta).to(DEV)
        sd = torch.load(os.path.join(ROOT, "logs_v2", f"{tag}_model.pt"),
                        map_location=DEV, weights_only=True)
        model.load_state_dict(sd)
        model.eval()
        model.layers_count = layers
        print(f"census {tag} (d={d} k={k})...", flush=True)
        counts = census(model, corpus, d, k)
        ginis = [gini(c) for c in counts]
        # top-50 channels by win-rate, per layer -> set for cross-seed overlap
        top50 = [set(np.argsort(c)[::-1][:50].tolist()) for c in counts]
        np.save(os.path.join(ROOT, "logs_v2", f"elite_{tag}_winrate.npy"), counts)
        results[tag] = {"d": d, "k": k, "gini_per_layer": [round(g, 4) for g in ginis],
                        "gini_mean": round(float(np.mean(ginis)), 4),
                        "top50": {str(i): sorted(t) for i, t in enumerate(top50)}}
        models[tag] = top50
        del model
        torch.cuda.empty_cache()

    # cross-seed elite overlap (same-scale pairs)
    def overlap(a, b):
        return round(float(np.mean([len(models[a][i] & models[b][i]) / 50
                                    for i in range(len(models[a]))])), 4)
    for a, b in [("flynetS_k25_s8", "flynetS_k25_s9")]:
        if a in models and b in models:
            out[f"elite_overlap_{a}_vs_{b}"] = overlap(a, b)
    with open(os.path.join(ROOT, "logs_v2", "elite_census_result.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("saved logs_v2/elite_census_result.json", flush=True)
    for tag, r in results.items():
        print(f"{tag}: gini_mean={r['gini_mean']}", flush=True)


if __name__ == "__main__":
    main()
