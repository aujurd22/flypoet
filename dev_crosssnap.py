"""Developmental cross-snapshot code stability (committed version of the
analysis behind logs_v2/dev_crosssnap_result.json).

For consecutive k-WTA training snapshots: Jaccard between the top-25% codes
of the SAME 300 val windows at t-1 and t. Rising values = the address system
freezes (codes stop changing). Chance level: different windows, same
snapshot.
"""
import json, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import train_v2 as T
from code_address import codes_for

DEV = "cuda"


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "flynetS_snap"
    n_win = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    corpus = T.Corpus()
    model = T.GPT(corpus.V, kwta_opts={"impl": "torch", "k_frac": 0.25}).to(DEV)

    import re
    snaps = sorted(int(re.search(r"snap(\d+)\.pt$", f).group(1))
                   for f in os.listdir(os.path.join(ROOT, "logs_v2"))
                   if f.startswith(f"{tag}_snap") and f.endswith(".pt"))
    if not snaps:
        print("no snapshots found"); return
    print("snapshots:", snaps, flush=True)

    torch.manual_seed(11)
    ix = torch.randint(len(corpus.val) - 64 - 1, (n_win,))
    res = {}
    prev = None
    for st in snaps:
        sd = torch.load(os.path.join(ROOT, "logs_v2", f"{tag}_snap{st}.pt"),
                        map_location=DEV, weights_only=True)
        model.load_state_dict(sd)
        model.eval()
        C, _ = codes_for(model, corpus, ix.numpy())
        if prev is not None:
            inter = (C & prev).sum(1).float()
            union = (C | prev).sum(1).float()
            j = float((inter / union.clamp(min=1)).mean())
            res[st] = round(j, 4)
            print(f"cross-snapshot Jaccard (prev-> {st}): {j:.4f}", flush=True)
        prev = C
    ix2 = torch.randint(len(corpus.val) - 64 - 1, (n_win,))
    C2, _ = codes_for(model, corpus, ix2.numpy())
    inter = (C & C2).sum(1).float()
    union = (C | C2).sum(1).float()
    chance = float((inter / union.clamp(min=1)).mean())
    print(f"chance (different windows, same snapshot): {chance:.4f}", flush=True)
    res["chance_same_snapshot"] = round(chance, 4)
    with open(os.path.join(ROOT, "logs_v2", "dev_crosssnap_result.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("saved logs_v2/dev_crosssnap_result.json", flush=True)


if __name__ == "__main__":
    main()
