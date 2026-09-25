"""Train/infer k matrix: for each trained-in k checkpoint (10/25/50/dense),
evaluate val loss at every inference k on the ladder. Fills the
k_train x k_infer matrix that separates "optimization state" from
"architecture regime".
"""
import json, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import train_v2 as T
from elastic_inference import val_loss

DEV = "cuda"
PAIRS = [("flynetS", 0.10, "_s10"),   # 10%-trained (from sweep)
         ("flynetS_k25", 0.25, ""),   # 25%-trained
         ("flynetS_k50", 0.50, ""),   # 50%-trained
         ("std", None, "")]           # dense-trained
LADDER = [0.10, 0.25, 0.50, 0.75, 1.00]


def main():
    corpus = T.Corpus()
    matrix = {}
    for tag, k_train, _ in PAIRS:
        fp = os.path.join(ROOT, "logs_v2", f"{tag}_model.pt")
        if not os.path.exists(fp):
            print(f"skip {tag} (no ckpt)", flush=True)
            continue
        row = {}
        for k_inf in LADDER:
            kwta = None if k_inf >= 1.0 else {"impl": "torch", "k_frac": k_inf}
            model = T.GPT(corpus.V, kwta_opts=kwta).to(DEV)
            sd = torch.load(fp, map_location=DEV, weights_only=True)
            # dense-trained checkpoint has identical key set; sparse ckpts too
            # (same shapes) — loading is safe across k settings
            try:
                model.load_state_dict(sd)
            except RuntimeError as e:
                print(f"  {tag}@k{k_inf}: load failed ({e})", flush=True)
                del model
                torch.cuda.empty_cache()
                continue
            v = val_loss(model, corpus)
            row[str(k_inf)] = round(v, 4)
            print(f"{tag} (train k={k_train}) @ infer k={k_inf}: {v:.4f}",
                  flush=True)
            del model
            torch.cuda.empty_cache()
        matrix[f"{tag}|k_train={k_train}"] = row
    with open(os.path.join(ROOT, "logs_v2", "elastic_matrix_result.json"), "w") as f:
        json.dump(matrix, f, indent=1)
    print("saved logs_v2/elastic_matrix_result.json", flush=True)


if __name__ == "__main__":
    main()
