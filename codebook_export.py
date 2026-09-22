"""Export the sparse-code address book from a trained k-WTA checkpoint.

Produces logs_v2/codebook.npz:
  codes_packed uint8 (N, 96)   top-25% codes of the window's last position,
                               packed 768 bits -> 96 bytes
  dense16      float16 (N,768) dense vector for rerank (4x smaller than fp32)
  labels       int8 (N,)      8-domain label
  q_codes_packed / q_dense16 / q_labels  — held-out query side (val split)
"""
import json, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import train_v2 as T

DEV = "cuda"
SEQ = 192
PER_LABEL = 750
Q_PER_LABEL = 120


@torch.no_grad()
def build(model, rows, idx):
    codes, dense, labels = [], [], []
    for i in range(0, len(idx), 48):
        chunk = idx[i:i + 48]
        X = torch.zeros(len(chunk), SEQ, dtype=torch.long)
        for b, (ri, lab) in enumerate(chunk):
            ids = rows[ri][0][:SEQ]
            X[b, :len(ids)] = torch.tensor(ids)
            labels.append(lab)
        xb = X.to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(xb, collect_final=True)
        h = model._last_hidden[:, -1].float().cpu()
        thr = torch.kthvalue(h, int(h.shape[-1] * 0.75), dim=-1, keepdim=True).values
        codes.append(h >= thr)
        dense.append(h)
    return torch.cat(codes), torch.cat(dense), np.array(labels, dtype=np.int8)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "flynetS_k25_24k"
    torch.manual_seed(7)
    corpus = T.Corpus()
    model = T.GPT(corpus.V, kwta_opts={"impl": "torch", "k_frac": 0.25}).to(DEV)
    sd = torch.load(os.path.join(ROOT, "logs_v2", f"{which}_model.pt"),
                    map_location=DEV, weights_only=True)
    model.load_state_dict(sd)
    model.eval()

    def load(split):
        rows = []
        fp = os.path.join(ROOT, "decide_data", f"{split}.jsonl")
        with open(fp, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                rows.append((d["ids"], d["label"]))
        return rows

    rng = np.random.default_rng(5)
    out = {}
    for split, per, tag in [("train", PER_LABEL, ""), ("val", Q_PER_LABEL, "q_")]:
        rows = load(split)
        by_label = {}
        for i, (_, lab) in enumerate(rows):
            by_label.setdefault(lab, []).append(i)
        idx = []
        for lab in sorted(by_label):
            lst = by_label[lab]
            take = rng.choice(lst, size=min(per, len(lst)), replace=False)
            idx += [(int(i), lab) for i in take]
        C, D, L = build(model, rows, idx)
        packed = np.packbits(C.numpy().astype(np.uint8), axis=1)
        out[f"{tag}codes_packed"] = packed
        out[f"{tag}dense16"] = D.numpy().astype(np.float16)
        out[f"{tag}labels"] = L
        print(f"{split}: {packed.shape[0]} codes, {packed.shape[1]} bytes each",
              flush=True)

    np.savez_compressed(os.path.join(ROOT, "logs_v2", "codebook.npz"), **out)
    mb = os.path.getsize(os.path.join(ROOT, "logs_v2", "codebook.npz")) / 1e6
    print(f"saved logs_v2/codebook.npz ({mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
