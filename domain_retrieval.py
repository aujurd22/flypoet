"""Experiment: concept-level retrieval — does the sparse code address TOPIC?

sparse_retrieval.py tested next-char payload (weak semantics). Here the
payload is the 8-domain label: build an address book from labeled training
windows (k25-24k codes), query with held-out windows, ask whether the
Hamming-NN neighbours share the query's TOPIC (above the balanced-chance
12.5%), and compare Hamming vs dense cosine.
"""
import json, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import train_v2 as T

DEV = "cuda"
SEQ = 192
PER_LABEL = 500          # store per label
Q_PER_LABEL = 120        # query per label


@torch.no_grad()
def build(model, corpus, rows, idx):
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
    return torch.cat(codes), torch.cat(dense), np.array(labels)


def hitk(qm, sm, ql, sl, k):
    d = (qm.sum(1, keepdim=True) + sm.sum(1).unsqueeze(0)
         - 2.0 * (qm.float() @ sm.float().T))
    idx = d.argsort(dim=1)[:, :k].numpy()
    return float(np.mean((sl[idx] == ql[:, None]).any(axis=1)))


def hitk_cos(qv, sv, ql, sl, k):
    qn = qv / qv.norm(dim=1, keepdim=True).clamp(min=1e-6)
    sn = sv / sv.norm(dim=1, keepdim=True).clamp(min=1e-6)
    d = 1.0 - qn @ sn.T
    idx = d.argsort(dim=1)[:, :k].numpy()
    return float(np.mean((sl[idx] == ql[:, None]).any(axis=1)))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "flynetS_k25_24k"
    torch.manual_seed(7)
    corpus = T.Corpus()
    model = T.GPT(corpus.V, kwta_opts={"impl": "torch", "k_frac": 0.25}).to(DEV)
    sd = torch.load(os.path.join(ROOT, "logs_v2", f"{which}_model.pt"),
                    map_location=DEV, weights_only=True)
    model.load_state_dict(sd)
    model.eval()

    rows = []
    with open(os.path.join(ROOT, "decide_data", "train.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append((d["ids"], d["label"]))
    vrows = []
    with open(os.path.join(ROOT, "decide_data", "val.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            vrows.append((d["ids"], d["label"]))

    rng = np.random.default_rng(5)
    by_label = {}
    for i, (_, lab) in enumerate(rows):
        by_label.setdefault(lab, []).append(i)
    store_idx = []
    for lab, lst in sorted(by_label.items()):
        take = [i for i in rng.choice(lst, size=min(PER_LABEL, len(lst)), replace=False)]
        store_idx += [(i, lab) for i in take]
    by_label_v = {}
    for i, (_, lab) in enumerate(vrows):
        by_label_v.setdefault(lab, []).append(i)
    query_idx = []
    for lab, lst in sorted(by_label_v.items()):
        take = [i for i in rng.choice(lst, size=min(Q_PER_LABEL, len(lst)), replace=False)]
        query_idx += [(i, lab) for i in take]

    print(f"store={len(store_idx)} query={len(query_idx)}", flush=True)
    S_code, S_dense, S_lab = build(model, corpus, rows, store_idx)
    Q_code, Q_dense, Q_lab = build(model, corpus, vrows, query_idx)

    out = {"checkpoint": which, "n_store": len(store_idx), "n_query": len(query_idx)}
    for k in (1, 10, 50):
        h = hitk(Q_code, S_code, Q_lab, S_lab, k)
        c = hitk_cos(Q_dense, S_dense, Q_lab, S_lab, k)
        out[f"hamming_topic_hit{k}"] = round(h, 4)
        out[f"cosine_topic_hit{k}"] = round(c, 4)
        print(f"topic hit@{k}: hamming={h:.3f} cosine={c:.3f} (chance 0.125)", flush=True)
    with open(os.path.join(ROOT, "logs_v2", "domain_retrieval_result.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved logs_v2/domain_retrieval_result.json", flush=True)


if __name__ == "__main__":
    main()
