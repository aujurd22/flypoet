"""Clean calibration runner - no inline code, no unpacking bugs."""
import sys, os, json
sys.path.insert(0, r"D:\user\flypoet")
sys.path.insert(0, r"D:\user\flypoet\decide_head")

import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda"
DATA = r"D:\user\flypoet\decide_data"
TRUNK_PT = r"D:\user\flypoet\logs_v2\flynetS_adaptive_model.pt"
HEAD_PT = os.path.join(DATA, "best_decider.pt")
SEQ = 200

import train_v2 as T
import train_v2

corpus = T.Corpus()

# --- load trunk
trunk = T.GPT(corpus.V, kwta_opts={"impl": "cuda"}).to(DEV)
sd = torch.load(TRUNK_PT, map_location=DEV, weights_only=True)
trunk.load_state_dict(sd)
trunk.eval()

# --- load head
head_sd = torch.load(HEAD_PT, map_location=DEV, weights_only=True)
# Reconstruct head as simple linear (it was saved as Sequential)
head = torch.nn.Linear(head_sd["0.weight"].shape[1], head_sd["0.weight"].shape[0]).to(DEV)
head.weight.data = head_sd["0.weight"]
head.bias.data = head_sd["0.bias"]
# Actually Sequential has keys like "0.weight" not "weight"
# Let's just build it to match
keys = list(head_sd.keys())
print("head keys:", keys[:5], flush=True)

# --- simpler: just load the full FlyDecider model
from decide_head.train_decider import FlyDecider
model = FlyDecider(trunk).to(DEV)
model.load_state_dict(torch.load(HEAD_PT, map_location=DEV, weights_only=True))
model.eval()

# --- load val data
val_rows = []
with open(os.path.join(DATA, "..", "decide_data", "val.jsonl"), encoding="utf-8") as f:
    for line in f:
        val_rows.append(json.loads(line))

# --- compute logits + confidence + correctness in one pass
@torch.no_grad()
def eval_calibration(model, rows):
    """Returns (confidences, correctness) for non-abstain predictions."""
    confs, corr = [], []
    bs = 32
    for i in range(0, len(rows), bs):
        chunk = rows[i:i+bs]
        max_len = max(len(r["ids"]) for r in chunk)
        x = torch.zeros(len(chunk), SEQ, dtype=torch.long, device=DEV)
        y = torch.zeros(len(chunk), dtype=torch.long, device=DEV)
        for b, r in enumerate(chunk):
            ids = r["ids"][:SEQ]
            x[b, :len(ids)] = torch.tensor(ids)
            y[b] = r["label"]
        logits = model(x)  # FlyDecider returns logits directly
        probs = F.softmax(logits.float(), dim=-1)
        conf, pred = probs.max(dim=-1)
        for p_, c_, pr in zip(pred.cpu().tolist(), y.cpu().tolist(), conf.cpu().tolist()):
            confs.append(pr)
            corr.append(1.0 if p_ == c_ else 0.0)
    return np.array(confs), np.array(corr)

confs, corr = eval_calibration(model, val_rows)
ece, edges = 0.0, np.linspace(0, 1, 16)
for b in range(15):
    m = (confs > edges[b]) & (confs <= edges[b+1])
    if m.sum():
        ece += m.mean() * abs(corr[m].mean() - confs[m].mean())

result = {"n": len(confs), "acc": float(corr.mean()),
          "avg_conf": float(confs.mean()),
          "gap": float(confs.mean() - corr.mean()),
          "ECE": float(ece)}
print(json.dumps(result, indent=1))
json.dump(result, open("calib_debug.json", "w"), indent=1)
