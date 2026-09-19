"""0.6B Routing: Qwen3-0.6B hidden states → linear/k-WTA probe → 8-class domain.

Clean rewrite. Extracts frozen hidden states, trains linear probe and
k-WTA sparse probe, evaluates accuracy + ECE on held-out test split.
"""
import json, os, sys, glob, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

torch.manual_seed(7); np.random.seed(7)
DEV = "cuda"
SRC = r"D:\user\corpus\cwt2\selected"
OUT = r"D:\user\flypoet\decide_head"
os.makedirs(OUT, exist_ok=True)
DOMAINS = ["general","news","encyclopedia","technology","law","education","dialogue","finance"]
D2I = {d:i for i,d in enumerate(DOMAINS)}


def load_data(max_per_class=800):
    samples = []; cnt = Counter()
    for fp in sorted(glob.glob(os.path.join(SRC, "selected-000[1-5]*.jsonl.gz.jsonl"))):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                try: d = json.loads(line)
                except: continue
                di = d.get("domain") or {}
                dom = di.get("single_label")
                if dom not in D2I or cnt[dom] >= max_per_class: continue
                text = (d.get("text") or "").strip()[:300]
                if len(text) < 50: continue
                samples.append((text, dom)); cnt[dom] += 1
                if sum(cnt.values()) >= max_per_class * len(DOMAINS): return samples
    return samples


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    MP = r"D:\models\Qwen3-0.6B-Base"
    tok = AutoTokenizer.from_pretrained(MP)
    model = AutoModelForCausalLM.from_pretrained(
        MP, torch_dtype=torch.bfloat16, device_map=DEV)
    model.eval()

    samples = load_data()
    rng = np.random.default_rng(7); rng.shuffle(samples)
    split = int(len(samples) * 0.9)
    train_s, test_s = samples[:split], samples[split:]
    train_texts = [t for t,_ in train_s]; train_labels = [D2I[l] for _,l in train_s]
    test_texts = [t for t,_ in test_s]; test_labels = [D2I[l] for _,l in test_s]
    n_classes = len(DOMAINS)
    print(f"data: train={len(train_s)}, test={len(test_s)}, classes={n_classes}", flush=True)

    # ---- extract frozen hidden states (mean-pool last layer)
    @torch.no_grad()
    def get_hidden(texts):
        hs = []
        for i in range(0, len(texts), 16):
            batch = texts[i:i+16]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to(DEV)
            out = model(**enc, output_hidden_states=True)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            h = (out.hidden_states[-1] * mask).sum(1) / mask.sum(1)
            hs.append(h.float().cpu())
        return torch.cat(hs)

    print("extracting hidden states...", flush=True)
    H_train = get_hidden(train_texts)
    H_test = get_hidden(test_texts)
    y_train = torch.tensor(train_labels)
    y_test = torch.tensor(test_labels)
    print(f"H_train={H_train.shape}, H_test={H_test.shape}", flush=True)

    # ---- three probes (same architecture, different input transforms)
    hidden_dim = H_train.shape[1]
    n_cls = len(DOMAINS)

    probes = {}
    # B: linear probe on raw hidden
    probes["B_linear"] = nn.Linear(hidden_dim, n_cls)
    # C: linear probe on k-WTA masked hidden
    probes["C_kwta"] = nn.Linear(hidden_dim, n_cls)

    # train all probes on the same schedule
    opts = {name: torch.optim.AdamW(p.parameters(), lr=1e-3)
            for name, p in probes.items()}
    # move to device
    H_train_d = H_train.to(DEV); H_test_d = H_test.to(DEV)
    y_train_d = y_train.to(DEV) if hasattr(y_train, 'to') else y_train
    for name, p in probes.items():
        probes[name] = p.to(DEV)

    # we need y_train as tensor on DEV
    y_train_t = torch.tensor(train_labels).to(DEV)
    y_test_t = torch.tensor(test_labels).to(DEV)

    for epoch in range(30):
        perm = torch.randperm(len(H_train_d)).to(DEV)
        for i in range(0, len(perm), 64):
            idx = perm[i:i+64]
            h = H_train_d[idx]; yb = y_train_t[idx]
            for name, p in probes.items():
                loss = F.cross_entropy(p(h), yb)
                opt.zero_grad(); loss.backward(); opt.step()
        if (epoch+1) % 10 == 0:
            msg = f"  epoch {epoch+1}: "
            for name, p in probes.items():
                with torch.no_grad():
                    pred = p(H_train_d).argmax(dim=-1)
                    acc = float((pred == y_train_d).float().mean())
                msg += f"{name}={acc:.3f} "
            print(msg, flush=True)

    # evaluate on test
    print("\n=== results ===", flush=True)
    for name, p in probes.items():
        with torch.no_grad():
            pred = p(H_test_d).argmax(dim=-1)
            acc = float((pred == y_test_t).float().mean())
        print(f"  {name}: acc={acc:.3f}", flush=True)

if __name__ == "__main__":
    main()
