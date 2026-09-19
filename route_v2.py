"""Clean routing experiment: Qwen3-0.6B hidden states, 3 probe types, 8-class."""
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
    n_cls = len(DOMAINS)
    print(f"data: train={len(train_s)}, test={len(test_s)}, n_cls={n_cls}", flush=True)

    # ---- extract hidden states (mean-pool last layer)
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
    H_train = get_hidden(train_texts).to(DEV)
    H_test = get_hidden(test_texts).to(DEV)
    y_train = torch.tensor(train_labels).to(DEV)
    y_test = torch.tensor(test_labels).to(DEV)
    print(f"H_train={H_train.shape}, H_test={H_test.shape}", flush=True)

    # ---- train simple MLP probe
    probe = nn.Sequential(nn.Linear(H_train.shape[1], 256), nn.GELU(),
                          nn.Linear(256, n_cls))
    probe = probe.to(DEV)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    for epoch in range(30):
        perm = torch.randperm(len(H_train)).to(DEV)
        for i in range(0, len(perm), 64):
            idx = perm[i:i+64]
            loss = F.cross_entropy(probe(H_train[idx]), y_train[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch+1) % 10 == 0:
            with torch.no_grad():
                pred = probe(H_train).argmax(dim=-1)
                print(f"  epoch {epoch+1}: train acc={float((pred==y_train.to(DEV)).float().mean()):.3f}", flush=True)

    # ---- evaluate
    with torch.no_grad():
        pred = probe(H_test).argmax(dim=-1)
        acc = float((pred == y_test.to(DEV)).float().mean())
    print(f"\n=== 0.6B 路由准确率: {acc:.3f} ({int(pred.eq(y_test.to(DEV)).sum())}/{len(test_labels)}) ===")
    json.dump({"accuracy": acc, "test_size": len(test_s), "n_classes": n_cls},
              open(os.path.join(OUT, "qwen06_routing_result.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
