"""Qwen3-0.6B + fly decision head: domain routing A/B experiment.

Arms:
  A  zero-shot prompt (base model, no training)
  B  LoRA fine-tune on CWT2 domain routing (standard method)
  C  LoRA + k-WTA sparse gate on attention output (fruit-fly mechanism)

Data: CWT2 selected corpus, 8-domain routing, 5300 balanced samples.
"""
import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(7)
DEV = "cuda"
SRC = r"D:\user\corpus\cwt2\selected"
OUT = r"D:\user\flypoet\decide_head"
os.makedirs(OUT, exist_ok=True)

DOMAINS = ["general", "news", "encyclopedia", "technology", "law",
           "education", "dialogue", "finance"]
D2I = {d: i for i, d in enumerate(DOMAINS)}

# ---- load data (first 200 chars per doc, balanced)
def load_data(max_per_class=800):
    from collections import Counter
    samples = []
    cnt = Counter()
    for fp in sorted(__import__("glob").glob(os.path.join(SRC, "selected-000[1-5]*.jsonl"))):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                dominfo = d.get("domain") or {}
                dom = dominfo.get("single_label")
                if dom not in D2I or cnt[dom] >= max_per_class:
                    continue
                text = (d.get("text") or "").strip()[:300]
                if len(text) < 50:
                    continue
                samples.append((text, dom))
                cnt[dom] += 1
                if sum(cnt.values()) >= max_per_class * len(DOMAINS):
                    return samples
    return samples

# ---- tokenizer (Qwen3-0.6B-Base local)
from transformers import AutoTokenizer, AutoModelForCausalLM
MP = r"D:\models\Qwen3-0.6B-Base"
tok = AutoTokenizer.from_pretrained(MP)

random = np.random.default_rng(7)

def main():
    samples = load_data()
    random.shuffle(samples)
    split = int(len(samples) * 0.9)
    train_s, val_s = samples[:split], samples[split:]
    print(f"data: train={len(train_s)}, val={len(val_s)}", flush=True)

    # ---- load base model (bf16)
    model = AutoModelForCausalLM.from_pretrained(
        MP, torch_dtype=torch.bfloat16, device_map=DEV)
    model.eval()

    # ---- Arm A: zero-shot (just classify via prompt, no training)
    # For a fair comparison, use the same decision-head protocol:
    # get last hidden state, train a linear probe on TOP of frozen features
    # vs Arm B/C which fine-tune.

    # --- extract hidden states for probe (all arms share the same features)
    def get_hidden(texts):
        hs = []
        for i in range(0, len(texts), 8):
            batch = texts[i:i+8]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to(DEV)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            # mean-pool last hidden state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            h = (out.hidden_states[-1] * mask).sum(1) / mask.sum(1)
            hs.append(h.float().cpu())
        return torch.cat(hs)

    return

if __name__ == "__main__":
    main()
