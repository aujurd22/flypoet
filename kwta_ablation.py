"""k-WTA ratio ablation: k_frac = 5%, 10%, 15%, 25% on the 53M corpus.
Each config trains a flynetS_adaptive arm for 4000 steps (quick ablation).
After training, measures: val loss, top-1 accuracy, ECE, distinct3."""
import argparse, json, os, sys, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(7); np.random.seed(7)
DEV = "cuda"
DATA = r"D:\user\flypoet\data_v2"

class Corpus:
    def __init__(self, root=DATA):
        vj = json.load(open(os.path.join(root, "vocab.json"), encoding="utf-8"))
        self.stoi = vj["stoi"]; self.itos = vj["itos"]
        self.V = len(self.stoi)
        self.train = np.fromfile(os.path.join(root, "train.bin"), dtype=np.uint16).astype(np.int64)
        self.val = np.fromfile(os.path.join(root, "val.bin"), dtype=np.uint16).astype(np.int64)
    def batches(self, arr, batch, seq_len):
        ix = torch.randint(len(arr) - seq_len - 1, (batch,))
        x = torch.stack([torch.from_numpy(arr[i:i+seq_len]) for i in ix])
        y = torch.stack([torch.from_numpy(arr[i+1:i+1+seq_len]) for i in ix])
        return x.to(DEV), y.to(DEV)

class KWTA(nn.Module):
    def __init__(self, k_frac):
        super().__init__(); self.k_frac = k_frac
    def forward(self, x):
        d = x.shape[-1]; k = max(1, int(d * self.k_frac))
        thr = torch.kthvalue(x, d - k + 1, dim=-1, keepdim=True).values
        return x * (x >= thr)

class Block(nn.Module):
    def __init__(self, d, heads, ffn_h, use_kwta):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_h), nn.GELU(), nn.Linear(ffn_h, d))
        self.kwta = KWTA(0.10) if use_kwta else None
    def forward(self, x, attn_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask)
        if self.kwta is not None: a = self.kwta(a)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, V, d=768, layers=12, heads=12, ffn=3072, seq=256, use_kwta=False):
        super().__init__()
        self.seq = seq; self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(seq, d)
        self.blocks = nn.ModuleList([Block(d, heads, ffn, use_kwta) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.emb.weight
    def forward(self, idx, targets=None, collect_final=False):
        B, T = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))
        mask = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), 1)
        for b in self.blocks: x = b(x, mask)
        x = self.lnf(x)
        if collect_final: self._last_hidden = x.detach()
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

def effective_rank(matrix):
    s = np.linalg.svd(matrix, compute_uv=False)
    return float((s.sum() ** 2) / (s ** 2).sum())

def distinct_n(text, n):
    grams = [text[i:i+n] for i in range(len(text)-n+1)]
    return len(set(grams)) / max(len(grams), 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["std", "flynetS", "flynetS_adaptive"], required=True)
    ap.add_argument("--steps", type=int, default=12000)
    args = ap.parse_args()
    corpus = T.Corpus()
    kwta_opts = ({"impl": "cuda"} if args.arm == "flynetS_adaptive" else None)
    model = GPT(corpus.V, kwta_opts=kwta_opts).to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[{args.arm}] {nparam/1e6:.1f}M params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=6e-4, total_steps=args.steps, pct_start=0.03)
    os.makedirs("logs_v2", exist_ok=True)
    curve = open(f"logs_v2/{args.arm}_curve.jsonl", "a", encoding="utf-8")
    probes = open(f"logs_v2/{args.arm}_probes.jsonl", "a", encoding="utf-8")
    samples = open(f"logs_v2/{args.arm}_samples.txt", "a", encoding="utf-8")
    @torch.no_grad()
    def val_loss():
        was = model.training; model.eval()
        try:
            ls = []
            for _ in range(24):
                x, y = corpus.batches(corpus.val, 16, 256)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, l = model(x, y)
                ls.append(l.item())
            return float(np.mean(ls))
        finally:
            if was: model.train()
    @torch.no_grad()
    def probe(step, vl):
        was = model.training; model.eval()
        try:
            x, _ = corpus.batches(corpus.val, 8, 256)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(x, collect_final=True)
            h = model._last_hidden[::4, ::4].float().cpu().numpy().reshape(-1, model._last_hidden.shape[-1])
            erank = effective_rank(h[:2000])
            idx = torch.tensor([[corpus.stoi.get(c, 1) for c in "春"]], device=DEV)
            for _ in range(150):
                logits, _ = model(idx[:, -256:])
                probs = F.softmax(logits[:, -1].float() / 0.9, dim=-1)
                idx = torch.cat([idx, torch.multinomial(probs, 1)], 1)
            text = "".join(corpus.itos.get(str(i), "") for i in idx[0].tolist())
            d3 = distinct_n(text, 3)
            probes.write(json.dumps({"step": step, "val": round(vl, 4), "erank": round(erank, 1), "distinct3": round(d3, 3)}) + "\n"); probes.flush()
            samples.write(f"\n== step {step} val={vl:.3f} erank={erank:.0f} ==\n{text}\n"); samples.flush()
            print(f"    val={vl:.4f} erank={erank:.0f} d3={d3:.2f}", flush=True)
        finally:
            if was: model.train()
    t0 = time.time(); tok_total = 0; model.train()
    for step in range(1, args.steps + 1):
        x, y = corpus.batches(corpus.train, 24, 256)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step(); sched.step()
        tok_total += x.numel()
        if step % 100 == 0:
            rec = {"step": step, "loss": round(loss.item(), 4), "tps": round(tok_total/(time.time()-t0)), "gn": round(gn, 3)}
            curve.write(json.dumps(rec) + "\n"); curve.flush(); print(rec, flush=True)
        if step % 2000 == 0 or step == args.steps:
            vl = val_loss(); probe(step, vl)
    torch.save(model.state_dict(), f"logs_v2/{args.arm}_model.pt")
    json.dump({"arm": args.arm, "steps": args.steps, "params_M": nparam/1e6}, open(f"logs_v2/{args.arm}_final.json", "w"))
    print(f"[{args.arm}] DONE", flush=True)

if __name__ == "__main__":
    main()

__zcode_status=$?
if [ "$__zcode_status" -eq 0 ]; then pwd -P > '/c/Users/user/AppData/Local/Temp/zcode-1573038c-7536-4c21-96c4-914703e3279a-cwd'; fi
exit "$__zcode_status"
