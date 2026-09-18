"""FlyPoet: train a ~0.1B char-level Chinese-literature model from scratch.

Two arms, same seed & budget:
  arm "std"  — vanilla GPT
  arm "fly"  — fruit-fly parts: k-WTA sparse attention (top 10%),
               MB-style sparse FFN (top 25%), error-gated updates
               (gatedlearn ErrorGatedOptimizer)

Usage:  python train_flypoet.py --arm std   /  --arm fly
Logs:   logs/<arm>_curve.jsonl, samples written every EVAL_EVERY steps.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(7)
np.random.seed(7)
DEV = "cuda"
DATA = r"D:/user/flypoet/data"

# ---------------- data
def load_data(seq_len=256):
    vocab = json.load(open(os.path.join(DATA, "vocab.json"), encoding="utf-8"))
    stoi = vocab["stoi"]
    V = len(stoi)
    train = np.fromfile(os.path.join(DATA, "train.bin"), dtype=np.uint16).astype(np.int64)
    val = np.fromfile(os.path.join(DATA, "val.bin"), dtype=np.uint16).astype(np.int64)
    return V, train, val

def batches(arr, batch, seq_len, step):
    ix = torch.randint(len(arr) - seq_len - 1, (batch,))
    x = torch.stack([torch.from_numpy(arr[i:i+seq_len]) for i in ix])
    y = torch.stack([torch.from_numpy(arr[i+1:i+1+seq_len]) for i in ix])
    return x.to(DEV), y.to(DEV)

# ---------------- model
class KWTA(nn.Module):
    """Keep top-k activations of the last dim, zero the rest (re-normalised)."""
    def __init__(self, k_frac):
        super().__init__()
        self.k_frac = k_frac
    def forward(self, x):
        k = max(1, int(x.shape[-1] * self.k_frac))
        topk = torch.topk(x, k, dim=-1)
        mask = torch.zeros_like(x).scatter_(-1, topk.indices, 1.0)
        return x * mask

class Block(nn.Module):
    def __init__(self, d, heads, ffn_h, cfg):
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_h), nn.GELU(), nn.Linear(ffn_h, d))
        self.attn_kwta = KWTA(cfg["attn_keep"]) if cfg.get("attn_kwta") else None
        self.ffn_kwta = KWTA(cfg["ffn_keep"]) if cfg.get("ffn_kwta") else None
        self.gain = nn.Parameter(torch.ones(1))   # layer-2 modulation hook

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask)
        if self.attn_kwta is not None:
            a = self.attn_kwta(a)
        x = x + a
        h = self.ln2(x)
        f = self.ffn(h)
        if self.ffn_kwta is not None:
            f = self.ffn_kwta(f)
        x = x + self.gain * f
        return x

class NanoGPT(nn.Module):
    def __init__(self, V, d=768, layers=12, heads=12, ffn=3072, seq=256, cfg=None):
        super().__init__()
        self.seq = seq
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(seq, d)
        cfg = cfg or {}
        self.blocks = nn.ModuleList([Block(d, heads, ffn, cfg) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.emb.weight     # weight tying

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))
        mask = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), 1)
        for b in self.blocks:
            x = b(x, mask)
        x = self.lnf(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ---------------- gatedlearn (fly arm)
def make_optimizer(model, arm):
    import sys
    sys.path.insert(0, r"D:/user/gatedlearn")
    from gatedlearn import ErrorGatedOptimizer
    base = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1)
    if arm == "fly":
        return ErrorGatedOptimizer(base, z_threshold=3.5, ema=0.98,
                                   scale_lr=True, warmup=200)
    class Passthrough:
        def zero_grad(self): base.zero_grad()
        def step(self, loss=None, **kw): base.step(); return {}
        stats = {"steps": 0, "gated_off": 0, "accepted": 0}
    return Passthrough()

# ---------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["std", "fly"], required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=24)
    args = ap.parse_args()

    V, train, val = load_data()
    cfg = {}
    if args.arm == "fly":
        cfg = {"attn_kwta": True, "attn_keep": 0.10, "ffn_kwta": True, "ffn_keep": 0.25}
    model = NanoGPT(V, cfg=cfg).to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[{args.arm}] params: {nparam/1e6:.1f}M")

    gate = make_optimizer(model, args.arm)
    os.makedirs("logs", exist_ok=True)
    curve = open(f"logs/{args.arm}_curve.jsonl", "a", encoding="utf-8")
    samples = open(f"logs/{args.arm}_samples.txt", "a", encoding="utf-8")

    def val_loss():
        model.eval()
        with torch.no_grad():
            losses = []
            for _ in range(20):
                x, y = batches(val, 16, 256, 0)
                _, l = model(x, y)
                losses.append(l.item())
        model.train()
        return float(np.mean(losses))

    def gen(prompt_ids, n=120, temperature=0.9):
        model.eval()
        idx = torch.tensor([prompt_ids], device=DEV)
        for _ in range(n):
            ctx = idx[:, -256:]
            logits, _ = model(ctx)
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], 1)
        model.train()
        return idx[0].tolist()

    t0 = time.time()
    step = 0
    tok_total = 0
    while step < args.steps:
        x, y = batches(train, args.batch, 256, step)
        gate.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
        gate.step(float(loss.item()), grad_norm=gn)
        tok_total += x.numel()
        step += 1
        if step % 50 == 0:
            rec = {"step": step, "loss": round(loss.item(), 4),
                   "tps": round(tok_total / (time.time() - t0)), "gn": round(gn, 3)}
            curve.write(json.dumps(rec) + "\n"); curve.flush()
            print(rec, flush=True)
        if step % 1000 == 0 or step == args.steps:
            vl = val_loss()
            curve.write(json.dumps({"step": step, "val": round(vl, 4)}) + "\n"); curve.flush()
            prompt = "床前明月光"
            pid = [vocab_stoi.get(c, 1) for c in prompt]
            out = gen(pid)
            itos = vocab_itos
            text = "".join(itos.get(str(i), "") for i in out)
            samples.write(f"\n== step {step} val={vl:.3f} ==\n{text}\n"); samples.flush()
            print(f"    val={vl:.4f} | {text[:60]}", flush=True)

    json.dump({"arm": args.arm, "params_M": nparam / 1e6, "steps": args.steps},
              open(f"logs/{args.arm}_final.json", "w"))
    torch.save(model.state_dict(), f"logs/{args.arm}_model.pt")

if __name__ == "__main__":
    vocab_stoi = json.load(open(os.path.join(DATA, "vocab.json"), encoding="utf-8"))["stoi"]
    vocab_itos = json.load(open(os.path.join(DATA, "vocab.json"), encoding="utf-8"))["itos"]
    main()
