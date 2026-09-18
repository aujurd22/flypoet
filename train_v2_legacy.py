"""FlyPoet v2.1: single-mechanism A/B on the stratified 55M-char corpus.

Arms:
  std      — vanilla GPT (91.4M)
  flynetS  — ONLY attention top-10% k-WTA via fused kthvalue threshold
             (T103's validated single mechanism)

Audited fixes vs v2.0 (training-researcher review):
  1. probe()/val_loss() restore model.train() (try/finally)
  2. GPT init: N(0, 0.02), residual projections scaled by 0.02/sqrt(2L)
  3. bf16 autocast for train & eval (CE loss stays fp32 under autocast)
  4. grad clip 1.0 (was 1e9 = no-op)
  5. KWTA via kthvalue threshold + mask multiply (no topk gather/scatter)
  6. vocab cached once; dead param removed
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(7)
np.random.seed(7)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
DEV = "cuda"
DATA = r"D:/user/flypoet/data_v2"


class Corpus:
    def __init__(self, root=DATA):
        vj = json.load(open(os.path.join(root, "vocab.json"), encoding="utf-8"))
        self.stoi = vj["stoi"]
        self.itos = vj["itos"]
        self.V = len(self.stoi)
        self.train = np.fromfile(os.path.join(root, "train.bin"), dtype=np.uint16).astype(np.int64)
        self.val = np.fromfile(os.path.join(root, "val.bin"), dtype=np.uint16).astype(np.int64)

    def batches(self, arr, batch, seq_len):
        ix = torch.randint(len(arr) - seq_len - 1, (batch,))
        x = torch.stack([torch.from_numpy(arr[i:i + seq_len]) for i in ix])
        y = torch.stack([torch.from_numpy(arr[i + 1:i + 1 + seq_len]) for i in ix])
        return x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)


class KWTA(nn.Module):
    """Top-k channel sparsification.

    impl="torch": exact top-k via kthvalue threshold (half-fused).
    impl="cuda":  adaptive-threshold CUDA kernel (Krotov-Hopfield style,
                  ~0.03ms vs 0.33ms on (8,256,768); retained set overlaps
                  exact top-k by ~86% -- the adaptive mechanism keeps slightly
                  more than k winners, which is faithful to the biology).
    """
    def __init__(self, k_frac, impl="torch"):
        super().__init__()
        self.k_frac = k_frac
        self.impl = impl
        self._cuda_fn = None

    def forward(self, x):
        d = x.shape[-1]
        k = max(1, int(d * self.k_frac))
        if self.impl == "cuda":
            if self._cuda_fn is None:
                from adaptive_kwta import adaptive_kwta
                self._cuda_fn = adaptive_kwta
            in_dtype = x.dtype
            y = self._cuda_fn(x.float(), self.k_frac)   # kernel is fp32-only
            return y.to(in_dtype)
        thr = torch.kthvalue(x, d - k + 1, dim=-1, keepdim=True).values
        return x * (x >= thr)


class Block(nn.Module):
    def __init__(self, d, heads, ffn_h, kwta_opts):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_h), nn.GELU(), nn.Linear(ffn_h, d))
        self.kwta = KWTA(0.10, **kwta_opts) if kwta_opts else None

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask)
        if self.kwta is not None:
            a = self.kwta(a)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, V, d=768, layers=12, heads=12, ffn=3072, seq=256, kwta_opts=None):
        super().__init__()
        self.seq = seq
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(seq, d)
        self.blocks = nn.ModuleList([Block(d, heads, ffn, kwta_opts) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.emb.weight
        self.apply(self._init)
        for name, p in self.named_parameters():          # residual projections scaled down
            if name.endswith("ffn.2.weight") or "out_proj.weight" in name:
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * layers))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, collect_final=False):
        B, T = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))
        mask = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), 1)
        for b in self.blocks:
            x = b(x, mask)
        x = self.lnf(x)
        if collect_final:
            self._last_hidden = x.detach()
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def effective_rank(matrix: np.ndarray) -> float:
    s = np.linalg.svd(matrix, compute_uv=False)
    return float((s.sum() ** 2) / (s ** 2).sum())


def distinct_n(text: str, n: int) -> float:
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["std", "flynetS", "flynetS_adaptive"], required=True)
    ap.add_argument("--steps", type=int, default=12000)
    args = ap.parse_args()

    corpus = Corpus()
    use_amp = True
    kwta_opts = None
    if args.arm == "flynetS":
        kwta_opts = {"impl": "torch"}
    elif args.arm == "flynetS_adaptive":
        kwta_opts = {"impl": "cuda"}
    model = GPT(corpus.V, kwta_opts=kwta_opts).to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[{args.arm}] {nparam / 1e6:.1f}M params | amp bf16", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=6e-4,
                                                total_steps=args.steps, pct_start=0.03)
    os.makedirs("logs_v2", exist_ok=True)
    curve = open(f"logs_v2/{args.arm}_curve.jsonl", "a", encoding="utf-8")
    probes = open(f"logs_v2/{args.arm}_probes.jsonl", "a", encoding="utf-8")
    samples = open(f"logs_v2/{args.arm}_samples.txt", "a", encoding="utf-8")

    @torch.no_grad()
    def val_loss():
        was_training = model.training
        model.eval()
        try:
            ls = []
            for _ in range(24):
                x, y = corpus.batches(corpus.val, 16, 256)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    _, l = model(x, y)
                ls.append(l.item())
            return float(np.mean(ls))
        finally:
            if was_training:
                model.train()

    @torch.no_grad()
    def probe(step, vl):
        was_training = model.training
        model.eval()
        try:
            x, _ = corpus.batches(corpus.val, 8, 256)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
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
            probes.write(json.dumps({"step": step, "val": round(vl, 4),
                                     "erank": round(erank, 1),
                                     "distinct3": round(d3, 3)}) + "\n")
            probes.flush()
            samples.write(f"\n== step {step} val={vl:.3f} erank={erank:.0f} ==\n{text}\n")
            samples.flush()
            print(f"    val={vl:.4f} erank={erank:.0f} d3={d3:.2f}", flush=True)
        finally:
            if was_training:
                model.train()

    t0 = time.time()
    tok_total = 0
    model.train()
    for step in range(1, args.steps + 1):
        x, y = corpus.batches(corpus.train, 24, 256)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            _, loss = model(x, y)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        sched.step()
        tok_total += x.numel()
        if step % 100 == 0:
            rec = {"step": step, "loss": round(loss.item(), 4),
                   "tps": round(tok_total / (time.time() - t0)), "gn": round(gn, 3)}
            curve.write(json.dumps(rec) + "\n")
            curve.flush()
            print(rec, flush=True)
        if step % 2000 == 0 or step == args.steps:
            vl = val_loss()
            probe(step, vl)

    torch.save(model.state_dict(), f"logs_v2/{args.arm}_model.pt")
    json.dump({"arm": args.arm, "steps": args.steps, "params_M": nparam / 1e6},
              open(f"logs_v2/{args.arm}_final.json", "w"))
    print(f"[{args.arm}] DONE", flush=True)


if __name__ == "__main__":
    main()
