"""FlyPoet v2.2 — modernized blocks (SDPA/FlashAttention, RoPE, RMSNorm,
SwiGLU) while keeping the fly-mechanism hooks (k-WTA on attention output).

Parameter budget aligned with the 0.1B std arm: SwiGLU hidden = 2048 keeps
FFN params equal to a 3072 GELU FFN. Both arms train with this same file;
the A/B diff stays ONLY the k-WTA module.
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
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data_v2")


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


def rope_cache(seq, d, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, d, 2, device=device).float() / d))
    t = torch.arange(seq, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)          # (T, d/2) each


def apply_rope(x, cos, sin):
    """x: (B, T, H, Dh) with even Dh. Rotates consecutive pairs."""
    T = x.shape[1]
    c, s = cos[:T].view(1, T, 1, -1), sin[:T].view(1, T, 1, -1)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)


class KWTA(nn.Module):
    """Channel sparsification. impl:
      'torch'  exact top-k keep
      'cuda'   adaptive threshold (Krotov-Hopfield style)
      'energy' keep the smallest channel set holding e_frac of |x| energy
               (graded, per-token budget — no fixed keep rate)"""
    def __init__(self, k_frac, impl="torch", e_frac=0.90):
        super().__init__()
        self.k_frac = k_frac
        self.impl = impl
        self.e_frac = e_frac
        self._cuda_fn = None
        self.last_keep = None      # tensor: mean keep fraction, set in forward

    def forward(self, x):
        d = x.shape[-1]
        k = max(1, int(d * self.k_frac))
        if self.impl == "cuda":
            if self._cuda_fn is None:
                from adaptive_kwta import adaptive_kwta
                self._cuda_fn = adaptive_kwta
            in_dtype = x.dtype
            y = self._cuda_fn(x.float(), self.k_frac)
            return y.to(in_dtype)
        if self.impl == "energy":
            in_dtype = x.dtype
            xf = x.float()
            mag = xf.abs()
            total = mag.sum(-1, keepdim=True) + 1e-8
            s, _ = torch.sort(mag, dim=-1, descending=True)
            csum = s.cumsum(-1)
            k_i = (csum < self.e_frac * total).sum(-1, keepdim=True)
            k_i = k_i.clamp(min=1, max=d)
            thr = s.gather(-1, k_i - 1)            # value of last kept channel
            keep = mag >= thr
            self.last_keep = keep.detach().float().mean()
            return (xf * keep).to(in_dtype)
        thr = torch.kthvalue(x, d - k + 1, dim=-1, keepdim=True).values
        return x * (x >= thr)


class Block(nn.Module):
    """Pre-RMSNorm block: RoPE-MHA (SDPA) with optional k-WTA on the
    attention output, then SwiGLU FFN."""

    def __init__(self, d, heads, ffn_h, kwta_opts, cos, sin):
        super().__init__()
        self.heads, self.dh = heads, d // heads
        self.ln1 = RMSNorm(d)
        self.wqkv = nn.Linear(d, 3 * d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.ln2 = RMSNorm(d)
        self.w13 = nn.Linear(d, 2 * ffn_h, bias=False)   # SwiGLU gate+value
        self.w2 = nn.Linear(ffn_h, d, bias=False)
        if kwta_opts:
            self.kwta = KWTA(kwta_opts.get("k_frac", 0.10),
                             impl=kwta_opts.get("impl", "torch"),
                             e_frac=kwta_opts.get("e_frac", 0.90))
        else:
            self.kwta = None
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
        B, T, _ = x.shape
        h = self.ln1(x)
        qkv = self.wqkv(h).view(B, T, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rope(q, self.cos, self.sin)
        k = apply_rope(k, self.cos, self.sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(B, T, -1)
        if self.kwta is not None:
            a = self.kwta(a)
        x = x + self.wo(a)
        g, u = self.w13(self.ln2(x)).chunk(2, dim=-1)
        x = x + self.w2(F.silu(g) * u)
        return x


class GPT(nn.Module):
    def __init__(self, V, d=768, layers=12, heads=12, ffn_h=2048, seq=256, kwta_opts=None):
        super().__init__()
        cos, sin = rope_cache(seq, d // heads, "cpu")   # per-head dim for RoPE
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([
            Block(d, heads, ffn_h, kwta_opts, cos, sin) for _ in range(layers)])
        self.lnf = RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.emb.weight
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * layers))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, collect_final=False):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
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
    ap.add_argument("--kfrac", type=float, default=0.10,
                    help="k-WTA channel keep fraction (sparsity sweep)")
    ap.add_argument("--impl", choices=["torch", "cuda", "energy"], default=None,
                    help="override k-WTA implementation")
    ap.add_argument("--e_frac", type=float, default=0.90,
                    help="energy target for impl=energy")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--ffn_h", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--sleep_every", type=int, default=0,
                    help="if >0, run a forgetting shower every N grad steps")
    ap.add_argument("--sleep_len", type=int, default=0,
                    help="shower steps per sleep phase (data-free, no grad)")
    ap.add_argument("--snap_every", type=int, default=0,
                    help="save a model snapshot every N steps")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _k02")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    corpus = Corpus()
    kwta_opts = None
    if args.arm in ("flynetS", "flynetS_adaptive"):
        impl = args.impl or ("cuda" if args.arm == "flynetS_adaptive" else "torch")
        kwta_opts = {"impl": impl, "k_frac": args.kfrac, "e_frac": args.e_frac}
    model = GPT(corpus.V, d=args.d, layers=args.layers, heads=args.heads,
                ffn_h=args.ffn_h, kwta_opts=kwta_opts).to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[{args.arm}{args.tag}] k={args.kfrac:g} impl={kwta_opts['impl'] if kwta_opts else '-'} "
          f"d={args.d} L={args.layers} seed={args.seed} | {nparam / 1e6:.1f}M params | bf16",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=6e-4,
                                                total_steps=args.steps, pct_start=0.03)
    os.makedirs(os.path.join(ROOT, "logs_v2"), exist_ok=True)
    name = f"{args.arm}{args.tag}"
    curve = open(os.path.join(ROOT, "logs_v2", f"{name}_curve.jsonl"), "a", encoding="utf-8")
    probes = open(os.path.join(ROOT, "logs_v2", f"{name}_probes.jsonl"), "a", encoding="utf-8")
    samples = open(os.path.join(ROOT, "logs_v2", f"{name}_samples.txt"), "a", encoding="utf-8")

    @torch.no_grad()
    def val_loss():
        was = model.training
        model.eval()
        try:
            # FIXED val windows (seeded generator): same windows every call,
            # removes eval-sampling noise from checkpoint comparisons
            gen = torch.Generator().manual_seed(1234)
            ls = []
            for _ in range(24):
                ix = torch.randint(len(corpus.val) - 256 - 1, (16,), generator=gen)
                x = torch.stack([torch.from_numpy(corpus.val[i:i + 256]) for i in ix]).to(DEV)
                y = torch.stack([torch.from_numpy(corpus.val[i + 1:i + 1 + 256]) for i in ix]).to(DEV)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, l = model(x, y)
                ls.append(l.item())
            return float(np.mean(ls))
        finally:
            if was:
                model.train()

    @torch.no_grad()
    def probe(step, vl):
        was = model.training
        model.eval()
        try:
            x, _ = corpus.batches(corpus.val, 8, 256)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(x, collect_final=True)
            h = model._last_hidden[::4, ::4].float().cpu().numpy().reshape(-1, model._last_hidden.shape[-1])
            erank = effective_rank(h[:2000])
            keeps = [float(b.kwta.last_keep) for b in model.blocks
                     if b.kwta is not None and b.kwta.last_keep is not None]
            keep_tag = round(sum(keeps) / len(keeps), 4) if keeps else None
            idx = torch.tensor([[corpus.stoi.get(c, 1) for c in "春"]], device=DEV)
            for _ in range(150):
                logits, _ = model(idx[:, -256:])
                probs = F.softmax(logits[:, -1].float() / 0.9, dim=-1)
                idx = torch.cat([idx, torch.multinomial(probs, 1)], 1)
            text = "".join(corpus.itos.get(str(i), "") for i in idx[0].tolist())
            d3 = distinct_n(text, 3)
            rec = {"step": step, "val": round(vl, 4),
                   "erank": round(erank, 1), "distinct3": round(d3, 3)}
            if keep_tag is not None:
                rec["keep"] = keep_tag
            probes.write(json.dumps(rec) + "\n")
            probes.flush()
            samples.write(f"\n== step {step} val={vl:.3f} erank={erank:.0f} ==\n{text}\n")
            samples.flush()
            print(f"    val={vl:.4f} erank={erank:.0f} d3={d3:.2f}", flush=True)
        finally:
            if was:
                model.train()

    t0 = time.time()
    tok_total = 0
    model.train()

    @torch.no_grad()
    def shower_step():
        """Active-forgetting shower: decay low-|w| band (Berry 2018 / SHY)."""
        for p in model.parameters():
            if p.ndim != 2:
                continue
            flat = p.data.abs().flatten()
            k = max(1, int(flat.numel() * 0.10))
            thr = flat.kthvalue(k).values
            low = p.data.abs() <= thr
            p.data[low] *= (1 - 1e-3)

    for step in range(1, args.steps + 1):
        x, y = corpus.batches(corpus.train, args.batch, 256)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        sched.step()
        if args.sleep_len and args.sleep_every and step % args.sleep_every == 0:
            for _ in range(args.sleep_len):
                shower_step()
            print(f"    sleep phase done after step {step}", flush=True)
        tok_total += x.numel()
        if step % 100 == 0:
            rec = {"step": step, "loss": round(loss.item(), 4),
                   "tps": round(tok_total / (time.time() - t0)), "gn": round(gn, 3)}
            curve.write(json.dumps(rec) + "\n")
            curve.flush()
            print(rec, flush=True)
        if args.snap_every and step % args.snap_every == 0:
            torch.save(model.state_dict(),
                       os.path.join(ROOT, "logs_v2", f"{name}_snap{step}.pt"))
        if step % 2000 == 0 or step == args.steps:
            vl = val_loss()
            probe(step, vl)

    torch.save(model.state_dict(), os.path.join(ROOT, "logs_v2", f"{name}_model.pt"))
    json.dump({"arm": args.arm, "tag": args.tag, "k_frac": args.kfrac,
               "impl": kwta_opts["impl"] if kwta_opts else None,
               "e_frac": args.e_frac, "d": args.d, "layers": args.layers,
               "heads": args.heads, "ffn_h": args.ffn_h,
               "seed": args.seed, "steps": args.steps, "params_M": nparam / 1e6},
              open(os.path.join(ROOT, "logs_v2", f"{name}_final.json"), "w"))
    print(f"[{name}] DONE", flush=True)


if __name__ == "__main__":
    main()
