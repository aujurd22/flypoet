"""P-COMP-1: compression scoring harness -- how much can a trained k-WTA
model be compressed at inference time, at what quality cost?

Pre-registered (registered in flymemory research/RESEARCH.md before this
ran; git timestamps are the proof). Two arms, four ratios, deterministic
scoring:

  arms
    D  dynamic dial  -- inference top-k with k_frac lowered (the elastic
       dial, re-scored under the deterministic protocol)
    S  static elite mask -- per-layer channel win-rates calibrated on the
       TAIL of train.bin (128 windows, deterministic), then each layer's
       keep-set frozen to its top-r fraction; forward keeps magnitudes
       inside the set, zeroes the rest. Same arithmetic as selector="fixed"
       but the mask is EARNED, not seed-42 random.

  ratios r in {0.25, 0.15, 0.10, 0.05}; baseline = native dynamic k=0.25.

  scoring (per checkpoint x arm x ratio), all on val.bin with FIXED window
  slices (wi*stride, 320 windows of 256 tokens, bf16 autocast, batch 1 --
  no RNG anywhere):
    val_loss            primary quality metric
    delta               val_loss - baseline (same checkpoint)
    verdict per cell    LOSSLESS (delta <= +0.01) / HALF-COST
                        (<= +0.05) / DEGRADED
    compression         activation ratio r; theoretical attention-output
                        FLOP saving (1-r); theoretical sparse-storage bound
                        (W_O rows, noted as small for this architecture)

  pre-registered criteria:
    P1 (selection-mode insensitivity): |delta(S @ 0.25)| <= 0.01 on the
        main checkpoint -- freezing the keep-set to the calibration-chosen
        elite half-set is functionally equivalent to dynamic top-k.
    P2 (exploratory, direction not predicted): S @ 0.10 vs D @ 0.10 --
        does the earned elite subset beat full-channel dynamic selection
        at the same activation budget?
    P3 (headline number): the largest compression with delta <= +0.01
        (the "lossless point") is reported per checkpoint; no single
        scalar "score" is invented (restraint policy).

Run:  py -3.13 bench_compress_score.py [ckpt_name ...]
      default: flynetS_k25_24k flynetS_k25 flynetS_k25_s8
"""
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import train_v2 as T  # noqa: E402

DEV = "cuda"
SEQ = 256
N_EVAL = 320          # fixed windows for scoring
N_CAL = 128           # fixed windows for win-rate calibration (train tail)
RATIOS = [0.25, 0.15, 0.10, 0.05]
STRIDE = 512          # non-overlapping-ish fixed slices


@torch.no_grad()
def fixed_val_loss(model, data):
    """Deterministic val loss over N_EVAL fixed windows, batch 1."""
    model.eval()
    total, count = 0.0, 0
    for wi in range(N_EVAL):
        p = wi * STRIDE
        if p + SEQ + 1 > len(data):
            p = len(data) - SEQ - 1
        x = torch.from_numpy(data[p:p + SEQ]).unsqueeze(0).to(DEV)
        y = torch.from_numpy(data[p + 1:p + SEQ + 1]).unsqueeze(0).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, l = model(x, y)
        total += l.item()
        count += 1
    model.train()
    return total / max(count, 1)


@torch.no_grad()
def calibrate_winrate(model, data, d, k):
    """Per-layer channel win counts over N_CAL fixed train-tail windows."""
    counts = [torch.zeros(d, dtype=torch.float64, device=DEV)
              for _ in model.blocks]
    model.eval()
    for wi in range(N_CAL):
        p = len(data) - (N_CAL - wi) * (SEQ + 1)
        x = torch.from_numpy(data[p:p + SEQ]).unsqueeze(0).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x)  # blocks store last pre-mask attn output via hooks below
        for li, b in enumerate(model.blocks):
            h = b.kwta.last_input
            top = torch.topk(h.float(), k, dim=-1).indices.reshape(-1)
            counts[li] += torch.bincount(top, minlength=d).double()
    model.train()
    total = N_CAL * SEQ
    return [(c / total) for c in counts]


def install_static_mask(model, winrates, r):
    """Replace each block's KWTA.forward with a static elite-set mask."""
    d = winrates[0].shape[0]
    k_keep = max(1, int(round(d * r)))
    handles = []
    for li, b in enumerate(model.blocks):
        if b.kwta is None:
            continue
        wr = winrates[li]
        idx = torch.topk(wr, k_keep).indices.to(DEV)
        mask = torch.zeros(d, device=DEV)
        mask[idx] = 1.0

        def mk(mask):
            def fwd(x):
                return x * mask.to(x.dtype)
            return fwd
        b.kwta.forward = mk(mask)
        handles.append(b.kwta)
    return k_keep


def load_model(ckpt, k_frac=0.25):
    corpus = T.Corpus()
    model = T.GPT(corpus.V, kwta_opts={"impl": "torch", "k_frac": k_frac})
    sd = torch.load(os.path.join(ROOT, "logs_v2", f"{ckpt}_model.pt"),
                    map_location=DEV, weights_only=True)
    model.load_state_dict(sd)
    model.to(DEV)
    return model, corpus


def add_hooks_capture_input(model):
    """KWTA.forward normally discards its input; capture it for calibration
    by wrapping once (before any static install)."""
    for b in model.blocks:
        if b.kwta is None or getattr(b.kwta, "_capture", False):
            continue
        orig_fwd = b.kwta.forward

        def mk(orig, blk):
            def fwd(x):
                blk.kwta.last_input = x.detach()
                return orig(x)
            return fwd
        b.kwta.forward = mk(orig_fwd, b)
        b.kwta._capture = True


def sparse_storage_bound(model, keep_frac, d):
    """Theoretical storage if the static keep-set were exploited: only the
    attention output-projection rows feeding surviving channels differ from
    zero. Returns fraction of TOTAL params that could be dropped."""
    wo_params = sum(p.numel() for n, p in model.named_parameters()
                    if n.endswith("wo.weight") or "attn.wo" in n
                    or "o_proj" in n)
    total = sum(p.numel() for p in model.parameters())
    return (wo_params * (1 - keep_frac)) / total if wo_params else 0.0


def main():
    which_list = sys.argv[1:] or ["flynetS_k25_24k", "flynetS_k25",
                                  "flynetS_k25_s8"]
    all_results = {}
    for ckpt in which_list:
        print(f"\n########## {ckpt} ##########", flush=True)
        model, corpus = load_model(ckpt, k_frac=0.25)
        add_hooks_capture_input(model)
        # probe forward to learn the block width d
        with torch.no_grad():
            probe = torch.zeros(1, SEQ, dtype=torch.long).to(DEV)
            model(probe)
        d = model.blocks[0].kwta.last_input.shape[-1]
        k = max(1, int(d * 0.25))

        rows = []
        # baseline: native dynamic 0.25 (restore original forwards first)
        base_loss = fixed_val_loss(model, corpus.val)
        rows.append({"arm": "D", "r": 0.25, "val": round(base_loss, 5),
                     "delta": 0.0, "note": "native dynamic k25 (baseline)"})
        print(f"baseline dynamic 0.25: val={base_loss:.5f}", flush=True)

        # arm D: dynamic dial
        for r in [x for x in RATIOS if x != 0.25]:
            model2, _ = load_model(ckpt, k_frac=r)
            v = fixed_val_loss(model2, corpus.val)
            del model2
            torch.cuda.empty_cache()
            rows.append({"arm": "D", "r": r, "val": round(v, 5),
                         "delta": round(v - base_loss, 5)})
            print(f"D dynamic {r:g}: val={v:.5f} "
                  f"delta={v-base_loss:+.5f}", flush=True)

        # arm S: static elite mask
        winrates = calibrate_winrate(model, corpus.train, d, k)
        wr = torch.stack(winrates)
        print(f"winrate gini-ish: top-decile share "
              f"{(torch.sort(wr, dim=1, descending=True).values[:, :d//10].sum() / wr.sum()).mean():.3f}",
              flush=True)
        for r in RATIOS:
            model3, _ = load_model(ckpt, k_frac=0.25)
            add_hooks_capture_input(model3)  # keep kwta.forward chain intact
            install_static_mask(model3, winrates, r)
            v = fixed_val_loss(model3, corpus.val)
            del model3
            torch.cuda.empty_cache()
            rows.append({"arm": "S", "r": r, "val": round(v, 5),
                         "delta": round(v - base_loss, 5)})
            print(f"S static {r:g}: val={v:.5f} delta={v-base_loss:+.5f}",
                  flush=True)

        # verdicts + storage bound
        for row in rows:
            dl = row.get("delta", 0.0)
            row["verdict"] = ("LOSSLESS" if dl <= 0.01 else
                              "HALF-COST" if dl <= 0.05 else "DEGRADED")
        store = sparse_storage_bound(model, 0.10, d)
        print(f"theoretical sparse-storage bound @r=0.10 "
              f"(W_O rows only): {store:.1%} of total params", flush=True)

        all_results[ckpt] = {"baseline": base_loss, "rows": rows,
                             "storage_bound_r10": store}
        # lossless point
        s_rows = [row for row in rows if row["arm"] == "S"]
        ll = [row["r"] for row in s_rows if row["delta"] <= 0.01]
        all_results[ckpt]["lossless_point"] = min(ll) if ll else None
        print(f"lossless point (S arm, delta<=+0.01): "
              f"{min(ll) if ll else 'none of ' + str([row['r'] for row in s_rows])}",
              flush=True)
        del model
        torch.cuda.empty_cache()

    ts = int(time.time())
    out = os.path.join(ROOT, "logs_v2", f"compress_score_{ts}.json")
    json.dump({"experiment": "P-COMP-1", "results": all_results},
              open(out, "w"), indent=1)
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    main()
