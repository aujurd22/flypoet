"""Three-way shuffle control for the causal-code claim (Cadence-style gate).

Previous result (code_transplant.py): zeroing the trained top-25% code does
1.76x more KL damage than zeroing a random 25%. Missing control: is the
SPECIFIC input's code what matters, or just any magnitude-ranked subset?

Conditions (all zero exactly 25% of last-position channels, head-input hook):
  zero_own        the window's own top-25% code (by value)
  zero_mismatch   the top-25% code of a DIFFERENT window in the batch
                  (the shuffle gate: breaks the code-input binding)
  zero_random     a random 25% subset
Prediction if the code is input-specific computation:
  own > mismatch > random (KL damage).
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import train_v2 as T

DEV = "cuda"
SEQ = 64
N_WIN = 400
D = 768


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "flynetS_k25_24k"
    torch.manual_seed(11)
    corpus = T.Corpus()
    model = T.GPT(corpus.V, kwta_opts={"impl": "torch", "k_frac": 0.25}).to(DEV)
    sd = torch.load(os.path.join(ROOT, "logs_v2", f"{which}_model.pt"),
                    map_location=DEV, weights_only=True)
    model.load_state_dict(sd)
    model.eval()

    # mutable mask cell consulted by the hook; None = no ablation
    cell = {"mask": None}

    def pre_hook(module, args):
        h = args[0]
        m = cell["mask"]
        if m is None:
            return None
        h = h.clone()
        h[:, -1, :][m] = 0
        return (h,)

    handle = model.head.register_forward_pre_hook(pre_hook)
    torch.manual_seed(11)
    ix = torch.randint(len(corpus.val) - SEQ - 1, (N_WIN,))
    X = torch.stack([torch.from_numpy(corpus.val[i:i + SEQ]) for i in ix])

    acc = {m: {"kl": [], "flip": []} for m in
           ["zero_own", "zero_mismatch", "zero_random"]}
    for i in range(0, len(X), 50):
        xb = X[i:i + 50].to(DEV)
        B = xb.shape[0]
        # pass 1: no ablation -> original logits + per-window own mask
        cell["mask"] = None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(xb, collect_final=True)
        h_last = model._last_hidden[:, -1].float()               # (B, d)
        thr = torch.kthvalue(h_last, int(D * 0.75), dim=-1,
                             keepdim=True).values
        own = h_last >= thr                                      # (B, d) bool
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            orig_logits, _ = model(xb)
        orig = orig_logits[:, -1].float()
        ref = F.log_softmax(orig, dim=-1)
        orig_pred = orig.argmax(dim=-1)

        g = torch.Generator(device=DEV).manual_seed(11 + i)
        rnd = torch.rand(B, D, device=DEV) < 0.25
        mismatch = own[torch.randperm(B, device=DEV, generator=g)]
        for mode, mask in [("zero_own", own), ("zero_mismatch", mismatch),
                           ("zero_random", rnd)]:
            cell["mask"] = mask
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(xb)
            lg = logits[:, -1].float()
            ab = F.log_softmax(lg, dim=-1)
            kl = float(F.kl_div(ab, ref, log_target=True, reduction="batchmean"))
            flip = float((lg.argmax(dim=-1) != orig_pred).float().mean())
            acc[mode]["kl"].append(kl)
            acc[mode]["flip"].append(flip)
        cell["mask"] = None

    out = {}
    for m, v in acc.items():
        out[m] = {"kl": round(float(np.mean(v["kl"])), 4),
                  "top1_flip": round(float(np.mean(v["flip"])), 4)}
        print(f"{m}: KL={out[m]['kl']} top1_flip={out[m]['top1_flip']}",
              flush=True)
    ratio_own_rand = round(out["zero_own"]["kl"] / max(out["zero_random"]["kl"], 1e-9), 3)
    ratio_own_mis = round(out["zero_own"]["kl"] / max(out["zero_mismatch"]["kl"], 1e-9), 3)
    out["ratio_own_vs_random"] = ratio_own_rand
    out["ratio_own_vs_mismatch"] = ratio_own_mis
    print(f"ratios: own/random={ratio_own_rand} own/mismatch={ratio_own_mis}",
          flush=True)
    with open(os.path.join(ROOT, "logs_v2", "code_shuffle_result.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved logs_v2/code_shuffle_result.json", flush=True)
    handle.remove()


if __name__ == "__main__":
    main()
