"""Experiment: is the sparse code causally load-bearing?

All evidence so far is correlational (codes are stable/retrievable). Ablate:
zero the bottom-75% channels of the final hidden (KEEP the top-25% code) vs
zero the top-25% (keep the rest) vs a random-25% keep, applied at the LAST
position only, right before the LM head. Measure next-char distribution shift
(KL(orig || ablated)) and top-1 flip rate.

If keep-top25 hurts far more than random25, the trained-in k-WTA subset
carries the computation -> "representation" side. If not, "index" side wins.
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


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "flynetS_k25_24k"
    torch.manual_seed(7)
    corpus = T.Corpus()
    model = T.GPT(corpus.V, kwta_opts={"impl": "torch", "k_frac": 0.25}).to(DEV)
    sd = torch.load(os.path.join(ROOT, "logs_v2", f"{which}_model.pt"),
                    map_location=DEV, weights_only=True)
    model.load_state_dict(sd)
    model.eval()

    torch.manual_seed(11)
    ix = torch.randint(len(corpus.val) - SEQ - 1, (N_WIN,))
    X = torch.stack([torch.from_numpy(corpus.val[i:i + SEQ]) for i in ix])

    d = 768
    modes = ["orig", "zero_top25", "zero_random25", "keep_top25", "random25"]
    captured = {}

    def make_hook(mode):
        def pre_hook(module, args):
            h = args[0]                              # (B, T, d) after lnf
            if mode == "orig":
                captured["orig"] = h.detach()
                return None
            if mode in ("zero_top25", "keep_top25"):
                thr = torch.kthvalue(h[..., -1, :], int(d * 0.75),
                                     dim=-1, keepdim=True).values
                keep = h[..., -1, :] >= thr         # top-25% by value at last pos
            elif mode in ("keep_bottom75",):
                keep = h[..., -1, :] < thr
            elif mode == "zero_random25":
                keep = (torch.rand(d, device=h.device) < 0.75).expand(h.shape[0], d)
            elif mode == "random25":
                keep = (torch.rand(d, device=h.device) < 0.25).expand(h.shape[0], d)
            else:  # orig
                keep = torch.ones(d, device=h.device).expand(h.shape[0], d)
            new = h.clone()
            new[:, -1, :] = h[:, -1, :] * keep
            return (new,)
        return pre_hook

    results = {m: {"kl": [], "flip": []} for m in modes}
    for i in range(0, len(X), 50):
        xb = X[i:i + 50].to(DEV)
        orig_logits = None
        for mode in modes:
            h = model.head.register_forward_pre_hook(make_hook(mode))
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = model(xb)
            finally:
                h.remove()
            logits = logits[:, -1].float()
            if mode == "orig":
                orig_logits = logits
                continue
            ref = F.log_softmax(orig_logits, dim=-1)
            ab = F.log_softmax(logits, dim=-1)
            kl = F.kl_div(ab, ref, log_target=True, reduction="batchmean")
            flip = (logits.argmax(-1) != orig_logits.argmax(-1)).float().mean()
            results[mode]["kl"].append(float(kl))
            results[mode]["flip"].append(float(flip))

    out = {}
    for m in modes:
        out[m] = {"kl": round(float(np.mean(results[m]["kl"])), 4),
                  "top1_flip": round(float(np.mean(results[m]["flip"])), 4)}
        print(f"{m}: KL={out[m]['kl']} top1_flip={out[m]['top1_flip']}", flush=True)
    with open(os.path.join(ROOT, "logs_v2", "code_transplant_result.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved logs_v2/code_transplant_result.json", flush=True)


if __name__ == "__main__":
    main()
