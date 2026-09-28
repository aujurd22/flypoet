# sanity: coverage of the static elite mask over dynamic top-k selections
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_v2 as T

DEV = 'cuda'; SEQ = 256; D = 768; K = 192
corpus = T.Corpus()
model = T.GPT(corpus.V, kwta_opts={'impl': 'torch', 'k_frac': 0.25})
sd = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'logs_v2', 'flynetS_k25_24k_model.pt'),
                map_location=DEV, weights_only=True)
model.load_state_dict(sd); model.to(DEV)
for b in model.blocks:
    orig = b.kwta.forward
    def mk(orig, blk):
        def fwd(x):
            blk.kwta.last_input = x.detach()
            return orig(x)
        return fwd
    b.kwta.forward = mk(orig, b)

# calibrate on train tail (same as harness)
N = 64
counts = [torch.zeros(D, dtype=torch.float64, device=DEV) for _ in model.blocks]
with torch.no_grad():
    for wi in range(N):
        p = len(corpus.train) - (N - wi) * (SEQ + 1)
        x = torch.from_numpy(corpus.train[p:p+SEQ]).unsqueeze(0).to(DEV)
        model(x)
        for li, b in enumerate(model.blocks):
            top = torch.topk(b.kwta.last_input.float(), K, dim=-1).indices.reshape(-1)
            counts[li] += torch.bincount(top, minlength=D).double()

# coverage on val windows: fraction of each token's dynamic top-K channels
# that fall inside the calibrated static set
masks = [torch.topk(c, K).indices.to(DEV) for c in counts]
cover = torch.zeros(len(model.blocks), device=DEV)
with torch.no_grad():
    for wi in range(32):
        p = wi * 512
        x = torch.from_numpy(corpus.val[p:p+SEQ]).unsqueeze(0).to(DEV)
        model(x)
        for li, b in enumerate(model.blocks):
            top = torch.topk(b.kwta.last_input.float(), K, dim=-1).indices  # (1,T,K)
            inset = torch.isin(top.reshape(-1), masks[li]).float().mean()
            cover[li] += inset
cover /= 32
print('per-layer coverage of dynamic top-192 by static set:',
      [round(float(c), 3) for c in cover])
print('mean coverage:', round(float(cover.mean()), 3))
