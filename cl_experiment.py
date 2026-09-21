"""Experiment 1: fly-style surprise-gated continual learning.

Fly story: mushroom-body memory writes are gated by dopaminergic prediction
error — no surprise, no write — and memories live in separate compartments.
ML test: sequentially fine-tune a pretrained trunk on 4 domains and measure
catastrophic forgetting of earlier domains.

Arms (all fine-tune the same std-24k checkpoint, same LR/steps/data order):
  ft    plain sequential fine-tune                      (baseline forgetting)
  comp  per-domain compartment masks: each domain may
        update only a fixed random 30% of weight elements (re-sampled per
        domain, like KC compartments)
  fly   comp + surprise gate: take an optimizer step only when the batch
        loss exceeds the running noise floor (mu + 0.25*sigma, EMA-updated)
        — "no surprise, no write"

Protocol: domain sequence news -> dialogue -> law -> technology, 1500 steps
each. After every stage, eval val loss on ALL domains seen so far (seeded
window sampling for comparability). Forgetting_j = loss_j(just after stage j)
- loss_j(final).
"""
import json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import train_v2 as T

DEV = "cuda"
SEQ = 192
BATCH = 24
STEPS = 1500
LR = 1e-4
DOMAINS = ["news", "dialogue", "law", "technology"]
D2I = {"general": 0, "news": 1, "encyclopedia": 2, "technology": 3, "law": 4,
       "education": 5, "dialogue": 6, "finance": 7}
COMP_FRAC = 0.30          # default; override with --comp_frac
GATE_K = 0.25          # step only when loss > mu + K*sigma


def load_rows(split):
    rows = []
    with open(os.path.join(ROOT, "decide_data", f"{split}.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append((d["ids"], d["label"]))
    return rows


def domain_stream(rows, label, stoi):
    sep = stoi.get("\n", 1)
    arr = []
    for ids, lab in rows:
        if lab == label:
            arr.extend(ids[:200])
            arr.append(sep)
    return np.array(arr, dtype=np.int64)


def batches(arr, batch, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    ix = torch.randint(len(arr) - SEQ - 1, (batch,))
    x = torch.stack([torch.from_numpy(arr[i:i + SEQ]) for i in ix])
    y = torch.stack([torch.from_numpy(arr[i + 1:i + 1 + SEQ]) for i in ix])
    return x.to(DEV), y.to(DEV)


@torch.no_grad()
def eval_domain(model, arr, n_batches=20):
    model.eval()
    ls = []
    for b in range(n_batches):
        x, y = batches(arr, 16, seed=777 + b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, l = model(x, y)
        ls.append(l.item())
    model.train()
    return float(np.mean(ls))


def comp_masks(model, domain_idx, frac=COMP_FRAC):
    """Element-wise Bernoulli(frac) keep-mask per 2D weight. Tied emb/head
    counted once. LN gains and any bias stay trainable."""
    g = torch.Generator(device=DEV).manual_seed(1000 + domain_idx)
    masks, seen = {}, set()
    for name, p in model.named_parameters():
        if p.ndim != 2 or id(p) in seen:
            continue
        seen.add(id(p))
        masks[name] = (torch.rand(p.shape, generator=g, device=DEV) < frac).float()
    return masks


def run_arm(arm, trunk, corpus, rows_train, rows_val, log, comp_frac=COMP_FRAC,
            gate_k=GATE_K, gate_type="loss"):
    kwta_opts = ({"impl": "cuda"} if trunk == "flynetS_adaptive" else
                 {"impl": "torch", "k_frac": 0.25 if "k25" in trunk else 0.10}
                 if trunk.startswith("flynetS") else None)
    dims = {}
    fin = os.path.join(ROOT, "logs_v2", f"{trunk}_final.json")
    if os.path.exists(fin):
        m = json.load(open(fin))
        dims = dict(d=m.get("d", 768), layers=m.get("layers", 12),
                    heads=m.get("heads", 12), ffn_h=m.get("ffn_h", 2048))
    model = T.GPT(corpus.V, kwta_opts=kwta_opts, **dims).to(DEV)
    sd = torch.load(os.path.join(ROOT, "logs_v2", f"{trunk}_model.pt"),
                    map_location=DEV, weights_only=True)
    model.load_state_dict(sd)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    streams_tr = {d: domain_stream(rows_train, D2I[d], corpus.stoi) for d in DOMAINS}
    streams_va = {d: domain_stream(rows_val, D2I[d], corpus.stoi) for d in DOMAINS}

    # stage -1: base model on all domains
    evals0 = {d: eval_domain(model, streams_va[d]) for d in DOMAINS}
    log.write(json.dumps({"arm": arm, "trunk": trunk, "stage": 0, "domain": None,
                          "evals": evals0}) + "\n")
    log.flush()
    print(f"[{arm}/{trunk}] base evals: " +
          " ".join(f"{d}={v:.3f}" for d, v in evals0.items()), flush=True)

    history = [{"stage": 0, "trained_on": None, "evals": evals0}]
    stage_losses = {}   # domain -> loss right after ITS stage

    for si, dom in enumerate(DOMAINS, start=1):
        masks = comp_masks(model, si, comp_frac) if arm in ("comp", "fly") else None
        mu = sigma = None
        mu_e = sigma_e = None
        n_onset, onset_ls, onset_e = 20, [], []
        gated = total = 0
        arr = streams_tr[dom]
        for step in range(1, STEPS + 1):
            x, y = batches(arr, BATCH)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, loss = model(x, y)
            l = loss.item()
            if mu is None:
                onset_ls.append(l)
                if len(onset_ls) == n_onset:
                    mu = float(np.mean(onset_ls))
                    sigma = float(np.std(onset_ls)) + 1e-6
            else:
                mu = 0.98 * mu + 0.02 * l
                sigma = 0.98 * sigma + 0.02 * abs(l - mu)
            allow = True
            if arm == "fly" and mu is not None:
                if gate_type == "entropy":
                    with torch.no_grad():
                        pr = F.softmax(logits.float(), dim=-1)
                        ent = float(-(pr * (pr + 1e-9).log()).sum(-1).mean())
                    if mu_e is None:
                        onset_e.append(ent)
                        if len(onset_e) == n_onset:
                            mu_e = float(np.mean(onset_e))
                            sigma_e = float(np.std(onset_e)) + 1e-6
                    else:
                        mu_e = 0.98 * mu_e + 0.02 * ent
                        sigma_e = 0.98 * sigma_e + 0.02 * abs(ent - mu_e)
                    allow = ent > mu_e + gate_k * sigma_e
                else:
                    allow = l > mu + gate_k * sigma
                gated += (not allow)
            total += 1
            if allow:
                loss.backward()
                if masks is not None:
                    for name, p in model.named_parameters():
                        mk = masks.get(name)
                        if mk is not None and p.grad is not None:
                            p.grad.mul_(mk)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            if step % 200 == 0:
                log.write(json.dumps({"arm": arm, "stage": si, "domain": dom,
                                      "step": step, "loss": round(l, 4),
                                      "gate_pass": round(1 - gated / total, 3)})
                          + "\n")
                log.flush()
                print(f"[{arm}/{trunk}] s{si} {dom} step{step} loss={l:.3f} "
                      f"gate_pass={1 - gated / total:.2f}", flush=True)

        evals = {d: eval_domain(model, streams_va[d]) for d in DOMAINS[:si]}
        if arm != "ft":   # unseen domains drift too; track for all arms
            for d in DOMAINS[si:]:
                evals[d] = eval_domain(model, streams_va[d])
        stage_losses[dom] = evals[dom]
        history.append({"stage": si, "trained_on": dom, "evals": evals})
        log.write(json.dumps({"arm": arm, "stage": si, "domain": dom,
                              "evals": evals}) + "\n")
        log.flush()
        print(f"[{arm}/{trunk}] after {dom}: " +
              " ".join(f"{d}={v:.3f}" for d, v in evals.items()), flush=True)

    forgetting = {d: round(stage_losses[d] - history[-1]["evals"][d], 4)
                  for d in DOMAINS}
    fwd_gain = {d: round(history[0]["evals"][d] - history[-1]["evals"][d], 4)
                for d in DOMAINS}
    return {"arm": arm, "trunk": trunk, "history": history,
            "forgetting": forgetting, "improvement": fwd_gain}


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "ft"
    global STEPS
    if len(sys.argv) > 2:
        STEPS = int(sys.argv[2])
    trunk = sys.argv[3] if len(sys.argv) > 3 else "std"
    comp_frac = float(sys.argv[4]) if len(sys.argv) > 4 else COMP_FRAC
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 7
    gate_k = float(sys.argv[6]) if len(sys.argv) > 6 else GATE_K
    gate_type = sys.argv[8] if len(sys.argv) > 8 else "loss"
    batch = int(sys.argv[9]) if len(sys.argv) > 9 else BATCH
    globals()['BATCH'] = batch
    global DOMAINS
    if len(sys.argv) > 7 and sys.argv[7]:
        DOMAINS = tuple(sys.argv[7].split(","))
    torch.manual_seed(seed)
    np.random.seed(seed)
    corpus = T.Corpus()
    rows_train = load_rows("train")
    rows_val = load_rows("val")
    logdir = os.path.join(ROOT, "logs_v2")
    os.makedirs(logdir, exist_ok=True)
    suffix = f"_c{comp_frac:g}" if arm in ("comp", "fly") and comp_frac != COMP_FRAC else ""
    if gate_k != GATE_K:
        suffix += f"_g{gate_k:g}"
    if gate_type != "loss":
        suffix += f"_gt-{gate_type}"
    if seed != 7:
        suffix += f"_s{seed}"
    with open(os.path.join(logdir, f"cl_{arm}_{trunk}{suffix}_curve.jsonl"), "a",
              encoding="utf-8") as log:
        res = run_arm(arm, trunk, corpus, rows_train, rows_val, log, comp_frac,
                      gate_k, gate_type)
    avg_f = float(np.mean(list(res["forgetting"].values())))
    avg_i = float(np.mean(list(res["improvement"].values())))
    res["avg_forgetting"] = round(avg_f, 4)
    res["avg_improvement"] = round(avg_i, 4)
    res["comp_frac"] = comp_frac
    res["gate_k"] = gate_k
    res["seed"] = seed
    with open(os.path.join(logdir, f"cl_{arm}_{trunk}{suffix}_result.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"[{arm}/{trunk}{suffix}] DONE avg_forgetting={avg_f:.4f} "
          f"avg_improvement={avg_i:.4f}", flush=True)


if __name__ == "__main__":
    main()
