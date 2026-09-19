"""Evolution experiment: a population of sparsity genotypes under new-
environment selection pressure.

Population (existing 12k checkpoints, one per genotype):
  k02 / k15 / k25 / k40  exact top-k at that keep fraction
  e90                    energy-budget k-WTA (per-token adaptive keep)
  std_s8                 dense control

Environments (novel to every genotype — outside the CL four domains):
  finance -> education, sequentially.

Adaptation = fly-style writing only: surprise gate (update iff batch loss >
mu+0.25 sigma, EMA) x random 30% parameter compartments per environment.

Fitness per genotype per environment:
  TTC   steps until env val loss drops 0.4 below its base (censored at cap)
  drop  final env val - base env val
  retention mean of previously-seen task vals after adapting

Key question: TTC(E2) < TTC(E1)?  — do gated populations learn the second
environment faster (learning-to-learn), and which genotype wins?
"""
import json, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import train_v2 as T
from cl_experiment import (load_rows, domain_stream, batches, eval_domain,
                           comp_masks, D2I, GATE_K)

DEV = "cuda"
SEQ = 192
BATCH = 24
COMP_FRAC = 0.30
LR = 1e-4
ENV_THRESHOLD_DROP = 0.4     # competent = env val <= base - 0.4
POPULATION = ["flynetS_k02", "flynetS_k15", "flynetS_k25", "flynetS_k40",
              "flynetS_e90", "std_s8"]
ENVS = ["finance", "education"]
RETENTION_DOMAINS = ["news", "dialogue", "law", "technology"]


def kwta_opts_for(tag):
    if tag.startswith("std"):
        return None
    if "e90" in tag:
        return {"impl": "energy", "e_frac": 0.90}
    if "k25ad" in tag:
        return {"impl": "cuda", "k_frac": 0.25}
    tail = tag.split("_")[-1]                 # e.g. "k02", "k15", "k40"
    if tail.startswith("k") and tail[1:].isdigit():
        return {"impl": "torch", "k_frac": int(tail[1:]) / 100.0}
    return {"impl": "torch", "k_frac": 0.10}


def adapt(model, arr_tr, arr_va, base_val, opt, seed_idx, steps, log_prefix):
    """Gated+compartment fine-tune; returns TTC/drop/final/gate stats."""
    masks = comp_masks(model, seed_idx, frac=COMP_FRAC)
    mu = sigma = None
    onset = []
    ttc = None
    final = base_val
    for step in range(1, steps + 1):
        x, y = batches(arr_tr, BATCH)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        l = loss.item()
        if mu is None:
            onset.append(l)
            if len(onset) == 20:
                mu = float(np.mean(onset))
                sigma = float(np.std(onset)) + 1e-6
        else:
            mu = 0.98 * mu + 0.02 * l
            sigma = 0.98 * sigma + 0.02 * abs(l - mu)
        allow = True if mu is None else l > mu + GATE_K * sigma
        if allow:
            loss.backward()
            for name, p in model.named_parameters():
                mk = masks.get(name)
                if mk is not None and p.grad is not None:
                    p.grad.mul_(mk)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if step % 50 == 0:
            v = eval_domain(model, arr_va, n_batches=8)
            final = v
            if ttc is None and v <= base_val - ENV_THRESHOLD_DROP:
                ttc = step
            if step % 300 == 0:
                print(f"    {log_prefix} step{step} env_val={v:.3f}", flush=True)
    return {"ttc": ttc if ttc is not None else steps,
            "censored": ttc is None,
            "final_env_val": round(final, 4),
            "drop": round(base_val - final, 4)}


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    torch.manual_seed(7)
    np.random.seed(7)
    corpus = T.Corpus()
    rows_train = load_rows("train")
    rows_val = load_rows("val")
    streams_tr = {e: domain_stream(rows_train, D2I[e], corpus.stoi) for e in ENVS}
    streams_va = {e: domain_stream(rows_val, D2I[e], corpus.stoi) for e in ENVS}
    streams_ret = {d: domain_stream(rows_val, D2I[d], corpus.stoi)
                   for d in RETENTION_DOMAINS}

    results = {}
    seed_idx = 0
    for tag in POPULATION:
        print(f"=== individual {tag} ===", flush=True)
        model = T.GPT(corpus.V, kwta_opts=kwta_opts_for(tag)).to(DEV)
        sd = torch.load(os.path.join(ROOT, "logs_v2", f"{tag}_model.pt"),
                        map_location=DEV, weights_only=True)
        model.load_state_dict(sd)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

        seen_vals = {}
        entry = {"base_env": {}, "envs": {}}
        for env in ENVS:
            base = eval_domain(model, streams_va[env])
            entry["base_env"][env] = round(base, 4)
            seed_idx += 1
            res = adapt(model, streams_tr[env], streams_va[env], base, opt,
                        seed_idx, steps, f"{tag}/{env}")
            seen_vals[env] = res["final_env_val"]
            ret = {d: round(eval_domain(model, streams_ret[d], n_batches=10), 4)
                   for d in RETENTION_DOMAINS}
            ret.update({e: round(v, 4) for e, v in seen_vals.items()})
            res["retention"] = ret
            entry["envs"][env] = res
            print(f"  {env}: TTC={res['ttc']} drop={res['drop']} "
                  f"retention_mean={np.mean(list(ret.values())):.3f}", flush=True)
        results[tag] = entry
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(ROOT, "logs_v2", "evolution_result.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("saved logs_v2/evolution_result.json", flush=True)

    print("\n=== leaderboard ===", flush=True)
    for tag, e in results.items():
        t1 = e["envs"][ENVS[0]]["ttc"]
        t2 = e["envs"][ENVS[1]]["ttc"]
        base_ret = {d: None for d in RETENTION_DOMAINS}
        print(f"{tag}: TTC {t1} -> {t2} "
              f"(transfer {'YES' if t2 < t1 else 'no'}) "
              f"final_finance={e['envs'][ENVS[0]]['final_env_val']}", flush=True)


if __name__ == "__main__":
    main()
