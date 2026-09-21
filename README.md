# flypoet

The fruit fly mushroom body does sparse coding with a winner-take-all rule (k-WTA): a stimulus lights up a small subset of Kenyon cells and silences the rest. This repo puts fly-derived mechanisms — sparse competition, gated memory writes, parameter compartments, active forgetting — into small language models and measures, with controls, what each one actually does.

Short version after ~20 experiments: **the fly mechanisms that survive controls are memory-management mechanisms (write less, partition parameters, erase selectively), not representation improvements.**

## Setup

A 92.6M-parameter char-level GPT (RoPE / RMSNorm / SwiGLU / SDPA) trained on Chinese literary text, scaled to 216M / 334M for boundary tests. Fly mechanisms under test:

- `flynetS` — exact top-k channel sparsification after attention (~25% kept at the sweet spot)
- `flynetS_adaptive` — same with an online-adapting threshold in a CUDA kernel
- surprise-gated continual learning — update weights only when batch loss exceeds a running noise floor, into random parameter compartments
- forgetting shower — data-free decay of low-magnitude weights

Note: channel-sparse *activation*, not token-sparse attention.

## What survived controls

**Update throttling, not surprise.** Sequential fine-tuning on 4 domains: plain FT forgets 0.69 nats; gate + compartments forget 0.14–0.17. But a random-skip control (same compartments, coin-flip batch skipping at the same rate) matches the gate exactly (0.130 vs 0.143). The active ingredient is updating *less often* plus parameter partitioning — "learning only when surprised" adds nothing measurable. A sparse *representation* alone (k-WTA trunk, plain FT) forgets just as catastrophically as dense (0.674).

**A U-shaped sparsity sweet spot near 25% (9-point sweep).** Keeping 25% of channels beats dense by ~0.08 nats at every checkpoint measured and across 3 seeds; 2% clearly hurts. But it is a *regime* property, not a constant: the advantage flips at an intermediate scale (216M, heavily data-starved) and returns at 334M. Convergence-speedup effects fade with training time; sparse-training gains persist.

**One-way elastic dial.** A model trained at k=25% degrades gracefully when you *reduce* inference-time k (+0.05 nats at 10%), but collapses when you increase it (k=100% is worse than chance). Downward elasticity only.

**Causal, but not semantic, sparse codes.** The trained top-25% code is stable (1.5× cross-context overlap), completable from partial context, compositionally bound (1.67× vs random pairs), causally load-bearing (matched-fraction ablation does 1.76× more damage than random), and developmentally freezing (0.45→0.89 cross-snapshot stability). It is *not* semantic: code-PMI correlation ~0.03 vs 0.21 for dense vectors, and Hamming retrieval merely ties dense cosine. The code is an engine cylinder, not a library catalogue.

**Targeted forgetting helps; uniform forgetting kills.** From an overfit checkpoint, 500 data-free steps of low-magnitude decay recover val (3.65→3.59), while matched uniform decay destroys the model (3.65→6.60). Boundary: the 48k schedules show no overfit tail, so there is nothing for the shower to fix there (re-verification in `shower_verify.py`).

**Entropy gate ≈ loss gate.** Gating writes by predictive entropy (no targets needed) matches loss-based gating almost exactly — the mechanism transfers to unlabeled streams. Boundary: on a heavily undertrained base everything is surprising and the gate degenerates to pass-through.

**Negative results, reported in full:** "free calibration gain" at 12k was an undertraining artifact (dense catches up by 24k); sparse representation does not reduce forgetting; k-WTA probe on frozen features loses to linear; single-window contamination detection is weak (set-level AUC 0.93 works); activation rate carries no seen/unseen signal.

Full numbers, pre-registered criteria, and every correction are in [REPORT_V2.md](REPORT_V2.md) and [REPORT_MEM.md](REPORT_MEM.md) (§10–§12). The consolidated paper draft is [PAPER.md](PAPER.md). Methodology log: [NOTES_RLCD.md](NOTES_RLCD.md) (internal codenames).

## Reproducing

```bash
python train_v2.py --arm std --steps 24000
python train_v2.py --arm flynetS --kfrac 0.25 --tag _k25 --steps 12000
python cl_experiment.py fly 1500 std                 # gated continual learning
python cl_experiment.py skip 1500 std                # random-skip control
python shower_verify.py                              # fixed-eval shower re-verification
python code_address.py && python sparse_retrieval.py # sparse-code analyses
python calibration_eval.py --arm std --model logs_v2/std_model.pt --seed 0
```

`train_v2.py`, `cl_experiment.py` and the newer analysis scripts resolve paths relative to the repo root. A few older helpers show an illustrative `D:\user\...` layout — junction or edit for your machine. Model weights (350MB each) are not committed; curves and eval JSONs live in `logs_v2/`. GPU jobs are strictly serial — the 334M CL runs trigger CUDA sysmem fallback at batch ≥ 16 and stall.
