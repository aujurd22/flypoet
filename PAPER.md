# What Survives When Fruit-Fly Memory Mechanisms Are Transplanted into a Language Model?

*FlyPoet working paper, v2 — 2026-09-23. Consolidates ~30 controlled experiments across four model scales (92.6M–478M) and five matched evaluation points, all char-level Chinese-literature LMs trained from scratch on one consumer GPU. Every number is reproducible from committed scripts; three of our own earlier claims were retracted after stronger controls and are reported as such.*

## Abstract

We transplant four Drosophila mushroom-body mechanisms into a small transformer language model and audit, with pre-registered criteria and matched controls, which ones survive. (1) *Sparse competition* (winner-take-all channel activation, trained in from scratch): a nine-point sweep finds a U-shaped optimum near 25% kept channels that beats dense by ~0.08 nats; the advantage holds across three seeds, four clean scale points (92.6M–477.8M), and a 4× data increase — with one configuration-bound exception we dissect. Trained-in sparse codes turn out to be *causally load-bearing* (matched-fraction ablation does 1.76× more damage than random) and function as *topic-level addresses* (2.6× chance), while remaining semantically silent and developmentally frozen (cross-snapshot stability 0.45→0.89 while code–PMI correlation stays at 0.03). (2) *Gated memory writes*: restricting fine-tuning updates cuts catastrophic forgetting by ~80%, but a random-skip control shows the benefit is entirely update *throttling* — surprise selectivity adds nothing measurable. (3) *Active forgetting*: a targeted low-magnitude decay initially appeared to repair an overfit tail; a fixed-window re-verification shows the true effect is zero and the observation was evaluation noise. (4) *Parameter compartments* contribute little beyond the throttle. The surviving story is austere: of the fly's mechanisms, what transfers is not sparse *representation* but conservative *memory management* — write less, partition parameters, and never forget uniformly.

## 1. Why transplant, and what to expect

The Drosophila mushroom body is the best-characterized memory system in neuroscience: ~2,000 Kenyon cells (KCs) sparsify odor input through winner-take-all competition, dopaminergic neurons gate plasticity by prediction error, learned associations are compartmentalized, and dedicated circuitry actively erases memories. Each mechanism has an obvious machine-learning analogue, and "bio-inspired" papers routinely report that one of them helps. We ask a harder question: **when each mechanism is given a matched control, which benefits survive?** The answer, across ~30 experiments, is a strict subset of what the bio-inspiration suggests — and knowing which parts die is as useful as knowing which live.

## 2. Setup

Char-level GPT, 92.6M parameters (d=768, 12 layers, 12 heads, SwiGLU-2048, RoPE, RMSNorm, tied embeddings), Chinese literary corpus (~0.9B-token stream); AdamW + OneCycle; main protocol 12k steps (batch 24×256), extensions to 24k/48k; scale ladder to 216M/334M/478M (d=1024–1536). k-WTA is applied to each block's attention output: keep the top-k channels by value, with k a training-time constant unless stated. All validation comparisons in this version use a **fixed validation-window protocol** (seeded generator) after we discovered the original random 24-batch protocol swings ±0.06 nats on an unchanged function — enough to fabricate effects of the size we care about.

## 3. Results

### 3.1 Sparsity: a robust U-shaped sweet spot, not a fly constant

Nine-point sweep at 92.6M (12k steps): 2%→4.156, 5%→3.992, 10%→3.889, 15%→3.860, **25%→3.827**, 40%→3.850, 50%→3.873, 60%→3.899, dense→3.906. Moderate sparsity beats dense at every checkpoint measured, across three seeds (k25 3.819±0.016 vs dense 3.875±0.027, 3/3 seed wins), and across a 4× data increase (48k steps: 3.525 vs 3.546). The advantage peaks at the largest scale: raising the training budget at 478M from 12.3M to 24.6M tokens roughly *doubles* the gap (from +0.090 to +0.180 nats) — the opposite of the shrink-with-training pattern at 92.6M. Notably, within the 478M 24k-step run itself the k25 arm trails at the 12k probe (−0.057) and crosses over by 24k (+0.180): at this scale the sparse advantage *emerges late*.

The early-calibration story corrected itself: at 12k the sparse arms looked dramatically better calibrated (top-1 accuracy 0.31–0.33 vs 0.08–0.10, ECE roughly halved); by 24k dense caught up on both. We therefore claim *convergence speedup and a mid-training advantage*, not a permanent calibration gain — the original observation came from comparing against an undertrained baseline.

### 3.2 The one inversion is configuration-bound

At 216M under the original protocol (batch 16, 393M tokens), k25 *lost* to dense by 0.139 — the only reversal anywhere in the project. Rerunning the same scale under the standard batch-8 protocol (24.6M tokens) erases it (dense 4.870 vs k25 4.855); note the original pair is internally batch-matched at 49M tokens. The elite-channel census adds a structural lead: the inverted point has the most concentrated win-rate distribution of any scale (Gini 0.578 vs 0.50–0.57). We do not claim the batch size is the cause; we claim the sign of the k25-vs-dense gap *flips with the training configuration at this scale* (−0.139 at 49M tokens/batch 16, +0.015 at 24.6M/batch 8), that both points are internally consistent far beyond evaluation noise, and that the mechanism is genuinely unresolved. Resolving it is the primary target of the scale×budget grid. The fly's ~5% KC sparsity does not transfer as a number.

### 3.3 Continual learning: throttling and compartments, not surprise

Sequential fine-tuning on four domains (1500 steps each) from the 24k dense checkpoint: plain FT forgets 0.692 nats, and two of four domains show *zero or negative* improvement over the never-finetuned base. Adding a surprise gate (update only when batch loss exceeds a running noise floor μ+0.25σ, EMA) plus random 30% parameter compartments cuts forgetting to 0.143 and turns all four domains positive (+0.735 mean).

Three controls dissect this:
- **Sparse trunk, plain FT**: 0.674 — sparse representation alone does nothing for interference.
- **Compartments only** (no gate): 0.380.
- **Random-skip** (same compartments, coin-flip batch skipping at the gate's pass rate): **0.130 / +0.745 vs the gate's 0.143 / +0.735** — statistically indistinguishable.

The verdict: the benefit is *update throttling plus parameter partitioning*. "Write only when surprised" — the most biological-looking part — adds nothing measurable over a coin flip at matched budget. A dose-response over pass rates (15/30/50/90%) is cleanly monotone and the gate sits on the curve. An entropy-gate variant (no targets needed) matches the loss gate, so the throttle transfers to unlabeled streams. Boundary: on a heavily undertrained 334M base, everything is surprising, the gate degenerates to pass-through, and plain FT overtakes it (+1.26 vs +0.98) — gating requires a base with a meaningful prediction baseline.

### 3.4 Active forgetting: retracted

Our initial report claimed 500 data-free steps of low-magnitude decay repair an overfit tail (3.65→3.59) while uniform decay destroys the model (→6.60). The destruction is real (100+ evaluation sigma); the *repair* was not — under fixed validation windows the shower's effect is exactly zero across four base checkpoints (max |Δ| 0.002), while the original random protocol swings ±0.06 on an unchanged function. No active-erasure benefit is currently demonstrated in this setup.

### 3.5 What the sparse code is — and is not

A five-part characterization of the trained k25 code (top-25% channels of the final hidden state):
1. **Stable**: same-char codes across independent contexts overlap 1.5× more than different-char pairs.
2. **Compositional (provisionally)**: bigram codes overlap the union of their char codes 1.67× more than random pairs from the same frequent-char pool (caveat: the baseline is pool-uniform, not frequency-matched).
3. **Causally load-bearing**: matched-fraction ablation of the code channels does 1.76× more damage (KL) than random-25% ablation.
4. **A topic address**: Hamming lookup over an address book retrieves same-domain neighbours at 2.6× chance (hit@1 0.325), tying dense cosine.
5. **Not semantic, developmentally frozen**: code–PMI correlation is 0.03 (dense: 0.21) and never develops; cross-snapshot code stability rises 0.45→0.89 over training. The address system freezes; semantics never enter.
6. **Elite structure without elite identity**: channel win-rates are concentrated (Gini 0.50–0.58, rising with scale; the inversion point is the most concentrated), but cross-seed elite overlap is at chance (6.7% vs 6.5% expected).
7. **Stability is the mechanism, not competition**: a four-arm selection control (top-k vs random-k vs fixed-k vs sigmoid-gate, all at 25% keep) shows three *stable*-subset mechanisms tie exactly (3.823–3.827) while random-k — the only one that re-draws the subset every step — loses by 0.24 nats. The benefit comes from a *stable channel subset structure* that downstream weights can adapt to; neither competition, input-dependence, nor dynamics are necessary. This also reframes the code-stability findings: they are largely a byproduct of subset constancy.

### 3.6 Further negatives

k-WTA probe on frozen Qwen3-0.6B features loses to a linear probe (0.570 vs 0.642) — the mechanism must be trained in. A single-window contamination detector built from four trace features (NLL, activation rate, max-NLL, bigram score) does not beat NLL alone (ensemble 0.517 vs 0.605); set-level detection (AUC 0.93) remains the only usable mode. Per-token activation rate carries no seen/unseen signal (OOD AUC 0.5005). Inference-time elasticity shows the trained model tolerates *reducing* k gracefully (+0.05 nats at 10%) but collapses when *increasing* it (k=100% is worse than chance) — weights co-adapt to exactly one sparsity.

## 4. What survives

| Mechanism | Verdict | Effect |
|---|---|---|
| Write throttling | **survives** | −80% forgetting; dose-response monotone; surprise selectivity adds nothing |
| Parameter compartments | **survives (secondary)** | +0.45 improvement without gate; small beyond throttle with it |
| Trained-in 25% sparsity | **survives** | beats dense at 4 clean scale points; U-shaped; elastic downward only |
| Targeted (non-uniform) forgetting | **open** | repair claim retracted; "band decay is a no-op, uniform decay destroys" stands |
| Surprise selectivity | **dies** | random-skip control |
| Sparse code as semantic representation | **dies** | PMI 0.03; retrieval ties dense |
| Sparse code as address/index | **lives (partial)** | stable, completable, causal, topic-level 2.6× |
| Fly-derived sparsity constants (5%) | **dies** | optimum is regime-dependent; 25% here |

## 5. Limitations

Char-level, Chinese-only, single architecture family; scale ladder tops out at 478M with 12.3M–74M tokens (all points are data-starved by Chinchilla standards — which is the regime we study, but the sweet spot's behavior in data-rich regimes is untested); CL protocols use one domain order and one seed for most arms (two seeds for the headline comparisons); the shower retraction rests on a fixed-window protocol adopted after the fact; the compositionality baseline is pool-uniform rather than frequency-matched.

## 6. Reproducing

```bash
python train_v2.py --arm std --steps 24000
python train_v2.py --arm flynetS --kfrac 0.25 --tag _k25 --steps 12000
python cl_experiment.py fly 1500 std          # gated continual learning
python cl_experiment.py skip 1500 std         # random-skip control
python shower_verify.py                       # fixed-window shower audit
python code_address.py && python sparse_retrieval.py && python domain_retrieval.py
python calibration_eval.py --arm std --model logs_v2/std_model.pt --seed 0
```

Reports: REPORT_V2.md (three-arm study, sweep, scale ladder), REPORT_MEM.md (memory trilogy, follow-ups, retractions), NOTES_RLCD.md (pre-registration and full methodology log).
