# What the Fruit Fly Teaches a Small Language Model: Sparse Activation, Gated Memory Writes, and Active Forgetting

*FlyPoet working paper — consolidated from REPORT_V2.md and REPORT_MEM.md. All experiments: 92.6M char-level GPT (RoPE / RMSNorm / SwiGLU / SDPA) on Chinese literary text, single consumer GPU. Code in this repository; every number below is reproducible from the committed scripts and curves.*

## Abstract

We transplant three mechanisms of the Drosophila mushroom body into a small transformer language model and test each against pre-registered baselines. (1) Winner-take-all channel sparsification (k-WTA) accelerates convergence: at mid-training the sparse model is calibrated and accurate while the dense baseline is neither, though dense catches up given 2× the training. A sparsity sweep (9 points) shows the gain is U-shaped — keeping 25% of channels beats dense at every checkpoint we measured, while 2% clearly hurts. (2) Fly-style *memory management* — per-task parameter compartments plus an update gate that skips most batches — reduces catastrophic forgetting by ~79% in sequential domain fine-tuning, while sparse activation *per se* does not help at all; a random-skip control shows the benefit comes from updating less often, not from surprise selectivity. (3) A data-free "forgetting shower" (targeted decay of low-magnitude weights) was initially reported to remove the late-training overfitting tail; a fixed-validation-window re-verification shows its true effect is zero across four base checkpoints — the original observation was evaluation noise (the random protocol swings ±0.06 on an unchanged function). The uniform-decay control destroying the model stands, but only as the trivial statement that scaling all weights by 0.6 is harmful. We also report a familiarity probe (set-level contamination detection AUC 0.93, single-window weak) and three negative results. We argue the productive import of the fly for language models is not sparse representation alone but the management of *when to write, what to keep, and when to erase*.

## 1. Background

The Drosophila mushroom body achieves sparse, high-dimensional coding: an odor activates ~5% of Kenyon cells, patterns are high-dimensional and decorrelated, dopaminergic neurons gate memory writes by prediction error, memories are compartmentalized, and dedicated neurons actively erase memories. Each of these has an obvious machine-learning analogue, and each invites the same question: does it help a language model?

Prior sparse-activation work in ML (MoE, sparse attention) sparsifies computation; the fly's sparsification is of *representation*. This project keeps a dense transformer compute path and sparsifies the channel activation pattern after attention, measuring the effect on language modeling, calibration, continual learning, and training stability. We are careful throughout to distinguish channel-sparse *activation* from token-sparse *attention* — nothing here changes which tokens attend to which.

## 2. Setup

Char-level GPT, 92.6M parameters (d=768, 12 layers, 12 heads, SwiGLU ffn=2048), RoPE, RMSNorm, tied embeddings; corpus: Chinese literary text (~0.9B tokens training stream, `data_v2`); AdamW + OneCycle; batch 24 × 256 tokens; 12k-step main protocol, 24k-step extension. k-WTA is applied to the attention output of every block: keep top-k channels by value (exact), by an online-adaptive threshold (CUDA kernel), or by an energy criterion. Three arms by default: `std` (dense), `flynetS` (exact top-k, k=10% unless swept), `flynetS_adaptive`.

Pre-registered judgment criteria (fixed before the 12k run, in NOTES_RLCD.md): sample efficiency at 2k, representation health (effective rank), collapse warning (distinct-3), and calibration (ECE/top-1 on next-char confidence).

## 3. Results

### 3.1 Calibration looked like a free lunch, and wasn't

At 12k steps both k-WTA arms were dramatically better calibrated than dense: top-1 next-char accuracy 0.31–0.33 vs 0.08–0.10, ECE 0.042–0.049 vs 0.094, overconfidence gap ≈ 0 vs +0.09 (3 evaluation seeds × 400 windows, consistent across seeds). At 24k the dense model caught up on both accuracy (0.365 vs 0.368) and calibration. The surviving claim is *convergence acceleration*: k-WTA reaches the "accurate and calibrated" state with roughly half the training. We consider this correction the most important methodological output of the project: a single-checkpoint advantage over an undertrained baseline is the calibration analogue of testing on the training set.

### 3.2 The sparsity gain is U-shaped, with a sweet spot near 25%

Sweeping only the keep fraction (12k steps, everything else fixed):

| keep | 2% | 10% | 15% | **25%** | 40% | 60% | dense |
|---|---|---|---|---|---|---|---|
| val loss | 4.156 | 3.889 | 3.860 | **3.827** | 3.850 | 3.899 | 3.906 |

Moderate sparsity beats dense everywhere we measured; the optimum is near 25% (≈192 of 768 channels per block), not the 5–10% suggested by biology alone. Extending the 25% arm to 24k keeps it ahead (3.635 vs 3.673 dense and 3.661 @10%), with the lead narrowing from 0.079 to 0.038 nats — part convergence speedup, part persistent edge. Distinct-3 and effective rank stay healthy in all arms; sparse arms run slightly slower in wall-clock (kthvalue overhead; the "savings" are in effective computation, not FLOPs).

### 3.3 Memory management, not sparse representation, prevents forgetting

Sequential fine-tuning on four domains (news → dialogue → law → technology, 1500 steps each, from the dense 24k checkpoint):

| arm | mechanism | avg forgetting ↓ | avg improvement over base |
|---|---|---|---|
| plain fine-tune | none | 0.692 | **−0.170** (3 of 4 domains worse than never fine-tuning) |
| compartments only | random 30% weight mask per domain | 0.380 | +0.451 |
| **fly (gate+compartments)** | surprise gate: update only when batch loss > μ+0.25σ (EMA) | **0.143** | **+0.735** (best on all four domains) |

A control isolating the representation: running *plain* fine-tuning on a k-WTA-trained trunk gives forgetting of 0.674 — statistically indistinguishable from dense. A sharper control isolating the mechanism: a *random-skip* arm (same 30% compartment mask, same ~70% of batches skipped by coin flip instead of by surprise) matches the gate almost exactly (forgetting 0.130 vs 0.143, improvement +0.745 vs +0.735). The anti-forgetting effect is update *throttling* plus parameter compartmentalization — not selective, surprise-timed writing. Sparse activation does not reduce interference at all.

### 3.4 Active forgetting is a feature, if it is targeted

RETRACTED after fixed-window re-verification (`shower_verify.py`): on deterministic validation windows the shower's effect is exactly zero across four base checkpoints (max |Δ| 0.002), while the original random 24-batch protocol swings ±0.06 on an unchanged function — the reported 3.651→3.588 recovery was sampling noise. Matched uniform decay driving val to 6.60 stands, but only as the trivial statement that scaling all weights by 0.6 is harmful. No active-erasure benefit is currently demonstrated in this setup.

### 3.5 Familiarity: the model knows what it has seen, set-level

A tiny probe on the trained model separates training-stream windows from held-out windows with AUC 0.93 at the set level (~50+ windows suffice to answer "was this evaluation set contaminated?"), while single windows are weak (AUC ≈ 0.59; recall at 5% contamination ≈ 2× random). The memorization signature in loss space is real (train windows NLL 3.53 vs val 3.72) but individually faint at this scale. Notably, sparsification does *not* reduce memorization (signature slightly stronger in the k-WTA model).

### 3.6 Negative results (reported in full)

- **k-WTA probe on frozen features loses**: on frozen Qwen3-0.6B representations, a k-WTA probe (0.570) underperforms a linear probe (0.642) on 8-way domain routing. The fly mechanisms must be present during representation formation; post-hoc sparsification does nothing good.
- **Sparse per se ≠ less forgetting** (§3.3 control).
- **"Free calibration gain" was an undertraining artifact** (§3.1 correction).

## 4. What we think is going on

The three positive results share one shape: the fly mechanisms that help are the ones that *manage* a dense computation — gating writes, erasing selectively, spending activation budget where it pays (the U-shaped sweep's optimum away from both extremes) — rather than the ones that merely change the representation's statistics. A dense transformer trained normally already learns good representations; what it lacks is an institutional memory policy. The fly, facing the same problem with 100k neurons, evolved the policy first.

## 5. Limitations

Single seed for the sweep and scale experiments (multi-seed replication of the 25% point is in progress); 92.6M–225M scale only, char-level, one language and corpus; wall-clock not yet improved (channel sparsity here costs a sort/threshold per block); continual-learning protocol uses one domain order; and "convergence acceleration" claims rest on 12k-vs-24k checkpoints rather than fitted learning curves.

## 6. Reproducing

```bash
python train_v2.py --arm std --steps 24000
python train_v2.py --arm flynetS --kfrac 0.25 --tag _k25 --steps 12000
python cl_experiment.py fly 1500 std          # gated CL arm
python fam_nll.py                              # familiarity / contamination
python forgetting_shower.py                    # active-erasure shower
```

Reports: REPORT_V2.md (three-arm study + sweep), REPORT_MEM.md (memory trilogy), NOTES_RLCD.md (methodology log, pre-registration, internal codenames).

## 7. Addendum (2026-09-20)

Four follow-ups strengthen and bound the claims: (a) the 25% sweet spot survives 4x the
training data (48k steps: k-WTA 3.525 vs dense 3.546); (b) a third scale point (334M)
is again won by k-WTA (+0.075 nats), showing the 216M inversion is an unstable
data-starvation artifact rather than a monotone scale trend; (c) the surprise-gate
threshold is a flat, tuning-free knob (forgetting 0.125-0.152 across K in [0.1, 0.5]);
(d) k-WTA codes work as retrieval addresses - Hamming lookup beats frequency and random
baselines by 2.5-4.8x - though not better than dense cosine, consistent with
"index, not representation".
