# flypoet

The fruit fly mushroom body does sparse coding with a winner-take-all rule (k-WTA): a stimulus lights up a small subset of Kenyon cells and silences the rest. This repo puts the same mechanism into a small language model and measures what it actually does to training.

## Setup

A 92.6M-parameter char-level GPT (RoPE / RMSNorm / SwiGLU / SDPA) trained on Chinese literary text, three arms:

- `std` — vanilla model
- `flynetS` — exact top-k sparsification of activations in every block (~10% of channels kept)
- `flynetS_adaptive` — same, but the threshold adapts online inside a CUDA kernel (Krotov–Hopfield-style threshold learning)

Each arm trains for 12,000 steps, then everything is re-run to 24,000 steps as a check. Note this is channel-wise sparse *activation*, not token-sparse attention — the sparsity lives in the feature dimension, not the sequence.

## Results (the important part is a correction)

At 12k steps the k-WTA arms looked great on calibration: top-1 accuracy was ~3.7× std, ECE roughly halved, and std was clearly overconfident. It was tempting to call this a "free calibration gain."

At 24k steps std caught up. The claim that survives: **k-WTA speeds up convergence** — the model reaches the "accurate and knows it" state with about half the training, but the endpoint doesn't move at equal data. PPL is essentially neutral across arms (3.660 / 3.661 vs 3.673, a 0.35% gap).

One negative result: a k-WTA probe on frozen Qwen3-0.6B features does *worse* than a linear probe (0.570 vs 0.642). The mechanism has to be grown into training from the start; bolting it on afterwards doesn't help.

Full numbers, pre-registered judgment criteria, and the mistakes made along the way are in [REPORT_V2.md](REPORT_V2.md). Methodology notes are in [NOTES_RLCD.md](NOTES_RLCD.md) (T59e, RLCD, etc. are internal experiment codenames).

## Reproducing

```bash
python train_v2.py --arm std --steps 24000
python train_v2.py --arm flynetS --steps 24000
# sparsity sweep: --kfrac 0.02 --tag _k02 (any keep fraction; outputs get the tag)
# the adaptive arm needs MSVC + TORCH_CUDA_ARCH_LIST=8.9 (kernel compiles lazily,
# cached builds are reused automatically)

python calibration_eval.py --arm std --model logs_v2/std_model.pt --seed 0
```

`train_v2.py` and `cl_experiment.py` resolve all paths relative to the repo root, so they run as-is once `data_v2/` exists. A few older helper scripts still show an illustrative `D:\user\...` layout — point them at your own checkout with a directory junction or an env var. Model weights are not committed (350MB each); training curves and eval JSONs live in `logs_v2/`.

## What else is in the reports

[REPORT_MEM.md](REPORT_MEM.md) extends the same trunk with three fly memory-management mechanisms: surprise-gated continual learning (79% less forgetting than plain fine-tuning), an NLL-based familiarity probe (set-level contamination detection AUC 0.93, single-window weak), and a data-free targeted forgetting shower that undoes the overfit tail (the uniform-decay control destroys the model — targeting is the mechanism).
