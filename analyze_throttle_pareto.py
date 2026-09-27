"""Throttle Pareto analysis: update-rate vs forgetting (2026-09-27 errata work).

Produces the figure that replaces the README's single-row comparison claim:
  x = realized skip rate (from each curve's gate_pass column)
  y = forgetting magnitude |avg_forgetting|  (lower is better)
  overlay = the gated arm's realized operating point at the same skip rate

Reading (from the committed artifacts):
  skip  9.5%  -> |F| 0.339
  skip 47.1%  -> |F| 0.190
  skip 68.1%  -> |F| 0.130   <- matched-rate point
  skip 83.7%  -> |F| 0.0925
  fly  68.0%  -> |F| 0.141   <- gate at the matched rate (worse than dice)

So "the gate's benefit is throttling" survives, and the honest statement is
stronger than the README wording: at MATCHED realized skip rate the random
skip control is *no worse* (slightly better) than the surprise gate, i.e. the
gate's selection carries no measurable advantage.

Run:  python analyze_throttle_pareto.py
Out:  figs/throttle_pareto.png
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, "logs_v2")
OUT = os.path.join(ROOT, "figs")


def realized_skip_rate(curve_path):
    if not os.path.exists(curve_path):
        return None
    rates = []
    with open(curve_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "gate_pass" in r:
                rates.append(1.0 - r["gate_pass"])
    return sum(rates) / len(rates) if rates else None


def collect(arm_filter):
    pts = []
    for f in sorted(glob.glob(os.path.join(LOGS, "cl_*_result.json"))):
        base = f[: -len("_result.json")]
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if not arm_filter(d):
            continue
        skip = realized_skip_rate(base + "_curve.jsonl")
        if skip is None:
            continue
        pts.append({
            "file": os.path.basename(base),
            "arm": d.get("arm"), "gate_k": d.get("gate_k"), "seed": d.get("seed"),
            "skip": skip,
            "forget": abs(d.get("avg_forgetting") or 0.0),
            "improve": d.get("avg_improvement") or 0.0,
        })
    return pts


skip_pts = collect(lambda d: d.get("arm") == "skip")
fly_pts = collect(lambda d: d.get("arm") == "fly" and d.get("gate_k") == 0.25)
ft_pts = collect(lambda d: d.get("arm") == "ft")

# dedupe skip points by realized rate (keep the s8 dose-response family + s7 default)
skip_pts = [p for p in skip_pts if not p["file"].endswith("g0.85")]  # keep 4 main doses
skip_pts.sort(key=lambda p: p["skip"])

print("skip dose-response (x=realized skip, y=forgetting magnitude):")
for p in skip_pts:
    print(f"  {p['file']:34s} gate_k={p['gate_k']} skip={p['skip']:.3f} |F|={p['forget']:.4f}")
print("gated arm (matched-rate overlay):")
for p in fly_pts:
    print(f"  {p['file']:34s} skip={p['skip']:.3f} |F|={p['forget']:.4f}")

# matched-rate comparison at the g0.7 dose (both ~68%)
fly68 = min(fly_pts, key=lambda p: abs(p["skip"] - 0.68))
skip68 = min(skip_pts, key=lambda p: abs(p["skip"] - 0.68))
gap = fly68["forget"] - skip68["forget"]
print(f"\nMATCHED-RATE VERDICT at skip~{skip68['skip']:.2f}: "
      f"skip |F|={skip68['forget']:.4f} vs gate |F|={fly68['forget']:.4f} "
      f"-> gate is {'WORSE' if gap > 0 else 'BETTER'} by {abs(gap):.4f}")

os.makedirs(OUT, exist_ok=True)
fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
ax.plot([p["skip"] * 100 for p in skip_pts], [p["forget"] for p in skip_pts],
        "o-", color="#3b6ea5", label="random skip (dice), dose-response")
for p in fly_pts:
    ax.scatter([p["skip"] * 100], [p["forget"]], marker="*", s=180,
               color="#c8442c", zorder=5,
               label="surprise gate (realized op. point)")
for p in ft_pts:
    if p["skip"] < 0.05:  # dense fine-tune baseline, skip ~0
        ax.scatter([2.0], [p["forget"]], marker="s", s=60, color="#555",
                   label="plain fine-tune")
        ax.annotate(f"fine-tune |F|={p['forget']:.3f}", (2.0, p["forget"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=8)
for p in skip_pts:
    ax.annotate(f"k={p['gate_k']:g}\n|F|={p['forget']:.3f}",
                (p["skip"] * 100, p["forget"]), textcoords="offset points",
                xytext=(4, -14), fontsize=7.5, color="#3b6ea5")
for p in fly_pts:
    ax.annotate(f"gate\n|F|={p['forget']:.3f}", (p["skip"] * 100, p["forget"]),
                textcoords="offset points", xytext=(6, 6), fontsize=7.5,
                color="#c8442c")
ax.set_xlabel("realized skip rate  (% of updates throttled)")
ax.set_ylabel("forgetting magnitude |avg_forgetting|  (lower = better)")
ax.set_title("Update throttling is the active ingredient: the gate sits ON the dice curve\n"
             f"(matched-rate check at {skip68['skip']*100:.0f}%: gate {fly68['forget']:.3f} "
             f"vs dice {skip68['forget']:.3f})", fontsize=9.5)
ax.grid(alpha=0.25)
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
out = os.path.join(OUT, "throttle_pareto.png")
fig.savefig(out)
print(f"\nfigure -> {out}")
