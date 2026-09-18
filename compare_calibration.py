"""Compare calibration (ECE/Brier/familiarity strata) across the three arms.
Run after each arm's calibration_eval: reads logs_v2/calibration_*.json.
"""
import json, os, glob

LOGS = r"D:\user\flypoet\logs_v2"
rows = []
for fp in sorted(glob.glob(os.path.join(LOGS, "calibration_*.json"))):
    arm = os.path.basename(fp)[12:-5]
    d = json.load(open(fp, encoding="utf-8"))
    rows.append((arm, d))

if not rows:
    print("no calibration results found")
    raise SystemExit

print(f"{'arm':22s} {'ECE':>7s} {'MCE':>7s} {'Brier':>7s} {'acc':>6s} {'avg_conf':>9s} {'gap':>7s}")
print("-" * 72)
for arm, d in rows:
    print(f"{arm:22s} {d['ECE']:7.3f} {d['MCE']:7.3f} {d['Brier']:7.3f} "
          f"{d['accuracy']:6.3f} {d['avg_confidence']:9.3f} {d['confidence_gap']:+7.3f}")

# familiarity strata comparison (low/mid/high)
print("\n熟悉度分层 ECE（低/中/高）:")
for arm, d in rows:
    st = d.get("familiarity_strata", [])
    eces = [f"{s['ece']:.3f}" for s in st[:3]] + ["—"] * (3 - len(st[:3]))
    print(f"  {arm:22s} {eces[0]:>7s} / {eces[1]:>7s} / {eces[2]:>7s}")

# verdict hints
lines = ["\n观察:"]
by = {arm: d for arm, d in rows}
if "std" in by:
    for arm in ("flynetS", "flynetS_adaptive"):
        if arm in by:
            de = by[arm]["ECE"] - by["std"]["ECE"]
            lines.append(f"- {arm} 相对 std 的 ECE 差: {de:+.3f} "
                         f"({'fly 校准更差' if de > 0.01 else 'fly 校准更好' if de < -0.01 else '基本相当'})")
lines.append("- 熟悉度分层若在 flynetS 臂出现'低熟悉桶 ECE 更高'的结构性梯度，")
lines.append("  支持'稀疏竞争产生熟悉度感知的校准'预言；三桶平坦则无此效应。")
print("\n".join(lines))


# ---- 自动判定（按 NOTES_RLCD.md 锁定标准）
def auto_verdict(rows):
    by = {arm: d for arm, d in rows}
    out = []
    if "std" in by and "flynetS" in by:
        de = by["flynetS"]["ECE"] - by["std"]["ECE"]
        out.append(f"P-校准对照: flynetS vs std ECE 差 {de:+.3f} → "
                   + ("fly 校准更差" if de > 0.02 else "fly 校准更好" if de < -0.02 else "相当"))
    gap_rel = []
    for arm in by:
        gap_rel.append(f"{arm} 过度自信 {by[arm]['confidence_gap']:+.3f}")
    out.append("过度自信排序: " + " | ".join(gap_rel))
    return out
print("自动判定:")
for l in auto_verdict(rows):
    print(" ", l)
