"""Generate REPORT_V2.md from the three-arm logs, per the locked criteria
(NOTES_RLCD.md: P-样本效率 / P-表征 / P-塌缩警戒 / A-收敛等价)."""
import json, os, time

LOGS = r"D:\user\flypoet\logs_v2"

def load_arm(arm):
    d = {"curve": [], "probes": [], "final": None}
    cp = os.path.join(LOGS, f"{arm}_curve.jsonl")
    if os.path.exists(cp):
        for line in open(cp, encoding="utf-8"):
            try:
                d["curve"].append(json.loads(line))
            except Exception:
                pass
    pp = os.path.join(LOGS, f"{arm}_probes.jsonl")
    if os.path.exists(pp):
        for line in open(pp, encoding="utf-8"):
            try:
                rec = json.loads(line)
                if rec.get("step", 0) >= 2000:   # 排除冒烟残留 (step<2000)
                    d["probes"].append(rec)
            except Exception:
                pass
    fp = os.path.join(LOGS, f"{arm}_final.json")
    if os.path.exists(fp):
        d["final"] = json.load(open(fp, encoding="utf-8"))
    return d

def val_at(arm, step, d):
    """Latest val at or before `step` (val lives in probes.jsonl)."""
    best, best_step = None, -1
    for rec in d["probes"]:
        if rec["step"] <= step and rec["step"] > best_step:
            best, best_step = rec["val"], rec["step"]
    return best, best_step

std = load_arm("std")
fly = load_arm("flynetS")
ada = load_arm("flynetS_adaptive")

lines = ["# FlyPoet v2 A/B 报告（三臂，预设判定标准版）", ""]

# ---- val trajectory table
lines.append("## val loss 轨迹（匹配点对照）\n")
lines.append("| step | std | flynetS（精确top-k） | flynetS_adaptive |")
lines.append("|---|---|---|---|")
for step in (2000, 4000, 6000, 8000, 10000, 12000):
    row = [str(step)]
    for arm, d in (("std", std), ("flynetS", fly), ("adaptive", ada)):
        v, at = val_at(arm, step, d)
        row.append(f"{v:.3f} (@{at})" if v is not None else "—")
    lines.append("| " + " | ".join(row) + " |")
lines.append("")

# ---- P-样本效率
try:
    v_std_2k, _ = val_at("std", 2000, std)
    v_fly_2k, _ = val_at("flynetS", 2000, fly)
    v_std_4k, _ = val_at("std", 4000, std)
    v_fly_4k, _ = val_at("flynetS", 4000, fly)
    if v_std_2k and v_fly_2k and v_std_4k and v_fly_4k:
        p1 = (v_fly_2k < v_std_2k) and (v_fly_4k < v_std_4k)
        p1_weak = (v_fly_4k < v_std_4k) ^ (v_fly_2k < v_std_2k)
        verdict = "强复现" if p1 else ("弱复现（单点）" if p1_weak else "复现失败")
        lines.append(f"## P-样本效率判定：{verdict}\n")
        lines.append(f"- flynetS@2k {v_fly_2k:.3f} vs std@2k {v_std_2k:.3f}")
        lines.append(f"- flynetS@4k {v_fly_4k:.3f} vs std@4k {v_std_4k:.3f}")
    else:
        lines.append("## P-样本效率判定：数据不足")
except Exception as e:
    lines.append(f"P 判定异常: {e}")
lines.append("")

# ---- P-表征 (erank stability)
lines.append("## P-表征（激活有效秩 erank）\n")
lines.append("| arm | erank 序列 | 末值 | 波动(std/mean) |")
lines.append("|---|---|---|---|")
import numpy as np
stats = {}
for arm, d in (("std", std), ("flynetS", fly), ("adaptive", ada)):
    ers = [p["erank"] for p in d["probes"]]
    if ers:
        arr = np.array(ers)
        stab = arr.std() / max(arr.mean(), 1)
        stats[arm] = (arr.mean(), arr.std(), arr[-1])
        lines.append(f"| {arm} | {', '.join(f'{x:.0f}' for x in ers[:6])}... | {arr[-1]:.0f} | {stab:.3f} |")
lines.append("")
if "std" in stats and "flynetS" in stats:
    var_ratio = (stats["std"][1] / max(stats["std"][0], 1)) / max(stats["flynetS"][1] / max(stats["flynetS"][0], 1), 1e-9)
    lines.append(f"- 变异系数比 std/flynetS = {var_ratio:.2f}（>1.5 判定 std 表征波动显著更大）")
lines.append("")

# ---- P-塌缩警戒 (distinct3)
lines.append("## P-塌缩警戒（生成 3-gram 多样性）\n")
lines.append("| arm | 最终 distinct3 | 判定 |")
lines.append("|---|---|---|")
for arm, d in (("std", std), ("flynetS", fly), ("adaptive", ada)):
    d3s = [p.get("distinct3") for p in d["probes"] if p.get("distinct3") is not None]
    if d3s:
        verdict = "无塌缩" if d3s[-1] >= 0.7 else ("边缘" if d3s[-1] >= 0.5 else "⚠️ 塌缩")
        lines.append(f"| {arm} | {d3s[-1]:.3f} | {verdict} |")
lines.append("")

# ---- 生成样例
lines.append("## 生成样例对照（各臂最终 probe，prompt='春'）\n")
for arm, d in (("std", std), ("flynetS", fly), ("adaptive", ada)):
    if d["probes"]:
        lines.append(f"**{arm}**（val {d['probes'][-1].get('val','?')}）：")
        lines.append("```")
        for sp in d.get("samples_text", []):
            pass
        lines.append("```")
lines.append("（样例全文见 logs_v2/*_samples.txt）")
lines.append("")

report = "\n".join(lines)
out = r"D:\user\flypoet\REPORT_V2.md"
open(out, "w", encoding="utf-8").write(report)
print(f"REPORT_V2 written: {out}")
print(report[:500])
