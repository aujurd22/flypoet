"""Rerun the two failed arms sequentially (flynetS exact, adaptive via vcvars)."""
import subprocess, time, os

PY = r"C:\Users\user\AppData\Local\Programs\Python\Python313\python.exe"
WD = r"D:\user\flypoet"
VCVARS = r"D:\vs-install\vs\VC\Auxiliary\Build\vcvars64.bat"
PROG = os.path.join(WD, "PROGRESS.md")

def prog(msg):
    line = f"- [{time.strftime('%H:%M:%S')}] {msg}"
    with open(PROG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)

# flynetS (exact torch top-k), 12k steps
prog("**flynetS（精确 top-k）** 重训开始（12000 步）")
t0 = time.time()
r = subprocess.run([PY, "-X", "utf8", "train_v2.py", "--arm", "flynetS", "--steps", "12000"],
                   capture_output=True, text=True, encoding="utf-8", cwd=WD)
dt = (time.time() - t0) / 60
ok = r.returncode == 0 and "DONE" in (r.stdout or "")
prog(f"**flynetS** 重训: {'✅' if ok else '❌'} ({dt:.0f} 分钟)" + ("" if ok else f" err={(r.stderr or '')[-200:]}"))

# adaptive (CUDA), 6k steps, needs MSVC env -> vcvars chain
prog("**flynetS_adaptive（CUDA 自适应）** 重训开始（6000 步）")
t0 = time.time()
cmd = ("call \"" + VCVARS + "\" >nul 2>&1 && set \"TORCH_CUDA_ARCH_LIST=8.9\" && "
       f"\"{PY}\" -X utf8 train_v2.py --arm flynetS_adaptive --steps 6000")
r = subprocess.run(["cmd", "/c", cmd], capture_output=True, text=True,
                   encoding="utf-8", cwd=WD)
dt = (time.time() - t0) / 60
ok = r.returncode == 0 and "DONE" in (r.stdout or "")
prog(f"**flynetS_adaptive** 重训: {'✅' if ok else '❌'} ({dt:.0f} 分钟)" + ("" if ok else f" err={(r.stderr or '')[-200:]}"))
prog("🏁 两臂补跑结束 — 可汇总 REPORT_V2")
