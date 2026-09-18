"""Overnight runner: std 12k -> flynetS 12k -> flynetS_adaptive 6k (vcvars).
Writes PROGRESS.md after each arm; DONE marker at the end."""
import subprocess, time, os

PY = r"C:\Users\user\AppData\Local\Programs\Python\Python313\python.exe"
WD = r"D:\user\flypoet"
PROG = os.path.join(WD, "PROGRESS.md")

def prog(msg):
    line = f"- [{time.strftime('%H:%M:%S')}] {msg}"
    with open(PROG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)

def run(arm, steps, vcvars=False):
    prog(f"**{arm}** 训练开始（{steps} 步）")
    t0 = time.time()
    if vcvars:
        cmd = ("call \"D:\\vs-install\\vs\\VC\\Auxiliary\\Build\\vcvars64.bat\" >nul 2>&1 && "
               f"\"{PY}\" -X utf8 train_v2.py --arm {arm} --steps {steps}")
        r = subprocess.run(["cmd", "/c", cmd], capture_output=True, text=True,
                           encoding="utf-8", cwd=WD)
    else:
        r = subprocess.run([PY, "-X", "utf8", "train_v2.py", "--arm", arm,
                            "--steps", str(steps)], capture_output=True, text=True,
                           encoding="utf-8", cwd=WD)
    dt = (time.time() - t0) / 60
    ok = (r.returncode == 0) and ("DONE" in (r.stdout or ""))
    prog(f"**{arm}** 完成: {'✅' if ok else '❌'} ({dt:.0f} 分钟)" +
         ("" if ok else f" exit={r.returncode} stderr={ (r.stderr or '')[-300:]}"))

with open(PROG, "w", encoding="utf-8") as f:
    f.write(f"# FlyPoet 过夜训练进度（启动 {time.strftime('%m-%d %H:%M')}）\n\n"
            "三臂串行：std 12k → flynetS 12k → flynetS_adaptive 6k\n\n")

run("std", 12000)
run("flynetS", 12000)
run("flynetS_adaptive", 6000, vcvars=True)

with open(os.path.join(WD, "OVERNIGHT_DONE"), "w") as f:
    f.write(time.strftime("%m-%d %H:%M"))
prog("🏁 OVERNIGHT ALL DONE — 明早汇总 REPORT")
