"""Pipeline v2: wait for c4 download -> prepare_v2 -> std 12k -> flynetS 12k -> report."""
import subprocess, sys, os, time, glob

PY = r"C:\Users\user\AppData\Local\Programs\Python\Python313\python.exe"
LOG = open(r"D:/user/flypoet/pipeline_v2.log", "a", encoding="utf-8")

def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True); LOG.write(line + "\n"); LOG.flush()

# ---- wait for c4 (96 shards) up to 2.5h
log("waiting for c4 download (96 shards)...")
for check in range(150):
    n = len(glob.glob(r"D:/user/corpus/c4/*.jsonl.zst"))
    log(f"  c4 shards: {n}/96")
    if n >= 96:
        break
    time.sleep(60)
else:
    n = len(glob.glob(r"D:/user/corpus/c4/*.jsonl.zst"))
    log(f"timeout: proceeding with {n} shards (enough for 30M chars)")

# ---- prepare
log("prepare_v2...")
r = subprocess.run([PY, "-X", "utf8", r"D:\user\flypoet\prepare_v2.py"],
                   capture_output=True, text=True, encoding="utf-8")
log((r.stdout or "")[-1200:])
if r.returncode != 0:
    log("PREPARE FAILED: " + (r.stderr or "")[-1200:]); sys.exit(1)

# ---- arms
for arm in ("std", "flynetS", "flynetS_adaptive"):
    steps = 4000 if arm == "flynetS_adaptive" else 12000   # 首轮 adaptive 臂做消融对照
    log(f"train_v2 arm={arm} ({steps} steps)")
    r = subprocess.run([PY, "-X", "utf8", r"D:\user\flypoet\train_v2.py",
                        "--arm", arm, "--steps", str(steps)],
                       capture_output=True, text=True, encoding="utf-8")
    log(f"{arm} exit={r.returncode}\n{(r.stdout or '')[-600:]}")
    if r.returncode != 0:
        log((r.stderr or "")[-1200:])

log("PIPELINE V2 DONE")
