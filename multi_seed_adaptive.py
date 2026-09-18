"""Adaptive arm multi-seed calibration (3 seeds x 400 windows), vcvars env."""
import json
import subprocess, os

PY = r"C:\Users\user\AppData\Local\Programs\Python\Python313\python.exe"
VC = r"D:\vs-install\vs\VC\Auxiliary\Build\vcvars64.bat"
WD = r"D:\user\flypoet"
results = []
for seed in (1, 2, 3):
    cmd = (f'call "{VC}" >nul 2>&1 && set "TORCH_CUDA_ARCH_LIST=8.9" && '
           f'"{PY}" -X utf8 calibration_eval.py --arm flynetS_adaptive '
           f'--model logs_v2/flynetS_adaptive_model.pt '
           f'--seed {seed} --n_windows 400')
    r = subprocess.run(["cmd", "/c", cmd], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=WD)
    outp = os.path.join(WD, "logs_v2", f"calibration_flynetS_adaptive_seed{seed}.json")
    if os.path.exists(outp):
        d = json.load(open(outp, encoding="utf-8"))
        results.append(d)
        print(f"adaptive seed{seed}: acc={d['accuracy']:.3f} ECE={d['ECE']:.4f} "
              f"Brier={d['Brier']:.4f} gap={d['confidence_gap']:+.3f}", flush=True)
    else:
        print(f"adaptive seed{seed} FAIL", flush=True)

import numpy as np
if results:
    acc = np.array([r["accuracy"] for r in results])
    ece = np.array([r["ECE"] for r in results])
    gap = np.array([r["confidence_gap"] for r in results])
    print(f"\n[adaptive 3-seed] acc={acc.mean():.3f}±{acc.std():.3f} | "
          f"ECE={ece.mean():.4f}±{ece.std():.4f} | gap={gap.mean():+.3f}±{gap.std():.3f}")
    json.dump(results, open(os.path.join(WD, "logs_v2", "multi_seed_adaptive.json"), "w"),
              indent=1)
