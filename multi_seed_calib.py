"""Multi-seed calibration: std vs flynetS, 3 seeds x 400 windows each."""
import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
import subprocess, json, time

PY = r"C:\Users\user\AppData\Local\Programs\Python\Python313\python.exe"
results = {}
for arm, model in (("std", "logs_v2/std_model.pt"),
                   ("flynetS", "logs_v2/flynetS_model.pt"),
                   ("flynetS_adaptive", "logs_v2/flynetS_adaptive_model.pt")):
    results[arm] = []
    for seed in (1, 2, 3):
        r = subprocess.run([PY, "-X", "utf8", "calibration_eval.py", "--arm", arm,
                            "--model", model, "--seed", str(seed), "--n_windows", "400"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        outp = rf"D:\user\flypoet\logs_v2\calibration_{arm}_seed{seed}.json"
        try:
            d = json.load(open(outp, encoding="utf-8"))
            results[arm].append(d)
            print(f"{arm} seed{seed}: acc={d['accuracy']:.3f} ECE={d['ECE']:.4f} "
                  f"Brier={d['Brier']:.4f}", flush=True)
        except Exception as e:
            print(f"{arm} seed{seed} FAIL: {e}", flush=True)

print("\n=== 跨 seed 汇总 ===")
for arm in results:
    rs = results[arm]
    if not rs:
        continue
    import numpy as np
    acc = np.array([r["accuracy"] for r in rs])
    ece = np.array([r["ECE"] for r in rs])
    brier = np.array([r["Brier"] for r in rs])
    gap = np.array([r["confidence_gap"] for r in rs])
    print(f"[{arm}] acc={acc.mean():.3f}±{acc.std():.3f} | "
          f"ECE={ece.mean():.4f}±{ece.std():.4f} | "
          f"Brier={brier.mean():.4f}±{brier.std():.4f} | "
          f"gap={gap.mean():+.3f}±{gap.std():.3f}")
json.dump(results, open(r"D:\user\flypoet\logs_v2\multi_seed_calibration.json", "w"), indent=1)
