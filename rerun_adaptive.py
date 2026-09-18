import subprocess, time
r = subprocess.run(["cmd", "/c", "rerun_adaptive.bat"], capture_output=True,
                   text=True, encoding="utf-8", cwd=r"D:\user\flypoet")
open(r"D:\user\flypoet\ad_result.txt", "w", encoding="utf-8").write(
    f"exit={r.returncode}\n{(r.stdout or '')[-500:]}\nERR:\n{(r.stderr or '')[-500:]}")
