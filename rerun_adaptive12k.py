import subprocess, time
r = subprocess.run(["cmd", "/c", "rerun_adaptive12k.bat"], capture_output=True,
                   text=True, encoding="utf-8", cwd=r"D:\user\flypoet")
open(r"D:\user\flypoet\ad12_result.txt", "w", encoding="utf-8").write(
    f"exit={r.returncode}\n{(r.stdout or '')[-300:]}\nERR:\n{(r.stderr or '')[-300:]}")
