@echo off
call "D:\vs-install\vs\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "TORCH_CUDA_ARCH_LIST=8.9"
cd /d D:\user\flypoet
"C:\Users\user\AppData\Local\Programs\Python\Python313\python.exe" -X utf8 adaptive_kwta.py
