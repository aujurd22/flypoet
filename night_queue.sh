#!/bin/bash
# 过夜接力：等 XL24 对（call_bbaefd37）完成后串行执行今夜三实验
LOGF="C:/Users/djr82/.zcode/cli/exec/sess_52a0f73c-f15f-41e9-ab75-0aaab05607a4/call_bbaefd37c9b542d3aa8c7c11-stdout.log"
PY="C:/Users/djr82/AppData/Local/Programs/Python/Python313/python.exe"
while ! grep -q "EXT2R_DONE" "$LOGF" 2>/dev/null; do sleep 60; done
cd /d/djr82/flypoet
echo "=== [D1] 216M-b16 seed8 dense ==="
"$PY" -X utf8 train_v2.py --arm std --d 1024 --layers 16 --heads 16 --ffn_h 2816 --batch 16 --steps 12000 --seed 8 --tag _L_s8
echo "=== [D2] 216M-b16 seed8 k25 ==="
"$PY" -X utf8 train_v2.py --arm flynetS --kfrac 0.25 --d 1024 --layers 16 --heads 16 --ffn_h 2816 --batch 16 --steps 12000 --seed 8 --tag _Lk25_s8
echo "=== [A] 0.5B k40 ==="
"$PY" -X utf8 train_v2.py --arm flynetS --kfrac 0.40 --d 1536 --layers 16 --heads 16 --ffn_h 4224 --batch 4 --steps 12000 --tag _XLb4k40
echo "=== [C] store scaling retrieval ==="
"$PY" -X utf8 domain_retrieval.py flynetS_k25_24k 5000
echo NIGHT4B_DONE
