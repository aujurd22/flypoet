# flypoet

果蝇的蘑菇体靠"赢家通吃"（k-WTA）做稀疏编码：一个刺激只点亮一小撮 Kenyon 细胞，其余全部沉默。这个仓库把同一套机制放进小型语言模型，看它对训练到底有什么影响。

## 做了什么

92.6M 参数的 char-level GPT（RoPE / RMSNorm / SwiGLU / SDPA），中文文学语料，三臂对照：

- `std`：标准模型
- `flynetS`：每层激活做精确 top-k 稀疏化（只保留约 10% 通道）
- `flynetS_adaptive`：同上，但阈值由 CUDA kernel 在线自适应（Krotov–Hopfield 式阈值学习）

三臂各训 12000 步，再延长到 24000 步复核。注意这是通道稀疏激活，不是 token 稀疏注意力——稀疏在特征维，不在序列维。

## 结果（重要的部分是修正）

12k 步时 k-WTA 两臂的校准优势很显眼：top-1 准确率约为 std 的 3.7 倍，ECE 约减半，且 std 表现出明显的过度自信。当时差点写成"免费的校准增益"。

延长到 24k 后 std 追平了。所以站得住的结论是：**k-WTA 加速收敛**——它让模型提前一半训练量达到"预测准 + 自知之明"的状态，但等数据量下不改变终点。PPL 三臂基本持平（终点 3.660 / 3.661 vs 3.673，差距 0.35%）。

一个负结果：把 k-WTA 探针套在冻结的 Qwen3-0.6B 特征上，反而不如线性探针（0.570 vs 0.642）。这个机制必须从头长在训练里，事后贴上去没用。

完整数据、预设判定标准和踩过的坑见 [REPORT_V2.md](REPORT_V2.md)，方法论笔记见 [NOTES_RLCD.md](NOTES_RLCD.md)（文中 T59e、RLCD 等为内部实验代号）。

## 复现

```bash
python train_v2.py --arm std --steps 24000
python train_v2.py --arm flynetS --steps 24000
# adaptive 臂需要 MSVC 环境 + TORCH_CUDA_ARCH_LIST=8.9，见 rerun_adaptive24k.bat

python calibration_eval.py --arm std --model logs_v2/std_model.pt --seed 0
```

脚本里的路径是按 Windows 写死的（`D:\user\flypoet`），换机器需要改。模型权重不入库（单个 350MB），训练曲线和评估 JSON 都在 `logs_v2/`。
