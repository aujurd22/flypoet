# FlyPoet v2 三臂对照报告（终版 2026-09-19）

92.6M 参数、RoPE+RMSNorm+SwiGLU+SDPA，中文文学语料，三臂对照：
std（标准 GPT）/ flynetS（精确 top-k k-WTA）/ flynetS_adaptive（CUDA 自适应阈值 k-WTA）。

**TL;DR（24k 延长训练后的修正结论）**：k-WTA 稀疏门控的收益是**收敛加速**，不是收敛后的免费午餐。
训练中期（12k）：top-1 acc 0.09→0.33、ECE 0.094→0.042–0.049、过度自信 gap +0.09→≈0，
PPL 与 std 持平。延长到 24k 后 std 追平 acc（0.365 vs 0.368）与校准（ECE 0.056 vs 0.046/0.065），
k-WTA 的校准优势缩到 ~0.01；PPL 上两臂 k-WTA 自 12k 起全程略优（终点 3.660/3.661 vs 3.673）。
表征无塌缩（erank 稳定、distinct3 正常）。"免费校准增益"的早期说法被 24k 数据证伪并修正。

---

## 一、24k 延长训练（最终 checkpoint，OneCycle 全程）

| step | std val | flynetS val | adaptive val |
|---|---|---|---|
| 4000 | **4.3290** | 4.3623 | 4.3589 |
| 8000 | 4.1102 | 4.1137 | 4.1119 |
| 12000 | 3.8986 | 3.8870 | 3.8869 |
| 16000 | 3.7182 | 3.7089 | **3.6999** |
| 20000 | 3.6100 | 3.5924 | **3.5913** |
| 24000 | 3.6731 | 3.6613 | **3.6597** |

- 两个 k-WTA 臂自 12k 起每个匹配点 val 均 ≤ std；终点排序 adaptive 3.660 ≤ flynetS 3.661 < std 3.673。
- 三臂在 24k 同步小幅回升（如 std 3.610→3.673）：OneCycle 末段共有的过拟合，非机制差异。
- 24k distinct3：std 0.756 / flynetS 0.789 / adaptive 0.751，无塌缩；erank 末值 80.7 / 84.2 / 82.5，表征健康。
- PPL 上的差距（~0.013，0.35%）量级小，三臂本质仍是"PPL 中性"。

## 二、12k 主对照（三臂同窗）

| step | std | flynetS（精确top-k） | flynetS_adaptive |
|---|---|---|---|
| 2000 | 4.777 | **4.731** | 4.723 |
| 4000 | **4.348** | 4.359 | 4.310 |
| 8000 | **3.983** | 4.074 | 4.096 |
| 12000 | 3.906 | **3.889** | 3.911 |

P-样本效率：flynetS@2k 领先（4.731 vs 4.777），@4k 起反超消失——**弱复现，单点，不下结论**。
12k val 终值三臂差距 <0.7%，**PPL 中性成立**。

## 三、P-校准（评估 seed × 400 窗口，同协议）

**12k checkpoint（训练中期，3 seed）**

| 臂 | top-1 acc（seed 范围） | ECE（均值） | conf gap（均值） |
|---|---|---|---|
| std | 0.080–0.095 | 0.094 | **+0.093（过度自信）** |
| flynetS | 0.312–0.345 | 0.049 | ≈0 |
| flynetS_adaptive | 0.302–0.330 | **0.042** | ≈0（best seed −0.001） |

**24k checkpoint（收敛区，seed 0）**

| 臂 | top-1 acc | ECE | conf gap |
|---|---|---|---|
| std | 0.365 | 0.056 | +0.016 |
| flynetS | 0.368 | 0.065 | +0.026 |
| flynetS_adaptive | 0.368 | **0.046** | +0.031 |

**解读（修正）**：12k 处 k-WTA 臂校准大幅领先；24k 处 std 追平 acc，ECE 差距从 0.05+ 缩到 ~0.01
（adaptive 仍最低但幅度小）——校准增益主要是**收敛加速**（k-WTA 12k 达成校准，std 需 24k），
不是大的稳态差异。与 P-样本效率信号同向：k-WTA 的作用是把"高准确+校准"状态提前，等数据量下终点基本不变。

## 四、判定标准逐条结论（按 NOTES_RLCD.md 锁定）

| # | 标准 | 结论 |
|---|---|---|
| P-样本效率 | flynetS@2k 领先、@4k 消失 | 弱复现（单点） |
| P-表征 | erank 全程平稳、CV 比 0.93 | 通过（无机制性波动放大） |
| P-塌缩 | distinct3 0.74–0.81 | 通过（无塌缩） |
| P-校准 | 12k：ECE 减半 + top-1 ×3.7；24k：std 追平 | **修正为收敛加速效应**（12k 达成 vs 24k 达成） |

## 五、决策头温度缩放（decide_head/temperature_scaling.py）

对存续的 decider checkpoint（含 trunk 全量权重）做 Guo et al. 2017 单参数温度缩放，
val 对半分为 calib/test，LBFGS 拟合 T：

| 配置 | acc | ECE before | ECE after | T |
|---|---|---|---|---|
| 完整模型（9 类含弃权 argmax） | 0.251 | 0.209 | **0.014** | 2.622 |
| 消融：同一权重关闭 k-WTA mask 推理 | 0.465 | 0.077 | 0.023 | 1.359 |

- 温度缩放把 ECE 压低一个数量级（0.209→0.014），acc 不动（符合预期，T 不改 argmax）。
- 两个数字均**不可与 decider_result 的 57.5% 直接比较**：后者是训练中按非弃权口径的随机子采样评估。
- **事故**：`best_decider.pt` 为共享文件名，21:01 一次未留记录的重训覆盖了 57.5% 的好 head。
  已修复：checkpoint 按 trunk 命名（`best_decider_{trunk}.pt`），temperature_scaling 的 argmax 限定 8 真实类。

## 六、0.6B 底座路由（Qwen3-0.6B-Base 冻结特征）

- 数据：CWT2 精选 8 域均衡（train 47763 / val 5308）。
- A/B/C 实验（decide_head/qwen06_experiment.py）：zero-shot 0.139 / 线性探针 **0.642** / k-WTA 探针 0.570。
  冻结底座上 k-WTA 探针**不优于**线性探针——稀疏收益依赖全参数训练，不是特征级免费午餐。
- 自研 92.6M trunk + 决策头：flynetS_adaptive trunk acc(非弃权) **0.575** vs std trunk **0.390**（+18.5pp，
  decider_result_*.json，12k-era trunk）。

## 七、方法定位

- 本项目 k-WTA = **通道稀疏激活**，区别于 NSA/DSA 的 token 稀疏注意力（DeepSeek 2025 活跃前沿）。
- 最近参照：SPDF（Thangarasa 2023，稀疏预训练+稠密微调）；通道稀疏 + 校准的对照文献稀缺，本实验有增量。
- 神经科学动机：蕈状体 KC winnow / Krotov-Hopfield 胜者全取；自适应阈值即其在线阈值学习版本
  （CUDA kernel 21× 快于 torch.topk，见 adaptive_kwta.py 基准）。

## 八、事故与修复

| 事故 | 根因 | 修复 |
|---|---|---|
| decider 好 head 被覆盖 | 共享文件名 best_decider.pt | 按 trunk 命名 |
| 多 seed 校准不生效 | calibration_eval rng 硬编码 seed=0 | seed 参数贯通 |
| adaptive 臂无法导入 | TORCH_CUDA_ARCH_LIST 未设 / 懒编译 | 模块级设置 + vcvars bat |
| KWTA bf16 崩溃 | kernel 需 fp32 | KWTA.forward 内上转 fp32 |
| adaptive 反向 mask 失真 | y!=0 重构 mask | 显式 bool mask 保存/恢复 |
| 报告被生成脚本覆盖 | make_report_v2.py 全量覆写 | 本文件改为手工维护 |

## 九、稀疏度扫描（k_frac sweep，12k 步）

固定其余超参，只扫 k-WTA 保留比例（k = int(768·k_frac) 通道/层）。单 seed，12k 终点 val：

| k_frac | 活跃通道/层 | val@12k | vs dense | 备注 |
|---|---|---|---|---|
| 0.02 | 15 | 4.156 | +0.250 | 过稀伤学习（4k 时已落后 0.30） |
| 0.10 | 77 | 3.889 | −0.017 | 原始设置 |
| **0.25** | **192** | **3.827** | **−0.079** | **全臂最优，含 dense** |
| 1.00 (dense) | 768 | 3.906 | — | std |

**发现**：甜点不在当初拍脑袋的 10%，而在 **25% 附近**——适度稀疏（每层关 3/4 通道）
优于 dense 约 0.08 nats，且 4k 起每个检查点都领先；过稀（2%）则明显受损。
"存在中间甜区"本身比具体数值更重要：k-WTA 的收益是 U 形的，不是单调的。
注意事项：单 seed、仅 12k、仅本架构；稀疏臂 tps 略低于 dense（37.5–36.7k vs 40.4k，
kthvalue 开销），PPL 优势尚未折算成 wall-clock 优势。

## 十、产物清单

训练：train_v2.py / adaptive_kwta.py / rerun_adaptive24k.bat / overnight_runner.py
评估：calibration_eval.py / multi_seed_calib.py / compare_calibration.py
决策头：decide_head/{build_data,train_decider,temperature_scaling,qwen06_experiment}.py
数据：logs_v2/*_{curve,probes,samples}.* / logs_v2/multi_seed_*.json / decide_data/*
记录：NOTES_RLCD.md（方法论与全部负结果）/ 本报告

样例全文见 logs_v2/*_samples.txt（12k-era checkpoint 生成）。
