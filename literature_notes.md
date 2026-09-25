# 文献之夜笔记（2026-09-26 22:15 启动，至 05:00）

目标：为 FlyPoet 的四大发现找到学术定位、竞品边界、可借鉴方法。
组织：按五个主题块记录，每条带来源链接与"对 FlyPoet 的意义"标注。
05:00 输出：LITERATURE_SYNTHESIS.md（竞品地图 + 空白清单 + 下一步实验决策）。

## 块 1：稀疏/WTA 激活在 Transformer 中的竞品地图

（搜索中…）

### 块 1 记录（22:3x 检索）

**结论：无直接撞车。** 没有找到"从头训练、通道级 WTA、带匹配对照、小规模系统研究"的 2025-26 论文。
竞品地图（对我们 related-work 的定位）：

| 工作 | 做了什么 | 与 FlyPoet 的差异 |
|---|---|---|
| [4× FLOP Reduction (Hu et al., OpenReview)](https://openreview.net) | 从头训练稀疏网络，训练中动态调 mask | 面向 FLOP 削减，无机制对照/无码分析 |
| [N:M 细粒度结构稀疏](https://arxiv.org)（首个从头训练 N:M） | N:M 半结构化 | 结构约束不同（N:M vs top-k 通道级） |
| [Training-Free Activation Sparsity (2024)](https://arxiv.org) | 后训练激活稀疏化 | 我们是 trained-in + 机制审计 |
| DRSformer（CV） | attention 里 top-k 通道选择 | CV 任务、无对照无码分析——**最接近的机制先例** |
| [Less is MoE（2026-06）](https://arxiv.org) | 裁剪专家 | MoE 路由竞争 vs 我们通道竞争 |
| [PASS (OpenReview)](https://openreview.net) | almost-sure 稀疏目标剪头 | 统计保证思路可借 |
| 微预算稀疏 Diffusion（CVPR，1.16B/$1890） | 从头稀疏训练 | 规模路线参照 |

→ FlyPoet 的空位保持：**trained-in 通道 WTA + 匹配对照 + 码身份表征 + 跨规模**，四件套无人同时做。


## 块 2：SDM / 超维计算 / Kanerva 地址理论

**最重要发现：[Beyond LLMs, Sparse Distributed Memory, and Neuromorphics (Chuma et al., 2026-08)](https://arxiv.org/html/2604.11665v2)**
- Kanerva SDM 直系后代：用 Galois 域代数（BCH 码式 LFSR）替代 SDM 的随机投影，确定性"雪崩效应"（~50% 位翻转）产生准正交——不需要随机数
- 单次写入学习（无梯度），多数投票读出，Don't-Care 容错 + Rescue 精确匹配双模式
- 应用展示：WIKIDATA 47 万数学家的师承追溯（Field 奖得主 57 代谱系，2550 万条路径），量化了 Leibniz 前后的库恩式范式转换
- 对 LLM 的定位：LLM 有 binding problem/灾难遗忘/弱因果推理，HDC 是"可逆、可审计的推理基底"——互补而非替代

**对 FlyPoet 的意义**：
1. "码=地址"是公认研究方向（Kanerva 1988→HDC→2026 综述），我们的**独特贡献是"地址来自训练语言模型的内部激活"**——他们的地址是代数构造的，我们的是从自然语言计算中涌现的。这个对比本身就是论文的一段。
2. 他们的确定性构造 vs 我们的训练涌现——两种地址生成方式可以互相评测（Galois 地址 vs k25 码在同样地址簿上的 topic 检索对比）——潜在的低成本实验。
3. "binding problem"框架：LLM 把概念不可逆地融合进 embedding；码是二值可分离的——给 §3.5"索引非表示"一个理论语言的对接点。
- 基础文献：[Kanerva 1988 原书](https://dl.acm.org/doi/10.5555/534853)、[Extended SDM (Snaider 2011)](https://digitalcommons.memphis.edu/cgi/viewcontent.cgi?article=1059&context=ccrg_papers)、[SDM 综述 (Kanerva 1993)](https://redwood.berkeley.edu/wp-content/uploads/2020/08/KanervaP_SDMrelated_models1993.pdf)

## 块 3：随机子集 vs 幅值子集的理论位置

- **[RePr (Pratt et al., CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pratt_RePr_Improved_Training_of_Convolutional_Filters_CVPR_2019_paper.pdf)**：把"剪枝的过滤器比例"当作 Dropout 参数使用，通过重排序减少过滤器间干扰——**与我们 random-k 失败/固定子集成功的发现最近**：RePr 的核心也是"干扰的过滤器应被恢复而非丢弃"。差异：他们按相位相关性选，我们按稳定性结论。
- **Lottery Ticket Hypothesis**：幅值→可训练稀疏子网络。我们的数据细化它：**幅值选择不必要，子集稳定才必要**（fixed-random 打平 top-k）——对 LTH 的小幅修正性贡献。
- [Average Parameter Count over Pre-training (2025-01)](https://arxiv.org)：统一稀疏/Dense/MoE 缩放律的框架——我们的规模阶梯可以往这个框架里放。
- 空白确认：未找到"梯度干扰 + 随机子集激活 vs 幅值剪枝"的直接组合研究。

## 块 4：持续学习的更新节流/选择性可塑性

**竞品密度高，但我们的负结果/对照视角仍是增量**：

| 工作 | 机制 | 与 FlyPoet 的关系 |
|---|---|---|
| [MIST (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/40050/44011) | 互信息引导的稀疏调参（选小参数子集更新） | =我们的"小室"，但按 MI 选而非随机 |
| [UPGD](https://www.alphaxiv.org/abs/2404.00781) | 按效用扰动梯度（utility-based update throttling） | =我们的"门控"，但按效用评估 |
| [Look-Ahead Selective Plasticity (NeurIPS 2024)](https://neurips.cc/virtual/2024/102650) | 用新任务前几个样本决定更新哪里 | 与我们的门控（用 loss 决定是否更新）同族 |
| [Loss of Plasticity (Nature 2024, Dohare)](https://www.nature.com/articles/s41586-024-07711-7) | 深度持续学习可塑性丧失（500+ 引用） | 我们 0.5B/334M 上"门控退化直通"是同一现象的小规模观察 |
| [Does CL Equally Forget All Parameters? (ICML 2023)](https://proceedings.mlr.press/v202/zhao23n/zhao23n.pdf) | 哪些参数任务专属 vs 共享 | 我们的域序列遗忘地图可对接 |

**方法论增量（可写进论文的贡献）**：这批文献提出选择性更新方法时，**几乎都不带"匹配预算的随机跳批对照"**——我们的 random-skip 对照（0.130/+0.745 ≈ 门控）揭示了：在此类对照下，"选择"的收益可能普遍被高估。这可以定位为对整个 selective-CL 文献的方法论警示——需要文献确认是否有人做过同样对照（下一轮检索）。

## 块 3 追加：参数轴 vs 批次轴的关键区分（00:5x 检索）

**重要文献区分**：参数隔离文献（[HiDe-PET](https://arxiv.org)、[Capacity-Agnostic Parameter Isolation](https://openreview.net)、InfLoRA、O-LoRA）里的常见消融是"随机参数子空间 vs 任务专属子空间"——**文献报道随机子空间通常更差，即选择性在参数轴上是重要的**（[Awesome-Forgetting 综述](https://github.com/EnnengYang/Awesome-Forgetting-in-Deep-Learning)收录该线）。

**这迫使我们把 §十六 的裁决表述拆成两轴**：
- **批次轴（何时更新）**：随机跳批 ≈ 惊讶门控——选择性无关 ✓（我们的对照完整）
- **参数轴（更新哪里）**：我们只测了**随机**30% 小室；**任务专属参数选择（MIST 式 MI 引导）从未对照**——文献提示参数轴的选择性可能重要。

→ 措辞修正：§十六"选择性无可测增量"仅对批次轴成立；参数轴是开放对照。
→ 新实验候选（5 点后）：MIST 式 MI-选择小室 vs 随机小室，同预算对照——直接借用 MIST 的选择准则，融入我们的域序列协议。

## 块 5 追加：涌现稀疏的结构性佐证（01:0x 检索）

- [Pruning as Phase Transition (CMC)](https://www.techscience.com)：大网络训练后自然涌现强激活稀疏——自组织性质；
- PowerInfer/SparseInfer 线：训练后稳定神经元子集（activation locality）可被训练-free 预测——与我们"子集稳定即可"的裁决互证；
- [Activation Approximations Can Incur Safety Vulnerabilities (2025-02)](https://arxiv.org)：激活稀疏化引入安全漏洞的警示——若 flymemory 走位码路线需引用。

**结论：我们的"子集稳定即充分"不是孤例**——但我们的增量是：**从头训练时强加稳定子集（fixed-k）＝ 免费获得稀疏化的收益**，而文献路线是"先 dense 训练再发现稀疏涌现"。前者是更便宜的正向设计，这一对比 PAPER 里值得展开。

## 深读 1：[Lässig et al. 2023](https://arxiv.org/abs/2212.04316)——最接近的 CL 先例

**机制**：WTA 稀疏表征（前馈刺激特异 + 自上而下上下文特异双信号选择）+ 层内横向循环连接保护旧表征，DFC（bio-plausible 分层信用分配）框架。
**任务**：split-MNIST（CV）。**结果**：稀疏 + 横向循环的组合才超过标准 BP，匹敌 EWC/SI，且任务无关（不需要任务边界）。

**与 FlyPoet 的逐项对照**：
| 维度 | Lässig | FlyPoet |
|---|---|---|
| 模态/任务 | CV, split-MNIST | char-LM, 4-8 域序列 |
| 稀疏来源 | DFC 双信号选择 | 幅值 top-k（训练进模型） |
| 抗遗忘搭档 | 横向循环连接 | 参数小室 + 写入门控 |
| 基线超越 | = EWC/SI | = 随机跳批（关键差异！他们的对照没拆 throttling） |
| 码分析 | 无 | 稳定/因果/组合/主题地址五组 |

**结论**：同题不同深度——他们证明了"稀疏+横向连接在 CV 有效"，我们提供了 LM 域的匹配对照方法学 + 码身份的五组表征 + throttling 揭示。互补而非竞争，related work 必引。

## 深读 2：[MIST (AAAI 2026)](https://arxiv.org/abs/2505.19943)——参数轴选择性的关键证据

**机制**：MI-based Fisher（InfoNCE 梯度平方累积）选 top 5% 参数，运行时再随机 dropout 90%（实际每步只更新 ~0.5%），加上 MI loss 本身。RanPAC 骨干，五基准 SOTA 级增益。

**关键消融表（ImageNet-R，同 5% 预算）**：
| 选择策略 | Ā |
|---|---|
| 随机 5% | **52.4（崩溃）** |
| L2 范数 top-5% | 39.0 |
| 梯度幅值 top-5% | 54.8 |
| **MIST（MI 选择）** | **84.9** |

**两轴区分确立（本项目 + MIST 合并）**：
- **参数轴（更新哪里）**：MI 选择 ≫ 随机（52.4 vs 84.9）——**选择性重要**
- **批次轴（何时更新）**：随机跳批 ≈ 惊讶门控——**选择性不重要**
→ 干净的可发表命题：**选择性的价值集中在"哪里"，不在"何时"**。
→ 我们的门控实验升级方向自然浮现：把门控从"何时"（batch skip）搬到"哪里"
（用惊讶-梯度历史替代随机小室分配，即参数轴的 surprise 选择）——
这是 MIST（MI 选择）与 FlyPoet（惊讶门控）的真正杂交点。

## 块 5 补充：GitHub/仓库扫描结论

- 无"trained-in 通道 WTA + LM + 匹配对照"的直接仓库——flypoet 空位保持；
- 最近邻仓库/列表：[ContinualAI papers 列表](https://github.com/ContinualAI/continual-learning-papers)（343+ 篇）、SpikingLM（OpenReview）、BiRT（bio-replay ViT）、NM-Hebb（Hebb+WTA）；
- Lässig 的代码可用性未在摘要页确认（CatalyzeX 链接需跳转）。
