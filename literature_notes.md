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

## 深读 3：[UPGD (ICLR 2024)](https://arxiv.org/abs/2404.00781)——效用调制更新的完整版

**机制**：每参数维护效用度量 → 高效用参数的扰动更小（保护→抗遗忘），低效用参数扰动更大（ rejuvenate→保可塑性）。**一个效用度量同时解决遗忘与可塑性丧失两个失败**——多数先前方法只解决其一。

**实验**：数百个非平稳性的流式学习（任务边界未知），基线普遍逐任务衰退，UPGD 持续提升；PPO 强化学习同样避免 Adam 的性能下滑。ICLR 2024，代码 github.com/mohmdelsayed/upgd。

**与 FlyPoet 的关键映射**：
- 我们的门控 = UPGD 的"保护"半边（抗遗忘）；我们的 334M 观察门控退化直通 = 可塑性半边未处理；
- UPGD 启示的杂交实验：**惊讶-效用调制**——门控（何时写）+ 效用加权扰动（写哪里/怎么写）二合一，
  即 MIST（哪里）+ FlyPoet（何时）+ UPGD（怎么写）三原理的正交组合。
- 与 random-skip 裁决的关系：UPGD 的效用调制是**参数轴**的选择性——与我们"参数轴选择性未对照"
  的缺口正好互补，且它自带"保护 vs 新化"的对称设计。

## 深读 4：RePr（检索摘要，PDF 404 用搜索结果）

[RePr (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pratt_RePr_Improved_Training_of_Convolutional_Filters_CVPR_2019_paper.pdf)：两阶段训练——先剪除高干扰过滤器训练，再恢复部分被剪过滤器继续训练。"把剪枝比例当作 Dropout 参数"，核心概念是**过滤器间表示干扰（interference）**：干扰大的过滤器学到重复特征，剪除后恢复可减少重复、提升容量利用。
**与 FlyPoet random-k 失败的对接**：random-k25 每步换子集 = 每步强迫下游在全新过滤器组合上重建对应 = 干扰最大化；fixed-k = 零干扰。RePr 的"先剪后恢复"恰是我们 dose 曲线的静态版。理论对接点明确。

## 深读 5：[Bricken & Pehlevan, NeurIPS 2021](https://arxiv.org/abs/2111.05498)——"Attention ≈ SDM"理论锚

**对应关系**：Query=SDM 读出的地址线索；Key=硬位置地址向量（点积替代 Hamming 临界距离）；Value=位置上存的数据向量；**softmax 温度=SDM 的激活临界半径**（控制多少位置参与读取）。
**关键验证**：推导了等价成立的数据条件，并**在预训练 GPT-2 中确认条件满足**——真实的 attention 头确实运行在 SDM 机制区。

**对 FlyPoet 的意义（理论锚）**：我们的 k25 码 = 这个 SDM 框架里的**二值化硬地址**。
- Bricken 的 attention 是"软 SDM"（连续权重）；我们的 k-WTA 是"硬 SDM"（二值 top-k）——
  从软到硬正好是本文"码因果 1.76×"的试验田（硬化的代价与收益）；
- 我们的域检索（Hamming hit@1 2.6× chance）= 该理论在通道维度的实证；
- 引用位：PAPER §3.5 "码=地址"与 §"码非语义"两节的关键 related work。

## 蘑菇体计算模型基准（神经保真度对照，服务于 related work）

| 模型 | 年份/期刊 | 内容 | 保真度要点 |
|---|---|---|---|
| [Wang et al.](https://www.sciencedirect.com/science/article/pii/S0896627321006826) | 2021 Neuron | 用 ML（进化优化+监督）重建果蝇嗅觉回路：51 肾小球→2000 KC，PN→KC 收敛 ~50:1 | 解剖级保真（连接率、稀疏度与生物一致） |
| [Zhang & Sharpee](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2013.00141/full) | 2013 Front. Comput. Neurosci. | PN→KC 稀疏编码变换的设计原理 | 理论：维度扩张产生稀疏码 |
| [Honegger et al.](https://www.jneurosci.org) | 2011 J.Neurosci. | KC 群体成像实验基准（稀疏响应实测） | 生物基准数据 |
| Dasgupta et al.（"A fly-inspired HW solution"） | 2017 | 随机扩张重编码的计算理论 | 我们引言已引的相关理论 |
| [Babadi & Sompolinsky](https://www.cell.com/neuron/fulltext/S0896-6273(14)00370-4) | 2014 Neuron | PN→KC 维度扩张产生稀疏码的理论 | 理论基础 |

**FlyPoet 的位置**：这些模型以"解剖保真"为目标（50:1 收敛、5% 激活对齐生物实测）；
FlyPoet 反其道——**用 Transformer 的实际最优（25%）对照生物常数（5%）并证明后者次优**。
这个对比本身是一段：解剖保真不是性能保真，机制灵感的正确用法是"对照性地借用"而非"常数地照搬"。

## 深读 6：[Dasgupta, Sheehan, Stevens & Navlakha, PNAS 2017](https://www.pnas.org)——源头论文确认

**"A neural data structure for novelty detection"**：果蝇嗅觉回路 = 一种高效的局部敏感哈希（LSH）。
- 结构：~50 投影神经元 → 稀疏随机固定连接扩张 → ~2000 KC → **top 5% WTA** → 局部学习（只更新激活的 KC）
- 性能：相似性排序、最近邻、去重任务上匹敌或超过 SimHash
- 衍生谱系：Can a Fruit Fly Learn Word Embeddings? (Liang 2021)、Fly-CL、果蝇式联邦分类

**与 FlyPoet 的最终对齐**：Dasgupta 2017 的算法三件套（随机扩张+5%WTA+局部学习）
我们各有一个 LM 规模的对照答案：
1. 随机扩张 → **top-k 幅值扩张不必要**（fixed-k 打平，四轴裁决）
2. 5% WTA → **25% 最优**（九点扫描），且随规模/预算变化
3. 局部学习 → **门控=纯节流**（随机跳批对照）
我们的贡献可以定位为：**对这条"果蝇算法"谱系做了一次 LM 规模、匹配对照的系统重检**，
产出三个修正（最优稀疏度、选择规则、写入策略）+ 一个新现象（478M 晚期 crossover）。

## 深读 7：[SDMLP——"Sparse Distributed Memory is a Continual Learner"（ICLR 2023）](https://arxiv.org/abs/2303.11934)——**最近邻先例**

**作者**：Bricken, Davies, Singh, Krotov & Kreiman（MIT CBMM × IBM）。代码：github.com/trentbrick/sdmcontinuallearner
**机制**：Kanerva SDM 改造成单隐层 MLP（SDMLP），支持无 replay 的在线持续学习。
**关键数字**：**每输入约 20% 神经元激活**（稀疏、模式一致）——**与我们的 25% 甜点独立收敛**（不同架构 MLP vs Transformer、不同任务、不同团队）。
**他们的重要工程发现**：稀疏网络里**动量优化器会"陈旧动量"化**（stale momentum）→ 灾难遗忘；修复是训练方案的一部分——与我们的 fixed/random-k 对照（子集稳定性）同族的"稀疏网络特有优化病理"。
**稀疏的双重角色**：同一稀疏激活机制既影响 CL 又影响 NCL——CL 收益来自与 SDM 容量/精度权衡共享的设计，而非单独的 CL 机制。

**对 FlyPoet 的意义**：
1. **20%↔25% 独立收敛 = U 形甜点的跨架构佐证**（PAPER 可引：独立团队在 MLP 上得到相近的最优稀疏度）；
2. 我们的差异化：Transformer 主干 + 匹配对照方法学 + 码身份五组表征 + 跳批剂量曲线——他们都没有；
3. 他们的 stale-momentum 发现值得在我们的训练里检查（AdamW 的动量在 k-WTA 稀疏激活下是否有同样的病理——**潜在新实验**）。

## 险些撞车核实：Fly-CL (Zou et al., ICLR 2026)

果蝇嗅觉回路（稀疏随机投影+KC 去相关）→ **预训练模型的持续表示学习**：渐进去相关、训练时间大降、性能匹敌或超 SOTA CL 方法，代码在 GitHub。
**与 FlyPoet 的边界**：他们 = 预训练 backbone + 表征去相关 + 效率；我们 = 从头训练 + 通道 WTA + 匹配对照 + 码身份。不撞车，但 related work 必引（同果蝇灵感+同持续学习方向）。正确 arXiv 号待查（此前猜号两次落空，改用标题搜索验证）。

## 块 6：Gated Attention + Diff Transformer 深度定位（20:5x-21:1x 检索）

### [Gated Attention (Qiu et al., NeurIPS 2025)](https://arxiv.org/abs/2505.06708)

- 30 种 gated-attention 变体系统比较后收敛到：`O' = σ(g(X)) ⊙ SDPA(X)`
- **gating position 消融**：门控在 **attention output（O）** 最优；直接门控 value（V）退化
- **规模效应**：0.5B-8B 的消融显示 gate 主要惠及更长训练与更大规模——**我们 92.6M 上 sigmoid≈top-k 的打平可能在小规模才成立**
- 额外收益：训练稳定（loss spike 消除）、容忍更高学习率、消除 attention sink
- **与 FlyPoet 的关系**：我们的 sigmoid 臂（3.823 ≈ top-k 3.827）独立复现了"gating 不输 hard selection"，但规模效应（gating 在更大模型上是否反超）未验证

### [Differential Transformer (Ye et al., 2024)](https://arxiv.org/abs/2410.05258)

- `A = softmax(Q₁K₁ᵀ) − λ·softmax(Q₂K₂ᵀ)`，减去两个 attention map 消除噪声
- 437+ 引用，改善长上下文、关键信息检索、ICL、幻觉缓解
- **与 FlyPoet 的镜像关系**：他们在 token-attention 维消噪声→稀疏注意模式；我们在 channel 维竞争→稀疏通道——**同一个"抑制→稀疏"原理的不同维度实现**
- **SDT（Sparse Differential Transformer）**：top-K 稀疏掩码 + 差分注意力已有人组合（人脸聚类）——但 channel 级 WTA + 差分的组合仍空白

### 三轴统一框架（评审建议 + 文献补全后）

| 轴 | 机制 | 代表 | FlyPoet 状态 |
|---|---|---|---|
| Selection（选谁传播） | Top-k / Random / Fixed / Sigmoid | FlyPoet 四臂 | ✅ 已完成：稳定子集=活性成分 |
| Inhibition（抑制噪声） | 差分注意力 | Diff Transformer | 未做 |
| Memory（外部/内部记忆） | 码簿 / Hopfield / SDM | flymemory + SDMLP | 部分完成（检索✅/训练投影❌） |
| Gating（门控权重） | σ(g)⊙O / k-WTA | Gated Attention / FlyPoet | sigmoid 臂已测，大规模未验 |

→ **缺失的四轴交叉**：gating × inhibition（σ 门 + 差分通道）在 channel 维的组合——文献空白。

## 块 7：GitHub 仓库 + k-WTA scaling laws 检索结论（01:3x）

**结论：无直接竞品仓库。** "k-WTA scaling laws for transformers" 没有找到专门论文——
FlyPoet 的规模阶梯（五规模 × 匹配对 × 剂量曲线）在这个方向保持空白。
最接近的仓库/论文：
- [WTACRS (NeurIPS)](https://github.com/zirui-ray-liu/WTACRS/)：WTA 采样用于 LM 适配
- [400 activation functions survey (Kunc & Klezl 2024)](https://arxiv.org/pdf/2402.09092)：k-WTA 激活函数收录（对抗鲁棒性用途）
- [ContinualAI papers 列表](https://github.com/ContinualAI/continual-learning-papers)：343+ CL 论文
- [Awesome-Forgetting-in-Deep-Learning](https://github.com/EnnengYang/Awesome-Forgetting-in-Deep-Learning)：子空间 CL 方法目录
- [Awesome-SNN](https://github.com/TheBrainLab/Awesome-Spiking-Neural-Networks)：脉冲神经网络稀疏方法

→ flypoet 的空位保持：**trained-in WTA + Transformer + 匹配对照 + 码身份 + 跨规模**五件套仓库不存在。

## 块 8：attention sink → 幻觉 → 内部信号（03:5x 检索，高度活跃的 2025-26 前沿）

**发现：attention sink → 激活异常 → 幻觉的链路是 2025-26 的热门方向**：

| 工作 | 关键发现 |
|---|---|
| [SinkProbe (Binkowski et al., ICLR 2026)](https://arxiv.org/html/2604.10697v1) | 幻觉溯源至 attention sink 分数捕获的内部信息流崩塌——SinkProbe = 检测方法 |
| [Anatomy of Massive Activations (Sun et al., 2026)](https://ui.adsabs.harvard.edu/abs/2026arXiv260305498S/abstract) | attention sink 局部调制跨头注意力输出、偏置头向短程依赖 |
| [Enhancing Shallow Layers (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.174.pdf) | 幻觉与图像 token 的 attention sink 模式密切关联，浅层稠密 sink 行为 |
| [Enhancing Attention Heads (arXiv 2411.09968)](https://arxiv.org/html/2411.09968v1) | **稀疏 attention sink 易产生幻觉，dense sink 头不易** |
| [Attention Sinks: Catch-Tag-Release (NeurIPS 2025)](https://neurips.cc/virtual/2025/poster/115812) | LLM 把注意力集中在少数 token（如第一个 token） |

**与 FlyPoet 的关联**：
1. **k-WTA 强制通道稀疏**——如果稀疏 attention sink 易幻觉，我们的 k25 模型是否也更易幻觉？
   这是一个可检验的预测（用真实文本对照困惑度+域检索实验间接评估）；
2. 反面：如果 k25 模型的幻觉率**不高于** dense（因为码是因果承载的稳定子集，不是 sink 式
   的信息流崩塌），那就是**稀疏结构的抗幻觉证据**——项目级正面结果；
3. SinkProbe 方法论可借鉴：用内部信号（attention scores）做检测，而非输出层面——
   与我们的码分析（用内部激活做检索/因果消融）同一方法学家族。

## 块 9：博客/从业者/最近邻生物计算（04:1x 检索）

**最有价值的发现：[Zahn et al., PLOS Comp Biol——"Pruning deep neural networks generates a sparse, bio-inspired network"](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010512)**
- 直接把 DNN 剪枝与果蝇蘑菇体的稀疏高维编码对接
- **与 FlyPoet 最近**：他们从"剪枝产生稀疏"出发，我们从"训练时强加稀疏"出发——
  同一果蝇 MB 对接的两种方法（后验涌现 vs 先验强加）
- PLOS Comp Biol 期刊（非预印本），framing 面向从业者

**其它**：
- [DevFly (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/file/0fed4ca757f63257370f456def09d3eb-Paper-Conference.pdf)：MB 的发育过程模拟（稀疏连接的生物发育→模型初始化）
- [Nolta 2026](https://epapers2.org/nolta2026/ESR/paper_details.php?paper_id=9253)：昆虫 MB 的视觉路线导航（稀疏编码+简单突触可塑性）
- [SNN 视觉导航 (Frontiers 2024)](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2024.1379977/full)：SNN 学习复杂自然视觉场景（引用 23）
- [MB gap junction sparse reward (PMC 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11152299)：间隙连接网络的稀疏奖励学习
- 博客/教程极度稀缺：唯一 [tinyML EMEA 2021 视频](https://www.youtube.com/watch?v=aHHlBFqS99Y)
  （神经形态电路），从业者级教程空缺——**flypoet 的 README 恰好可以填补这个生态位**

## 文献综述完成度评估（01:2x→03:2x 检索+2 深读+1 撤回核实）

| 主题块 | 覆盖度 | 空白确认 |
|---|---|---|
| WTA/sparse activation in LM | ✅ 充分（10+ 论文） | 无撞车 |
| SDM/HDC 地址理论 | ✅ 充分（VaCoAl 深读+经典谱系） | 交叉空白（码来自 LM 内部） |
| 随机子集 vs 幅值选择理论 | ✅ 充分（RePr/LTH/干扰理论） | 通道级空白 |
| CL 选择性可塑性 | ✅ 充分（MIST/UPGD/Lässig/Doicare） | 批次轴对照空白（我们的贡献） |
| attention sink→幻觉 | ✅ 2025-26 活跃前沿 | 稀疏训练模型幻觉率未测 |
| 博客/从业者 | ⚠️ 极度稀缺 | flypoet README 可填补 |
| GitHub 仓库 | ✅ 无竞品 | flypoet 空位保持 |
