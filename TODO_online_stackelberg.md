# 动态在线Stackelberg博弈 — 实施路线图

## 方向1：动态Sycophancy调控 ✅ 优先级最高
**状态：方案设计中**

**核心思路：** 不依赖质量信号区分正确/错误，而是监控round间stance变化率（sycophancy proxy），自适应调节mechanism参数（γ, λ_c, EV权重），防止过早共识坍缩。

**为什么可行：**
- 不依赖信号区分力（AUC=0.55的问题不影响）
- θ*已经自动发现anti-sycophancy（star/debate上EV→0.01），证明这个方向有效
- Sycophancy检测只需比较round间stance变化，计算成本低
- 文献支持：Bertalanič & Fortuna (2026) 报告同质debate中sycophancy率达85.5%

**实验设计：**
- 拓扑：debate（3 rounds，最适合动态调控）
- 三个条件对比：
  1. Baseline: 无反馈的标准debate
  2. Static: 固定θ*参数的online feedback（现有实现）
  3. Dynamic: 根据stance变化率自适应调整γ/λ_c/EV权重
- 度量：sycophancy率（stance flip ratio）、最终MajVote accuracy、diversity preservation

**关键公式：**
```
stance_change_rate = |{agents who flipped stance}| / n
if stance_change_rate > τ_high:  # 过度sycophancy
    γ ← γ * (1 - α)           # 降低peer pressure
    w_EV ← w_EV * (1 - β)     # 降低从众信号权重
elif stance_change_rate < τ_low:  # 过度stubborn
    λ_c ← λ_c * (1 + α)       # 增大consensus pressure
```

**文献对标：**
- Bertalanič & Fortuna (2026): sycophancy率85.5%，isolated self-correction优于unguided debate
- SLHF (ICLR 2026): Stackelberg框架用于alignment，leader先commit评分规则
- RAID (arXiv 2026): 在线自适应激励设计，O(√t log t) regret

---

## 方向2：基于结果的历史激励机制
**状态：待启动**

**核心思路：** 不用质量信号预测谁对，而是用agent在同类问题上的历史正确率来分配权重。在sequential rounds / 多问题场景中，追踪agent表现，给历史上表现好的agent更高consensus weight。

**对应文献：**
- Yan & Zhang (NeurIPS 2025): "Stackelberg Learning with Outcome-based Payment" — 证明payment是cooperative learning的必要条件
- 联邦学习中的贡献度评估 (Shapley value, 边际贡献)

**需要什么：**
- 多问题sequential evaluation（需要ground truth）
- 领域分类器（不同问题类别的agent表现可能不同）
- 探索-利用平衡：新问题类型需要exploration

**与方向1的关系：** 互补。方向1调节debate动态，方向2调节agent权重分配。

---

## 方向3：信念塑造与认知干预
**状态：待启动**

**核心思路：** 通过注入精心设计的反馈改变agent的"信念"（epistemic state），影响后续推理行为。不是简单地报告质量分数，而是策略性地选择注入什么信息。

**对应文献：**
- Zhu (2025): "LLM-Stackelberg Games" — 定义推理均衡和猜测推理均衡，捕捉有限理性和信念更新
- 实验发现：online feedback在MMLU-Hard debate上+5.4pp，说明信念塑造确实有效

**可能的信念塑造策略：**
- 告诉agent"你的推理与多数一致/不一致"（现有）
- 告诉agent"你的推理链有逻辑gap"（需要NLI或LLM-as-judge）
- 选择性地只向minority agent注入反馈（鼓励坚持独立观点）
- 注入"devil's advocate"指令给convergent agents

**与方向1的关系：** 方向1决定"何时干预"，方向3决定"干预什么内容"。

---

## 整体时间线

1. **方向1**（当前）→ 方案设计 → 代码实现 → 实验 → 写入论文
2. **方向2**（方向1完成后）→ 设计历史激励机制 → 多问题evaluation
3. **方向3**（方向2完成后或并行）→ 设计信念塑造策略 → 对比实验

## 共同约束

- 所有实验代码必须支持断点续跑（checkpoint/resume）
- 使用DeepSeek-V3 API（通过hf-mirror.com endpoint不适用，需直接API）
- 论文格式：Springer LNCS
- 数据：MMLU-Hard, StrategyQA, TruthfulQA
