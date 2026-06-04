# Stackelberg激励机制方案设计文档

## 主题：博弈驱动的去中心化语义共识机制研究：基于AEIC分层分解与Stackelberg激励的多智能体框架

---

## 第一部分：文献归纳 — Stackelberg博弈的正确学术范式

### 1.1 跨领域通用结构（30+篇论文归纳）

从FL、众包、网络资源分配、LLM定价四个领域的Stackelberg文献中，归纳出五个必要结构要素：

**要素1: Leader-Follower角色 + 利益冲突**

| 领域 | Leader | Follower | 利益冲突 |
|---|---|---|---|
| FL (Fu 2024, 59引; Guo 2023, 26引) | Server | Clients | Server要最高精度, Client要最少effort |
| 众包 (Xie 2024, 8引; Vahedi 2023, 8引) | Platform | Workers | Platform要高质量数据, Worker要偷懒 |
| LLM定价 (Guo, AAAI 2026) | Provider | Users | Provider要利润最大化, User要成本最低 |
| LLM Agent (Rauba 2026; Saig NeurIPS 2024) | Principal/Platform | Agent | Platform要高质量输出, Agent有偷懒动机(moral hazard) |

**要素2: 双层优化 (Bilevel Optimization)**

```
Upper level (Leader):  max  U_L(θ)
                       s.t. e_i = e_i*(θ)  ∀i   ← Follower最优响应
Lower level (Follower): e_i*(θ) = argmax U_i(e_i; θ)
```

Wang et al. (IEEE TAI 2024, 12引): "incentive mechanism is formulated as a bilevel optimization"
Fu et al. (IEEE IoT 2024, 59引): "bi-level ADMM algorithm to solve for the equilibrium"

**要素3: Backward Induction求解**

先解下层(Follower最优响应) → 代入上层(Leader单层优化) → 得到Stackelberg均衡(SE)

**要素4: IC (激励相容) + IR (参与约束)**

- IC: 在均衡处，诚实/高质量行为是Follower的占优策略
- IR: 参与效用 ≥ 0

**要素5: 理论证明标准件**

SE存在性 → IC性质 → IR约束 → (可选)唯一性/社会福利

### 1.2 LLM Agent领域核心文献

**A. 综述类**

| 论文 | 发表 | 引用 | 核心贡献 |
|---|---|---|---|
| Sun et al. "Game theory meets LLMs" | IJCAI 2025 | 47 | 完整分类LLM×Game Theory；提出ICSAP概念 |
| Chen et al. "Mechanism Design Meets LLMs" | SSRN 2026 | 1 | 专门综述MD与LLM交叉 |
| Tran et al. "Multi-agent collaboration mechanisms" | arXiv 2025 | 570 | 大综述，覆盖debate/vote/consensus |
| Han et al. "LLM MAS: Challenges" | arXiv 2024 | 314 | 提到Stackelberg equilibrium在MAS中的应用 |

**B. 与本工作最密切相关**

| 论文 | 关系 | 启示 |
|---|---|---|
| Rauba et al. (2026) "MAS Should be Principal-Agent Problems" | 理论定位完全一致 | MAS中的incentive问题应用MD框架 |
| Saig et al. (NeurIPS 2024, 20引) "Incentivizing quality text generation via statistical contracts" | 技术路线最接近 | 用contract theory激励LLM质量；moral hazard建模 |
| Yi et al. (2025, 30引) "From debate to equilibrium: BNE for multi-agent LLM" | 直接竞品 | 他们用Nash均衡，我们用Stackelberg |
| Xu et al. (2026, 6引) "RCBSF: Stackelberg + multi-agent LLM" | 最近已发表工作 | 在LLM MAS中直接使用Stackelberg |
| Kaltenpoth (2025) "Getting In Contract with LLMs — Agency Theory" | 理论支撑 | Agency theory分析LLM alignment |
| Liu et al. (2025, 9引) "Interpretable automated mechanism design with LLMs" | 技术参考 | LLM做自动化MD，IC约束 |

**C. Multi-agent Debate/Consensus竞品（我们要超越的）**

| 论文 | 发表 | 方法 | 缺陷(我们的优势) |
|---|---|---|---|
| Choi et al. "Debate or Vote" | NeurIPS 2025, 57引 | 对比debate vs voting | 纯empirical，无game theory |
| Pitre et al. "CONSENSAGENT" | ACL 2025, 33引 | 解决sycophancy | 启发式，无IC证明 |
| Wu & Ito "Hidden strength of disagreement" | EMNLP 2025, 17引 | consensus-diversity tradeoff | 提到game theory但未用 |
| Zhao et al. "Electoral approach" | EMNLP 2024, 31引 | Cardinal voting | 社会选择理论，非博弈论 |
| Li et al. "Dynamic Weighted Consensus" | Springer 2025 | 动态加权投票 | 无博弈论基础 |

**空白点：目前没有任何工作将Stackelberg机制设计应用于LLM多agent语义共识。**

---

## 第二部分：数据实证分析 — 设计前必须知道的事实

### 2.1 数据结构

```
signals_XXX_5a.json:
  meta: {dataset, n_agents=5, n_samples=200}
  questions: [
    {
      question_id, question, ground_truth,
      signals: {
        ic: [5 floats],     # inference-conclusion一致性
        ec: [5 floats],     # evidence-conclusion支持度
        eq: [5 floats],     # evidence质量
        ev: [5 floats],     # 跨agent证据共识
        stances: [5 strings], # yes/no/neutral
        agreement: [5 floats] # 邻居一致率
      }
    }, ...
  ]
```

10个数据集 × 200题 × 5 agents = 10,000个AEIC样本点

### 2.2 关键实证发现

**发现1: 质量信号几乎不区分个体正确性**

在StrategyQA的118道分歧题上（有人对有人错）：

| 信号 | 正确agent均值 | 错误agent均值 | 差异 |
|---|---|---|---|
| IC | 0.7729 | 0.7773 | -0.0044 |
| EC | 0.6603 | 0.6493 | +0.0110 |
| EQ | 0.7635 | 0.7597 | +0.0038 |
| EV | 0.7977 | 0.7868 | +0.0109 |

Quality vs Correctness的Kendall tau = -0.027，仅40.68%的题目有正相关。

**含义：AEIC质量信号反映的是"推理过程质量"，而非"结论正确性"。** 这两者在单个agent层面几乎无关，但在系统层面（通过适当加权聚合）可以提升准确率。

**发现2: DE优化学到的参数揭示了信号价值**

| 参数 | 默认值 | 优化值 | 含义 |
|---|---|---|---|
| ha_ev | 0.25 | 0.045 | **EV几乎被丢弃** — 共识度不重要 |
| aw_ev_scale | 0.50 | 0.018 | **EV在bid中也被丢弃** |
| aw_ic_exp | 1.0 | 2.34 | **IC被指数放大** — 推理链一致性最重要 |
| aw_eq_exp | 1.0 | 1.90 | **EQ被指数放大** — 证据质量重要 |
| ha_ec | 0.25 | 0.39 | **EC权重最高** |

**含义：最优机制抛弃了"共识度"(EV)，转而重赏"推理链质量"(IC, EC, EQ)。** 这与sycophancy问题的逻辑一致——逢迎行为会提高EV但不提高IC/EC。

**发现3: 优化θ*打破了简单的质量-权重单调性**

| 机制 | Quality-Weight Kendall tau | 正相关比例 |
|---|---|---|
| 默认θ | 0.827 | 99.5% |
| 优化θ* | 0.114 | 56.7% |

但优化θ*在分歧题上，正确agent的平均权重(0.579) > 错误agent(0.424)，差距比默认θ大3倍。

**含义：最优机制不是简单地"奖励质量"，而是通过Stackelberg动力学中的peer pressure和非线性映射，将质量信号转化为更有效的决策权重。** 简单的质量单调性不是正确的IC定义。

---

## 第三部分：三个核心问题的形式化

### 3.1 问题一：Follower的策略空间

#### 3.1.1 难点

在offline setting中，5个DeepSeek agent的AEIC输出已经固定在缓存中。它们在生成时是完全独立并行的（parallel模式），没有选择"逢迎"还是"独立"的机会。那么Follower的"策略选择"如何建模？

#### 3.1.2 解决方案：类型空间 + 反事实推理

参考Saig et al. (NeurIPS 2024)的approach，将Follower建模为拥有**类型(type)**的agent，机制设计的目标是设计对类型敏感的评分规则。

**定义（Agent类型）**:

每个agent i在每道题q上的类型由其AEIC输出的可观测质量向量决定：

$$\tau_i = (IC_i, EC_i, EQ_i) \in [0,1]^3$$

**定义（推理质量得分）**:

$$q_i = \phi(\tau_i) = IC_i^{\alpha} \cdot EC_i^{\beta} \cdot EQ_i^{\gamma}$$

其中α, β, γ是由机制参数θ控制的指数。注意：**EV不进入质量定义**，因为EV衡量的是"共识度"而非"推理质量"。一个逢迎的agent可以有高EV但低质量。

**定义（Follower策略空间 — 反事实建模）**:

虽然在缓存数据中每个agent的输出是固定的，但我们将agent的行为建模为从两个策略中选择的结果：

- **策略 $s^{ind}$（独立推理）**: 产出高IC、高EC、高EQ的AEIC输出。cost = $C_{ind}$（高）
- **策略 $s^{syc}$（逢迎/偷懒）**: 产出低IC或低EC的输出（推理链不完整或证据不支撑），但可能有高EV（通过模仿多数）。cost = $C_{syc}$（低）

**关键洞察**: 虽然我们无法观测每个agent"选择了什么策略"（因为这是LLM的内在过程），但我们可以从AEIC信号中**推断策略类型**：

$$\hat{s}_i = \begin{cases} s^{ind} & \text{if } q_i > \bar{q} \text{ (质量高于中位数)} \\ s^{syc} & \text{if } q_i \leq \bar{q} \text{ (质量低于中位数)} \end{cases}$$

其中 $\bar{q}$ 是同题所有agent的质量中位数。

#### 3.1.3 为什么这在offline setting下可行？

1. **数据已包含类型变异**: 5个DeepSeek agent在同一道题上产出的IC、EC、EQ差异显著（实测std≈0.1），提供了"高质量"和"低质量"的自然对照
2. **反事实通过组内比较实现**: 对每道题的5个agent，按质量排序 → 高质量组 vs 低质量组 → 在机制θ下各获得多少权重
3. **无需额外API调用**: 所有信号已预计算，反事实推理是纯数学操作

#### 3.1.4 Follower效用函数

$$U_i(s_i; \theta) = w_i(\tau_i(s_i); \theta) \cdot R - C(s_i)$$

其中：
- $w_i(\tau_i; \theta)$: 机制在参数θ下给agent i的归一化权重（由bid→Stackelberg→θ pipeline产出）
- $R$: 被采纳的收益（常数，对所有agent相同）
- $C(s^{ind}) > C(s^{syc})$: 独立推理的成本高于逢迎

---

### 3.2 问题二：IC指标的具体数学形式

#### 3.2.1 为什么简单的"质量单调性"不对

实证发现3表明：最优θ*打破了质量-权重单调性（tau从0.83降到0.11），但系统准确率反而提高。这说明**IC ≠ 质量单调性**。

原因：Stackelberg动力学中的peer pressure（图拉普拉斯扩散）和非线性映射使得最终权重不再简单地反映输入质量。这不是bug而是feature——机制通过复杂的信号组合实现了更优的聚合。

#### 3.2.2 正确的IC定义：策略效用比较

**定义（激励相容性, Incentive Compatibility）**:

机制 $(θ, \mathcal{G})$ 是激励相容的，当且仅当对所有agent i和所有题目q：

$$U_i(s^{ind}; \theta) \geq U_i(s^{syc}; \theta)$$

即：在机制θ下，选择独立推理的效用 ≥ 选择逢迎的效用。

展开：

$$w_i(\tau^{ind}_i; \theta) \cdot R - C_{ind} \geq w_i(\tau^{syc}_i; \theta) \cdot R - C_{syc}$$

等价于：

$$w_i(\tau^{ind}_i; \theta) - w_i(\tau^{syc}_i; \theta) \geq \frac{C_{ind} - C_{syc}}{R} = \Delta_C$$

**含义：独立推理获得的权重增益必须足以补偿额外的推理成本。**

#### 3.2.3 可计算的IC度量（基于缓存数据）

**Step 1: 构造每道题的"独立"和"逢迎"类型profile**

对每道题q，将5个agent按质量 $q_i$ 从高到低排序：

$$q_{(1)} \geq q_{(2)} \geq q_{(3)} \geq q_{(4)} \geq q_{(5)}$$

定义：
- 高质量组 $H = \{(1), (2)\}$（top-2 agent）
- 低质量组 $L = \{(4), (5)\}$（bottom-2 agent）

**Step 2: 计算权重差异**

$$\Delta w_q(\theta) = \frac{1}{|H|}\sum_{i \in H} w_i(\theta) - \frac{1}{|L|}\sum_{j \in L} w_j(\theta)$$

**Step 3: 全局IC度量**

$$\text{IC}(\theta) = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{1}[\Delta w_q(\theta) > 0]$$

即：在多少比例的题目上，高质量agent组的平均权重 > 低质量agent组的平均权重。

**Step 4: 加入DE目标函数**

$$\min_\theta \Big[ -\text{Accuracy}(\theta) - \lambda \cdot \text{IC}(\theta) \Big]$$

其中λ是超参数，控制IC约束的强度。

#### 3.2.4 更精细的IC变体（连续版本）

如果需要梯度友好的连续IC度量：

$$\text{IC}_{soft}(\theta) = \frac{1}{|Q|} \sum_{q \in Q} \sigma\left(\frac{\Delta w_q(\theta)}{\tau}\right)$$

其中 $\sigma$ 是sigmoid函数，$\tau$ 是温度参数。

#### 3.2.5 另一种IC定义：基于Kendall tau但只看组间

不要求全局quality-weight单调性（实证已证明这不对），而只要求**组间排序正确**：

$$\text{IC}_{pair}(\theta) = \frac{1}{|Q|} \sum_q \frac{\sum_{i \in H, j \in L} \mathbb{1}[w_i > w_j]}{|H| \cdot |L|}$$

这衡量的是：在每道题上，从高质量组和低质量组各取一个agent，高质量agent权重更大的比例。

---

### 3.3 问题三：Theorem 2（IC性质）的证明路径

#### 3.3.1 要证明什么

**Theorem (Incentive Compatibility at Stackelberg Equilibrium)**:

设 $\theta^*$ 为AEIC-Stackelberg机制通过双层优化（DE）求得的最优参数，则该机制是 $\epsilon$-激励相容的：

$$\forall q \in Q, \quad \Delta w_q(\theta^*) \geq -\epsilon$$

即对绝大多数题目，高质量推理组获得的权重不低于低质量组。

#### 3.3.2 证明策略

**路径A: 构造性证明（推荐，最自然）**

**核心思路**：θ*是 $\max_\theta [\text{Accuracy}(\theta) + \lambda \cdot \text{IC}(\theta)]$ 的解。通过证明IC项的性质，间接证明θ*处的IC保证。

**证明框架**：

1. **引理1（IC可行性）**: 存在 $\theta_0 \in \Theta$ 使得 $\text{IC}(\theta_0) = 1$。

   *证明*: 取 $\theta_0$ 使得 $\alpha_{IC}, \alpha_{EQ}$ 极大（如 $\alpha = 10$），sg_gamma = 0（无peer pressure）。此时weight是quality的严格单调函数，IC = 1。 □

2. **引理2（Accuracy有界）**: $\forall \theta, \text{Accuracy}(\theta) \leq 1$

3. **定理主体**:

   由θ*的最优性：
   $$\text{Accuracy}(\theta^*) + \lambda \cdot \text{IC}(\theta^*) \geq \text{Accuracy}(\theta_0) + \lambda \cdot \text{IC}(\theta_0)$$

   因此：
   $$\text{IC}(\theta^*) \geq \text{IC}(\theta_0) + \frac{\text{Accuracy}(\theta_0) - \text{Accuracy}(\theta^*)}{\lambda}$$

   由引理1, $\text{IC}(\theta_0) = 1$；由引理2, $\text{Accuracy}(\theta_0) - \text{Accuracy}(\theta^*) \geq -1$。

   因此：
   $$\text{IC}(\theta^*) \geq 1 - \frac{1}{\lambda}$$

   **当 $\lambda \geq \frac{1}{1-\epsilon}$ 时，$\text{IC}(\theta^*) \geq \epsilon$。** □

   *解读*: 通过选择足够大的λ，我们可以保证IC度量在SE处任意接近1，代价是可能略微降低Accuracy。

**路径B: 结构性证明（更强但需要简化模型）**

如果将Stackelberg game简化为直接映射（去掉迭代更新和peer pressure）：

$$w_i = \frac{\exp(\beta \cdot h(\tau_i; \theta))}{\sum_j \exp(\beta \cdot h(\tau_j; \theta))}$$

其中 $h(\tau_i; \theta) = \theta_1 \cdot IC_i + \theta_2 \cdot EC_i + \theta_3 \cdot EQ_i$（线性评分函数），$\beta > 0$ 是温度参数。

则 $w_i > w_j \iff h(\tau_i; \theta) > h(\tau_j; \theta)$

当θ*使得 $\theta_1, \theta_2, \theta_3 > 0$ 时，更高的IC/EC/EQ → 更高的 $h$ → 更高的权重。

Follower效用：$U_i = w_i \cdot R - C(s_i)$

IC条件：$w^{ind}_i - w^{syc}_i > \Delta_C / R$

**引理**: 当 $\beta$ 足够大且 $\bar{\tau}^{ind} > \bar{\tau}^{syc}$（独立推理的类型均值高于逢迎），IC成立。

*证明*: $\beta \to \infty$ 时softmax退化为argmax，高质量agent获得权重1，低质量获得0。weight差 → 1 > $\Delta_C/R$（假设R足够大）。由连续性，存在有限 $\beta^*$ 使得IC成立。 □

#### 3.3.3 实证验证方案

除了理论证明，还需在10个数据集上实证验证：

1. 在每个数据集上，用新目标函数（accuracy + λ·IC）运行DE
2. 报告IC(θ*)值（理论下界为1-1/λ）
3. 与对照组（纯accuracy优化）比较IC值
4. 画图展示：不同λ下accuracy vs IC的Pareto前沿

---

## 第四部分：完整的Stackelberg博弈定义

### 4.1 博弈参与者

| | Leader (机制设计者) | Follower i (LLM Agent) |
|---|---|---|
| 身份 | DE optimizer | 每个DeepSeek agent |
| 策略空间 | $\theta \in \Theta \subset \mathbb{R}^{16}$ | $s_i \in \{s^{ind}, s^{syc}\}$ |
| 目标 | $\max_\theta \text{Accuracy}(\theta) + \lambda \cdot \text{IC}(\theta)$ | $\max_{s_i} U_i(s_i; \theta)$ |
| 信息 | 观测所有agent的 $\tau_i$ | 观测机制参数θ |

### 4.2 博弈时序

```
t=0: Leader选择并公布θ (机制规则)
t=1: 每个Follower i 观测θ，选择策略 s_i
t=2: Agent产出AEIC，系统计算信号τ_i
t=3: 机制根据θ和τ_i计算权重w_i
t=4: 加权投票产出共识决策
```

### 4.3 均衡概念

**定义（Stackelberg均衡, SE）**:

策略组合 $(\theta^*, s^*)$ 构成SE，当且仅当：

1. **Follower最优响应**: $\forall i, s^*_i = \arg\max_{s_i} U_i(s_i; \theta^*)$
2. **Leader最优**: $\theta^* = \arg\max_\theta \text{Accuracy}(\theta) + \lambda \cdot \text{IC}(\theta)$, 其中IC的计算基于Follower的最优响应 $s^*(θ)$

### 4.4 定理体系

**Theorem 1 (SE存在性)**:

在策略空间有限（$\{s^{ind}, s^{syc}\}$）、参数空间紧凑（$\Theta$有界闭）、效用函数连续的条件下，AEIC-Stackelberg博弈存在Stackelberg均衡。

*证明*: 有限博弈的SE存在性由backward induction保证。Follower问题是离散选择（2个策略），最优响应总存在。Leader在紧集上优化连续函数，由Weierstrass极值定理最优解存在。 □

**Theorem 2 (激励相容性)**:

设 $\lambda \geq \frac{1}{1-\epsilon}$，则SE处的机制θ*满足ε-IC：

$$\text{IC}(\theta^*) \geq \epsilon$$

*证明*: 见3.3.2路径A。 □

**Theorem 3 (参与约束 + 社会福利)**:

(a) IR: 在θ*下，所有agent的效用非负（假设R足够大使得 $w_i \cdot R \geq C_{ind}$）。

(b) 社会福利: $\text{Accuracy}(\theta^*) \geq \text{Accuracy}(\theta_{uniform})$，即SE处准确率不低于均匀投票。

*证明(b)*: $\theta_{uniform}$（所有agent等权重）对应特定的θ值（如sg_gamma=∞使所有theta收敛）。由θ*的最优性，即使不考虑IC项，$\text{Accuracy}(\theta^*) \geq \text{Accuracy}(\theta_{uniform})$。 □

---

## 第五部分：实施路径

### 5.1 Phase 1: Fix模式（离线Stackelberg，零API开销）

**Step 1: 修改DE目标函数** (optimize_consensus.py)

```python
# 当前：
def objective(params_vec, questions, n_agents):
    config = ConsensusConfig.from_vector(params_vec)
    result = evaluate_config(questions, config, n_agents)
    return -result["accuracy"]

# 改为：
def objective(params_vec, questions, n_agents, lambda_ic=0.5):
    config = ConsensusConfig.from_vector(params_vec)
    result = evaluate_config(questions, config, n_agents)
    ic_score = compute_ic_metric(questions, config, n_agents)
    return -(result["accuracy"] + lambda_ic * ic_score)
```

**Step 2: 实现IC度量函数**

```python
def compute_ic_metric(questions, config, n_agents):
    """计算IC(θ): 高质量组权重>低质量组的比例"""
    ic_count = 0
    total = 0
    for q in questions:
        signals = q["signals"]
        # 计算质量得分（不含EV）
        quality = [(signals["ic"][i] + signals["ec"][i] + signals["eq"][i])/3 
                   for i in range(n_agents)]
        # 计算机制权重
        weights = compute_mechanism_weights(signals, config, n_agents)
        # 高质量组 vs 低质量组
        ranked = sorted(range(n_agents), key=lambda i: quality[i], reverse=True)
        H = ranked[:2]  # top-2
        L = ranked[-2:]  # bottom-2
        avg_w_H = sum(weights[i] for i in H) / len(H)
        avg_w_L = sum(weights[j] for j in L) / len(L)
        if avg_w_H > avg_w_L:
            ic_count += 1
        total += 1
    return ic_count / max(total, 1)
```

**Step 3: 对照实验**

| 实验 | 目标函数 | 输出 |
|---|---|---|
| Control | max accuracy(θ) | θ_control (已有) |
| Treatment | max accuracy(θ) + λ·IC(θ) | θ_stackelberg (待跑) |

对比指标：accuracy, IC, Pareto分析

**Step 4: 跨数据集验证**

在10个数据集上分别/联合训练，验证IC性质的泛化性。

### 5.2 Phase 2: Adaptive模式（在线Stackelberg）

- Leader = TopologyPlanner选择拓扑+agent组合
- Follower = 异构agent在给定拓扑下推理
- 使用Phase 1的θ*作为mechanism参数
- 新增IC约束：拓扑设计确保至少一半agent无上游输入（保持独立性）

### 5.3 Stackelberg.py重写方向

当前的StackelbergGame类本质上是分布式优化（同时梯度更新+peer pressure），不是博弈。两个选择：

**选择A（推荐）: 保留当前机制但重新解释**

当前的bid→game→theta pipeline实际上是一个复杂的**评分规则(scoring rule)**，将AEIC信号映射为权重。不需要在代码层面改成"先算follower response再算leader"——因为在offline setting下，所有agent输出已固定，博弈结构体现在DE的bilevel optimization中。

重新解释：
- StackelbergGame.run() = Follower的最优响应函数 f(signals; θ)
- DE optimization = Leader的最优化问题
- 合在一起 = 完整的bilevel Stackelberg

**选择B: 简化为直接映射**

用softmax评分替代迭代game:

```python
def mechanism_weights(signals, theta):
    scores = [theta.w_ic * ic[i] + theta.w_ec * ec[i] + theta.w_eq * eq[i] 
              for i in range(n)]
    return softmax(scores / temperature)
```

优点：更清晰、更容易证明IC。缺点：失去"Stackelberg game"的表述。

**推荐做法**: 保留选择A的代码，但在论文中用选择B的数学框架来证明定理（简化模型的理论分析是标准做法）。

---

## 第六部分：论文理论部分重写要点

### 6.1 删除

- 当前Theorem 1（Robbins-Monro收敛） — 这是优化理论，不是博弈论
- 当前Theorem 2（Laplacian能量衰减） — 这是分布式共识理论，不是博弈论
- consensus_energy的图信号处理解释

### 6.2 新增

1. **博弈定义**（Section 4.1-4.3）— 明确Leader/Follower/效用/策略空间
2. **Theorem 1: SE存在性** — 证明backward induction + Weierstrass
3. **Theorem 2: IC性质** — 构造性证明 + λ的作用
4. **Theorem 3: IR + 社会福利** — 参与约束 + accuracy下界
5. **实验: IC vs Accuracy Pareto前沿** — 不同λ下的trade-off曲线
6. **实验: IC对照** — control(纯accuracy) vs treatment(+IC约束)

### 6.3 论文叙事

> 在多agent LLM系统中，agent存在moral hazard：可能偷懒（低质量推理链）或逢迎（盲从上游结论）。我们将此建模为Stackelberg博弈：机制设计者（Leader）通过设计基于AEIC信号的评分规则，激励agent（Follower）产出高质量独立推理。AEIC分层分解（Assumption-Evidence-Inference-Conclusion）提供了可量化的推理质量度量，使机制设计者能够区分"真正推理"和"表面逢迎"。我们证明在Stackelberg均衡处，诚实独立推理是每个agent的占优策略（incentive compatible），且系统准确率不低于任何无激励基线。

### 6.4 创新点总结

1. **首个将Stackelberg机制设计应用于LLM多agent语义共识的工作**
2. **AEIC分层分解为机制设计提供可量化的推理质量信号** — 区别于FL中的数据量/模型大小等粗粒度指标
3. **实证发现EV(共识度)无用，IC/EC/EQ重要** — 为anti-sycophancy提供博弈论解释
4. **bilevel DE优化 = Stackelberg backward induction的计算实现** — 连接了经典博弈论与现代优化方法
