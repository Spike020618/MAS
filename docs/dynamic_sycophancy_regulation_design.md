# 方向1：动态Sycophancy调控 — 详细设计方案

## 1. 问题定义

在multi-round debate中，同质LLM agent存在严重sycophancy问题（Bertalanič & Fortuna 2026报告85.5%）。现有online Stackelberg注入固定的质量反馈，但效果不稳定（MMLU-Hard +5.4pp, TruthfulQA -6.7pp）。

**核心问题：** 如何根据debate动态自适应调节机制参数，在"防止过早共识坍缩"和"促进有效信息交换"之间取得平衡？

## 2. 设计思路

### 2.1 Sycophancy检测

每轮debate后，计算三个动态指标：

```python
# 1. Stance变化率：多少agent改变了立场
stance_change_rate = |{i : stance_i^(t) ≠ stance_i^(t-1)}| / n

# 2. 趋同度：当前轮stance的集中程度
majority_ratio = max(count(stance)) / n

# 3. 趋同速度：majority_ratio的轮间增量
convergence_speed = majority_ratio^(t) - majority_ratio^(t-1)
```

**解读：**
- `stance_change_rate > 0.6` 且 `convergence_speed > 0.2`：sycophantic collapse，多数agent跟风翻转
- `stance_change_rate < 0.2` 且 `majority_ratio < 0.8`：僵局，agent固执己见不交换信息
- `stance_change_rate ∈ [0.2, 0.6]`：健康的debate动态

### 2.2 动态参数调节

基于检测到的动态状态，调整两类参数：

#### A. Mechanism参数（影响Stackelberg权重计算）
```python
if sycophantic_collapse:
    config.ev_scale *= (1 - α)      # 降低从众信号权重
    config.gamma *= (1 - β)          # 降低peer pressure
    config.lambda_c *= (1 - β)       # 降低consensus energy
elif deadlock:
    config.gamma *= (1 + β)          # 增大peer pressure
    config.lambda_c *= (1 + β)       # 增大consensus energy
# else: 保持不变（健康动态，不干预）
```

#### B. Prompt指令（影响agent行为）

三种差异化的反馈模板：

```
[ANTI-SYCOPHANCY模式]
"注意：本轮讨论检测到多数分析师改变了立场。
请特别注意：独立的深度推理比盲目跟随多数更有价值。
如果你有充分理由支持你的原始立场，请坚持并详细说明理由。
机制会给予独立思考更高的权重。"

[ENCOURAGE-EXCHANGE模式]
"注意：各方立场目前分歧较大但缺乏有效交互。
请仔细审视对方的证据和推理链，识别你可能遗漏的关键信息。
如果对方的证据确实有说服力，合理调整你的结论并非从众。"

[NEUTRAL模式]（健康动态，最小干预）
"请在考虑各方观点后，基于证据和逻辑给出你的判断。"
```

### 2.3 完整流程

```
Round 1: 所有agent独立AEIC推理 → 记录initial stances
          ↓
      计算dynamics: stance分布、majority_ratio
          ↓
Round 2: 
  1. 检测状态: sycophantic / deadlock / healthy
  2. 调整参数: config.ev_scale, config.gamma, config.lambda_c
  3. 选择模板: ANTI-SYCOPHANCY / ENCOURAGE-EXCHANGE / NEUTRAL
  4. 注入反馈 + 质量信号 → agents生成Round 2 AEIC
          ↓
      计算dynamics: stance_change_rate, convergence_speed
          ↓
Round 3:
  1. 再次检测状态（可能与Round 2不同）
  2. 再次调整参数
  3. 选择模板
  4. 注入反馈 → agents生成Round 3 AEIC
          ↓
      最终: evaluate_and_decide (用动态调整后的config)
```

## 3. 实验设计

### 3.1 条件对比（A/B/C实验）

| 条件 | 标签 | 描述 |
|------|------|------|
| A | `debate_baseline` | 标准3-round debate，无任何反馈注入 |
| B | `debate_static` | 现有online Stackelberg，固定θ*反馈 |
| C | `debate_dynamic` | 动态Sycophancy调控，自适应参数+差异化prompt |

### 3.2 数据集与规模

- MMLU-Hard, StrategyQA, TruthfulQA
- 每个数据集50题（共150题 × 3条件 = 450次debate）
- 5 agents, 3 rounds per debate
- 断点续跑：按(dataset, condition, question_id)去重

### 3.3 评估指标

**主指标：**
- MajVote accuracy per condition per dataset

**过程指标（理解mechanism效果）：**
- Per-round stance change rate
- Per-round majority ratio
- Sycophancy rate（定义：agent从正确stance翻转到错误stance的比例，需要ground truth）
- Diversity preservation（Round 3的stance entropy vs Round 1）

**分析维度：**
- Dynamic模式被触发的频率（anti-sycophancy vs encourage-exchange vs neutral）
- 触发anti-sycophancy时accuracy是否更高
- 与static feedback的case-by-case对比

## 4. 代码修改清单

### 4.1 `core/orchestrator.py`

1. 新增 `_compute_debate_dynamics()` 方法
   - 输入：prev_nodes, cur_nodes
   - 输出：stance_change_rate, majority_ratio, convergence_speed, state (sycophantic/deadlock/healthy)

2. 新增 `_adjust_dynamic_config()` 方法
   - 输入：base_config, dynamics_state
   - 输出：adjusted config (修改ev_scale, gamma, lambda_c)

3. 新增 prompt模板：`DEBATE_CONTEXT_DYNAMIC`
   - 包含三种子模板（anti-sycophancy, encourage-exchange, neutral）

4. 修改 `_run_debate()` 
   - 新增 `dynamic=True` 参数
   - 在每轮间调用dynamics检测和参数调节
   - 记录dynamics历史到result

### 4.2 `experiments/run_online_stackelberg.py`

5. 新增条件 `debate_dynamic`
6. 记录dynamics历史（触发模式、参数调整值）

### 4.3 新增分析脚本

7. `experiments/analyze_dynamic_results.py`
   - 三条件对比表
   - Per-round dynamics可视化
   - Sycophancy rate分析

## 5. 超参数

| 参数 | 含义 | 初始值 | 说明 |
|------|------|--------|------|
| τ_high | sycophancy阈值 | 0.6 | >60% agent翻转 → sycophantic |
| τ_low | deadlock阈值 | 0.2 | <20% agent翻转 → deadlocked |
| α | ev_scale衰减率 | 0.5 | sycophancy时EV权重减半 |
| β | gamma/lambda_c调节率 | 0.3 | peer pressure调节幅度 |
| conv_threshold | 趋同速度阈值 | 0.2 | majority_ratio增量>0.2视为快速趋同 |

注：5个agent时stance_change_rate离散值为{0, 0.2, 0.4, 0.6, 0.8, 1.0}，τ_high=0.6意味着≥3个agent翻转触发anti-sycophancy。

## 6. 预期结果与意义

### 预期效果
- Dynamic在sycophancy严重的场景（StrategyQA debate）应优于baseline和static
- Dynamic在TruthfulQA上应避免static feedback的负面效果（-6.7pp → ≥0pp）
- 整体：debate_dynamic ≥ debate_baseline > debate_static（或至少不差于baseline）

### 论文意义
1. **首次将自适应Stackelberg博弈应用于LLM debate的sycophancy调控** — 不是静态的post-hoc reweighting，而是实时检测consensus dynamics并干预
2. **验证了Stackelberg博弈在同质agent下的真正价值** — 不在于质量信号加权，而在于动态行为调控
3. **连接机制设计与debate质量** — 提供可解释的干预策略（何时抑制从众、何时鼓励交换）
4. **回应reviewer可能的质疑** — "Stackelberg不优于MajVote" → "静态加权确实不优于，但动态调控有独特价值"
