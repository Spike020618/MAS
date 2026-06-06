# Multi-Topology Adaptive Experiment Design

## 实验目标

验证 AEIC+Stackelberg 共识机制在多种通信拓扑下的有效性，并证明 adaptive 动态拓扑选择优于任何固定拓扑。

## 数据集

| 数据集 | 考察能力 | 选择理由 |
|--------|---------|---------|
| MMLU-Hard | 知识推理 | 难度高、区分度最好，出现在所有最优训练组合中 |
| StrategyQA | 多步常识推理 | 需要组合推理，提升幅度最大(73%→82.5%) |
| TruthfulQA | 抗幻觉 | 测试agent被误导时能否互相纠正 |

每个数据集 200 题，5 agents (DeepSeek + zhipu_flash 混合)。

## 拓扑模式

| 拓扑 | 通信结构 | 特点 |
|------|---------|------|
| parallel | 独立推理，无通信 | 基准，最快 |
| sequential | 链式传递 C_{i-1}→A_i | 信息逐步深化 |
| star | hub汇总所有leaf结论 | 中心化综合 |
| debate | 多轮全连接，每轮看他人观点 | 对抗性收敛 |
| agentverse | solver→reviewer→revision | 评审制 |
| adaptive | planner根据题目动态选 | 本文核心贡献 |

## 共识方法

| 方法 | 论文来源 | 共识策略 |
|------|---------|---------|
| ChatEval | Zhao et al. 2024 | parallel + 多数投票 |
| UnguidedDebate | Du et al. 2024 | debate + 最终轮多数投票 |
| AgentVerse | - | agentverse + solver决策 |
| MajVote | baseline | 任意拓扑 + 多数投票 |
| AEIC+SG (Ours) | 本文 | 任意拓扑 + Stackelberg加权(θ*_T) |
| Adaptive (Ours) | 本文 | planner选拓扑 + Stackelberg加权 |

## 实验设计

### 核心对比

每种 baseline 使用其原论文对应的拓扑（公平对比"方法"而非"拓扑"）：
- ChatEval 只在 parallel 上跑（它的设计就是独立agent投票）
- UnguidedDebate 只在 debate 上跑（它的设计就是多轮辩论）
- AgentVerse 只在 agentverse 上跑（它的设计就是solver-reviewer）

我们的方法在**所有拓扑上都跑**，用每种拓扑独立训练的 θ*_T。

### 关键区别

**为什么不能直接复用现有 parallel signals？**

因为拓扑影响 agent 的 AEIC 输出：
- parallel: 每个 agent 独立推理
- sequential: 后面的 agent 看到前面的结论，会补充或反驳
- debate: 多轮交互后 agent 可能改变立场
- 因此需要为每种拓扑重新调 LLM 生成数据

**Ours vs Adaptive 的区别：**
- Ours (AEIC+SG): 固定拓扑，使用该拓扑的 θ*_T
- Adaptive: TopologyPlanner 根据题目特征动态选拓扑，使用对应的 θ*

## 执行流程

```
Step 1: collect  — 调API收集各拓扑signals（断点续跑）
Step 2: optimize — 每种拓扑离线训练θ*（DE优化器）
Step 3: evaluate — 运行所有对比实验（离线）
Step 4: adaptive — 运行adaptive拓扑实验（planner调API）
```

## API开销估算

| 拓扑 | 每题调用数 | 3数据集×200题 | 备注 |
|------|-----------|-------------|------|
| parallel | 5 | 已有 | 不需重跑 |
| sequential | 5 | 3,000 | 链式但总数同 |
| star | 6 | 3,600 | leaf并行+hub |
| debate | 15 | 9,000 | 3轮×5agent |
| agentverse | 6 | 3,600 | solver+reviewer+revise |
| adaptive planner | 1 | 600 | 仅分类 |

**总计**: ~19,800 新调用 + 600 planner = ~$20 (DeepSeek定价)

## 预期结论

1. **AEIC+SG 在所有拓扑上优于 MajVote** → Stackelberg权重分配的价值
2. **不同拓扑下最优 θ* 不同** → 拓扑确实影响共识参数
3. **Adaptive 优于任何单一固定拓扑** → 动态选择的价值
4. **AEIC+SG 优于 ChatEval/UnguidedDebate/AgentVerse** → 方法层面的优势

## 文件结构

```
experiments/
  collect_cache_topo.py    # 多拓扑数据收集（支持断点续跑）
  optimize_per_topology.py # 每种拓扑训练θ*
  baselines_real.py        # baseline评估
  run_multi_topology.py    # 统一入口（status/verify/collect/optimize/evaluate/adaptive）

core/
  adaptive_executor.py     # 自适应执行器 + classify_topology()

results/
  cache/{dataset}_{topology}.json   # AEIC原始缓存
  signals/{dataset}_{topology}.json # 预计算信号
  configs/opt_{topology}.json       # 各拓扑θ*
  comparison_results.json           # 最终对比结果
  adaptive_results.json             # adaptive实验结果
```
