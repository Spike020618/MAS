# 博弈驱动的去中心化语义共识机制研究

Game-Driven Decentralized Semantic Consensus for Multi-Agent Systems

## 概述

本项目实现了 AEIC-Stackelberg 多智能体语义共识框架，核心思想：通过四层语义分解（Assumption-Evidence-Inference-Conclusion）实现推理链的结构化表示与跨层一致性验证，以 Stackelberg 博弈驱动激励相容的去中心化收敛。

**核心结果**：在对抗性场景（多数智能体被误导）下，本方法准确率 48.8%，分别超出 ChatEval（0%）、AgentVerse（0%）和无引导辩论（23.3%），所有改进 p < 0.05。

## 项目结构

```
MAS/
├── core/                          # 核心算法模块
│   ├── similarity.py              # 多方法相似度计算（Embedding/BM25/TF-IDF）
│   ├── consensus.py               # AEIC 共识引擎 + 自适应权重学习
│   └── stackelberg.py             # Stackelberg 博弈（Leader-Follower + 能量收敛）
│
├── network/                       # 去中心化 P2P 网络
│   ├── protocol.py                # 消息协议定义（AEIC_RECORD/THETA_UPDATE/CONSENSUS）
│   ├── registry.py                # 注册中心（节点发现 + 任务广播）
│   ├── agent_node.py              # Agent 节点（独立推理 + 分布式共识）
│   └── runner.py                  # 单机多节点演示（asyncio）
│
├── experiments/                   # 实验套件
│   ├── dataset_loader.py          # 数据集加载（StrategyQA + HotpotQA → AEIC 格式）
│   ├── data_generator.py          # 合成数据生成器
│   ├── baselines.py               # Baseline 实现（ChatEval/Unguided Debate/AgentVerse/Ours）
│   ├── plot_config.py             # IEEE 风格图表配置
│   ├── exp1_aeic_vs_e2e.py        # 实验1：AEIC 分层 vs 端到端
│   ├── exp2_adaptive_weights.py   # 实验2：自适应权重学习
│   ├── exp3_stackelberg.py        # 实验3：Stackelberg 博弈收敛
│   ├── exp4_cognitive_diversity.py # 实验4：认知多样性消融
│   ├── exp5_baseline_comparison.py # 实验5：Baseline 横向对比
│   └── run_all.py                 # 一键运行全部实验
│
├── figures/                       # 生成的实验图表（PDF + PNG）
├── papers/                        # 参考论文
│   ├── 2605.00914v1.pdf           # The Cost of Consensus (2026)
│   └── 2308.10848v3.pdf           # AgentVerse (2023)
│
├── 论文.tex                        # 完整论文源文件
└── requirements.txt               # Python 依赖
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

依赖：numpy, scipy, matplotlib, jieba, rank-bm25, sentence-transformers, aiohttp

### 运行全部实验

```bash
python experiments/run_all.py --sim embedding
```

首次运行会下载 Sentence-Transformer 模型（~420MB），后续使用缓存。生成图表保存至 `figures/`。

### 运行单个实验

```bash
python experiments/exp5_baseline_comparison.py --sim embedding
```

### 运行 P2P 网络演示

```bash
python -m network.runner
```

启动 1 个注册中心 + 3 个 Agent 节点，演示分布式 Stackelberg 共识过程。

## 核心算法

### AEIC 语义分解

将推理链 R 解耦为四个正交层：
- **A**（Assumption）：前提假设
- **E**（Evidence）：证据集合
- **I**（Inference）：推理过程
- **C**（Conclusion）：最终结论

加权相似度：`SIM(Ri, Rj) = wA·simA + wE·simE + wI·simI + wC·simC`

### 自适应权重学习

通过 softmax-logits 参数化，基于效用反馈梯度更新各层权重。实验中证据层自动获得最高权重（~0.50）。

### Stackelberg 博弈

- **共识能量**：E = (1/2) Σ wij(θi - θj)²
- **分布式更新**：θ ← θ + η·∇J - γ·∇E（梯度上升 + 同伴压力）
- **收敛保证**：能量指数衰减，15 轮内收敛

### 跨层信任评估（对抗场景核心）

1. **I→C 一致性**：推理方向与结论方向是否匹配（检测被操纵的智能体）
2. **证据共识度**：与群体证据基础的一致程度
3. **置信度校准**：过度自信但证据薄弱的智能体被降权

复合信任：`T = 0.50·IC + 0.25·EvCons + 0.25·Cal`

## 实验结果

### 实验5：Baseline 横向对比

| 方法 | 总体准确率 | Easy | Hard | Oracle Gap↓ | p-value |
|------|-----------|------|------|-------------|---------|
| ChatEval | 0.494 | 1.000 | 0.000 | 0.506 | 1.1e-6 |
| Unguided Debate | 0.612 | 1.000 | 0.233 | 0.388 | 1.1e-2 |
| AgentVerse | 0.459 | 0.929 | 0.000 | 0.541 | 1.4e-7 |
| **Ours** | **0.741** | **1.000** | **0.488** | **0.259** | — |

### 其他实验关键数据

- **Exp1**：AEIC 证据层冲突定位差距 Δ=0.60（最强信号源）
- **Exp2**：自适应权重 vs Uniform 提升 8.9%（p = 6.2e-56）
- **Exp3**：共识能量 0.025 → 7.6e-5（15轮，两个数量级衰减）
- **Exp4**：异质组证据覆盖率 100% vs 同质组 33%

## Baseline 说明

| Baseline | 来源 | 核心机制 | 失败原因 |
|----------|------|---------|---------|
| ChatEval | Chan et al., 2023 | 均等权重多数投票 | 错误多数直接胜出 |
| Unguided Debate | Bertalanič & Fortuna, 2026 | 多轮辩论 + 从众模拟 | 正确少数被从众压力翻转 |
| AgentVerse | Chen et al., 2023 | 中心化评估者选择最佳 | 评估者被过度自信的错误智能体误导 |

## 论文

完整论文源文件：`论文.tex`（LaTeX，ctexart 文档类）

编译（需安装 TeX Live 或 MacTeX）：
```bash
xelatex 论文.tex
```
