# MAS: 博弈驱动的去中心化语义共识机制

## 项目简介

本项目实现了一个融合 **Stackelberg 博弈论** 与 **多智能体语义共识** 的系统。核心目标是让多个 Agent 在无中心控制的条件下，通过 AEIC 四层结构和自适应权重学习达成语义一致。

## 项目结构

```
MAS/
├── mas/                           # 核心模块
│   ├── consensus/                # 共识与博弈核心
│   │   ├── consensus.py          # 多节点语义共识引擎
│   │   ├── stackelberg.py        # Stackelberg 博弈实现
│   │   ├── hybrid_semantic_engine.py  # 混合语义引擎
│   │   └── agentverse.py         # 协作编排框架
│   ├── agent_node.py             # 分布式节点
│   ├── coordination_engine.py    # 多节点协调
│   ├── memory.py                 # 内存管理
│   └── config.py                 # 配置管理
├── experiments/                  # 实验框架
│   ├── proper_baselines.py       # 对比实验基线
│   ├── proper_experiment_runner.py # 实验运行器
│   └── results_analyzer.py       # 结果分析器
├── run_proper_experiments.py     # 实验入口
├── requirements.txt              # Python 依赖
├── START_HERE.md                 # 快速开始指南
├── PROPER_EXPERIMENT_GUIDE.md    # 实验说明
└── README.md                     # 本文件
```

## 核心功能

- **AEIC 语义结构**：Assumptions / Evidence / Inference / Conclusion
- **多节点语义共识**：计算节点间对比相似度矩阵，输出平均共识度
- **自适应权重学习**：logits + softmax 参数化，按效用动态更新 AEIC 权重
- **Stackelberg 博弈**：Leader-Follower 激励-共识动态
- **正确实验框架**：基于真实 ConsensusEngine 的对比实验

## 快速开始

```bash
pip install -r requirements.txt
python run_proper_experiments.py
```

## 依赖说明

- `numpy`：数值计算
- `pandas`：结果与数据处理
- `scipy`：统计检验与实验分析
- `jieba`：中文分词（可选）
- `rank-bm25`：BM25 相似度（可选）
- `sentence-transformers`：Sentence-BERT 选项（可选）

## 运行入口

- `run_proper_experiments.py`：正确实验入口
- `experiments/proper_experiment_runner.py`：实验控制器
- `mas/consensus/consensus.py`：共识引擎核心
- `mas/consensus/stackelberg.py`：Stackelberg 共识博弈

## 文档说明

- `START_HERE.md`：快速开始与操作指南
- `PROPER_EXPERIMENT_GUIDE.md`：实验流程与结果说明

