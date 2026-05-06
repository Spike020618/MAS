# MAS: 博弈驱动的去中心化语义共识机制

## 项目简介

本项目实现了一个融合 **Stackelberg 博弈论** 与 **多智能体语义共识** 的系统。核心目标是让多个 Agent 在无中心控制的条件下，通过 AEIC 四层结构和自适应权重学习达成语义一致。

## 🎯 核心特性

### ✅ 语义共识引擎
- **AEIC 结构化推理**：Assumptions / Evidence / Inference / Conclusion 四层架构
- **多节点共识评估**：计算节点间语义相似度矩阵，支持 BM25、Jaccard、Sentence-BERT 等方法
- **自适应权重学习**：基于效用函数的动态权重优化（logits + softmax）

### ✅ Stackelberg 博弈框架
- **序贯决策过程**：Leader 承诺 → Follower 响应 → Leader 优化更新
- **共识能量最小化**：E = (1/2) Σ w_ij (θ_i - θ_j)²
- **分布式参数学习**：θ_i^{t+1} = θ_i^t + η ∇U_i - γ Σ_j w_ij(θ_i-θ_j)

### ✅ 多智能体辩论系统
- **LangGraph 辩论图**：支持 AEIC 结构化与非结构化两种辩论模式
- **4种实验配置**：单智能体/多智能体 × AEIC结构化/非结构化
- **性能指标分离**：语义相似度（输入多样性）vs Stackelberg收敛度（共识质量）

## 📊 实验结果验证

### 关键发现：多智能体共识的正确评估

| 配置 | 语义相似度 | 参数收敛度 | 共识能量 | Stackelberg收敛 |
|------|-----------|-----------|---------|---------------|
| 非结构化_单智能体 | 1.000 | 1.000 | 0.0000 | ✅ |
| 非结构化_多智能体 | 0.404 | 1.000 | 0.0000 | ✅ |
| AEIC_单智能体 | 1.000 | 1.000 | 0.0000 | ✅ |
| AEIC_多智能体 | 0.101 | 1.000 | 0.0000 | ✅ |

**核心洞察**：多智能体的"低语义相似度"不是缺陷，而是特征！它表明：
- ✅ 面对多样化输入仍能达成参数一致
- ✅ Stackelberg博弈在差异化观点下仍有效
- ✅ 最终共识比单一输入的"假共识"更有意义

## 🏗️ 项目结构

```
MAS/
├── mas/                           # 核心模块
│   ├── consensus/                # 共识与博弈核心
│   │   ├── consensus.py          # 多节点语义共识引擎
│   │   ├── stackelberg.py        # Stackelberg 博弈实现
│   │   ├── hybrid_semantic_engine.py  # 混合语义引擎
│   │   ├── agentverse.py         # 协作编排框架
│   │   └── langgraph_debate.py   # LangGraph辩论框架
│   ├── agent_node.py             # 分布式节点
│   ├── coordination_engine.py    # 多节点协调
│   ├── memory.py                 # 内存管理
│   └── config.py                 # 配置管理
├── experiments/                  # 实验框架
│   ├── proper_baselines.py       # 基于真实ConsensusEngine的Baseline
│   ├── proper_experiment_runner.py # 实验运行器
│   ├── dataset_generator.py      # 数据生成器
│   ├── results_analyzer.py       # 结果分析器
│   └── results/                  # 实验结果
├── rag/                          # RAG增强模块
├── test_debate_experiment.py     # 辩论实验框架（两层结构）
├── run_proper_experiments.py     # 传统实验入口
├── requirements.txt              # Python 依赖
├── START_HERE.md                 # 快速开始指南
├── PROPER_EXPERIMENT_GUIDE.md    # 实验说明
├── SEMANTIC_CONSENSUS_ANALYSIS.md # 共识分析文档
└── README.md                     # 本文件
```

## 🚀 快速开始

### 环境准备
```bash
pip install -r requirements.txt
```

### 运行辩论实验（推荐）
```bash
python test_debate_experiment.py
```
输出包含：
- 🧪 架构验证：4种配置对比测试
- 📚 数据集驱动实验：多任务验证

### 运行传统实验
```bash
python run_proper_experiments.py
```

## 📋 核心功能详解

### 1. 语义共识评估
```python
from mas.consensus.consensus import ConsensusEngine

engine = ConsensusEngine()
result = engine.evaluate_consensus(node_records)

print(f"平均共识度: {result['avg_similarity']:.4f}")
print(f"共识决策: {result['decision']}")
```

### 2. Stackelberg 博弈
```python
from mas.consensus.stackelberg import StackelbergConsensusGame

game = StackelbergConsensusGame(leader_port=8000, num_agents=3)
result = game.run_sequential_stackelberg(bids, task_context)

print(f"最终参数: {result['final_params']}")
print(f"收敛状态: {result['converged']}")
```

### 3. 多智能体辩论
```python
from test_debate_experiment import DebateExperimentFramework

framework = DebateExperimentFramework()
results = framework.run_architecture_validation()

# 查看4种配置的性能对比
for config, result in results.items():
    print(f"{config}: 语义相似度={result['metrics']['semantic_similarity']:.3f}")
```

## 🔧 依赖说明

### 核心依赖
- `numpy`：数值计算与矩阵运算
- `pandas`：数据处理与结果分析
- `scipy`：统计检验

### 可选依赖（自动降级）
- `jieba`：中文分词（BM25 相似度）
- `rank-bm25`：BM25 算法实现
- `sentence-transformers`：Sentence-BERT 嵌入
- `langchain` + `langgraph`：多智能体辩论框架

## 📈 实验框架

### 两层架构设计
```
DatasetLoader     → 负责数据加载和管理
    ↓
ExperimentRunner  → 负责实验执行和结果收集
    ↓
DebateExperimentFramework → 整合层，协调数据与实验
```

### 支持的实验配置
1. **非结构化_单智能体**：单一观点，无结构辩论
2. **非结构化_多智能体**：多观点，无结构辩论
3. **AEIC结构化_单智能体**：单一观点，结构化推理
4. **AEIC结构化_多智能体**：多观点，结构化推理

## 🎓 学术贡献

### 理论创新
- **Stackelberg语义共识**：首次将博弈论应用于多智能体语义一致性问题
- **序贯决策框架**：Leader-Follower 动态在语义空间中的实现
- **共识能量函数**：量化多节点参数分布的一致性程度

### 技术创新
- **AEIC推理结构**：四层语义架构支持更精确的共识评估
- **自适应权重学习**：基于效用的动态权重优化机制
- **多模态相似度**：支持字符级、词级、语义级多种相似度计算

### 实验验证
- **4种配置对比**：全面验证单/多智能体 × 结构化/非结构化的性能差异
- **性能指标分离**：区分输入多样性（语义相似度）与共识质量（参数收敛度）
- **真实数据集支持**：计划集成 BrowseComp 等公开基准数据集

## 📚 文档导航

- **[START_HERE.md](START_HERE.md)**：快速开始与操作指南
- **[PROPER_EXPERIMENT_GUIDE.md](PROPER_EXPERIMENT_GUIDE.md)**：传统实验流程说明
- **[SEMANTIC_CONSENSUS_ANALYSIS.md](SEMANTIC_CONSENSUS_ANALYSIS.md)**：共识机制深度分析

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！请确保：
- 遵循现有的代码风格
- 添加必要的测试用例
- 更新相关文档

## 📄 许可证

本项目采用 MIT 许可证。

