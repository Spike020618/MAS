# MAS: 博弈驱动的去中心化语义共识机制

<div align="center">

**Game-Driven Decentralized Semantic Mechanism for Multi-Agent Consensus**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Milvus](https://img.shields.io/badge/Milvus-2.5+-green.svg)](https://milvus.io)
[![DashScope](https://img.shields.io/badge/阿里云-DashScope-orange.svg)](https://dashscope.aliyuncs.com)

</div>

---

## 📖 项目概览

本项目实现了一个融合 **Stackelberg 博弈论** 与 **分布式共识控制** 的多智能体语义共识框架。

系统由多个独立节点组成，每个节点自主生成 AEIC 决策记录，节点间通过 HTTP 互相通信，共识引擎对全量节点对计算语义相似度矩阵，驱动分布式参数收敛。

### 三个核心公式

| # | 公式 | 含义 |
|---|------|------|
| 1 | `E = (1/2) Σ w_ij (θ_i - θ_j)²` | 共识能量，衡量全网节点参数一致性，目标 E→0 |
| 2 | `min_θ [-(ΣU_i) + λ·E]` | Leader 优化目标，λ 权衡激励与一致性 |
| 3 | `θ_i^{t+1} = θ_i^t + η∇U_i - γΣ_j w_ij(θ_i-θ_j)` | 分布式学习动态，无中央控制 |

---

## 🏗️ 项目结构

```
MAS/
├── mas/                           # 核心模块
│   ├── config.py                 # 配置管理
│   ├── consensus/                # 共识与博弈核心 ⭐
│   │   ├── consensus.py          # 多节点语义共识引擎
│   │   ├── stackelberg.py        # Stackelberg 博弈实现
│   │   ├── hybrid_semantic_engine.py  # 混合语义引擎
│   │   └── agentverse.py         # 协作编排框架
│   ├── data/                     # 数据管理
│   │   └── generator.py          # 数据生成器
│   ├── eval/                     # 评测模块
│   │   ├── export_to_evalscope.py
│   │   └── run_eval.py
│   ├── agent_node.py             # 分布式节点
│   ├── registry_center.py        # 服务发现
│   ├── coordination_engine.py    # 多节点协调
│   ├── memory.py                 # 内存管理
│   ├── expert_recruiter.py       # 专家招募
│   └── task_planner.py           # 任务规划
├── mas/rag/                       # RAG 系统（辅助支撑）✨ NEW
│   ├── config.py                 # RAG 配置
│   ├── embedding_model.py        # DashScope 向量化
│   ├── milvus_db.py             # Milvus 驱动
│   ├── rag_database.py           # RAG 接口
│   └── demo_rag_milvus.py        # 演示脚本
├── start.py                      # 实验入口
├── requirements.txt              # Python 依赖
├── docker-compose-milvus.yml    # Docker 配置（RAG）
├── QUICK_START.md                # 快速开始
└── README.md                     # 本文件
```

---

## 🎮 核心创新点

### 1️⃣ Stackelberg 共识博弈
- **Leader**：协调者设置激励参数 θ_i
- **Followers**：各智能体节点根据 θ_i 优化策略
- **共识约束**：参数收敛至一致

### 2️⃣ 多节点语义共识引擎
支持 **5 种相似度方法**，从快速到精确：
- ⚡ `char_jaccard` - 字符集合 Jaccard
- ⚡⚡ `word_tfidf` - TF-IDF（中文分词）
- ⚡⚡ `bm25` - Okapi BM25（推荐）
- ⚡ `sentence_bert` - 阿里云百炼 text-embedding-v4（向量化）
- 🐢 `llm_judge` - DeepSeek 作为评判者

### 3️⃣ AEIC 决策记录格式
```json
{
  "node_id": "node_1",
  "assumptions": "申请人信用评分良好，收入稳定",
  "evidence": ["征信报告", "银行流水", "收入证明"],
  "inference": "各项指标满足准入条件，无不良记录",
  "conclusion": "批准贷款申请"
}
```

加权相似度：`sim = 0.2·sim(A) + 0.3·sim(E) + 0.2·sim(I) + 0.3·sim(C)`

### 4️⃣ 权重学习机制 ✨ NEW（2026-03-13）
- **自适应权重**：logits + softmax 参数化
- **梯度学习**：∇w = normalized_utility × w
- **收敛保证**：Robbins-Monro 定理

---

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置 API

编辑 `mas/config.py`：

```python
# 阿里云百炼 Embedding
API_KEY = "sk-xxx"
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# DeepSeek（数据生成 + LLM-as-Judge）
DEEPSEEK_API_KEY = "sk-xxx"
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
```

### 运行三大实验

```bash
# 全部实验
python start.py

# 指定实验
python start.py --exp 1          # 收敛性验证
python start.py --exp 2          # 语义方法对比
python start.py --exp 3          # 基线对比

# 生成论文图表
python start.py --exp all --plot
```

---

## 🔬 三大实验说明

### Exp-1：收敛性验证
验证公式 3 的收敛性，观察共识能量 E 单调递减

```
预期输出（λ=1.0）：
  Round  1: E=0.1423  ΔΘ=0.0891
  Round 10: E=0.0128  ΔΘ=0.0047  ✅ 收敛
```

### Exp-2：语义相似度方法对比
在多节点数据集上对比 5 种相似度方法

| 方法 | 速度 | 精度 | 推荐场景 |
|------|------|------|----------|
| bm25 | ⚡⚡ | ★★★★ | **默认** |
| sentence_bert | ⚡ | ★★★★★ | **高精度** |

### Exp-3：基线对比
对比 4 种任务分配方案的社会福利

| 方案 | 说明 |
|------|------|
| Random | 随机分配 |
| Pure-Stackelberg | λ=0，无共识约束 |
| Pure-Consensus | 按容量均分 |
| **SCG (Ours)** | λ=1，共识+博弈 ✅ |

---

## ✨ RAG 系统（辅助支撑模块）

从 2026-03-17 起，项目集成了 **Milvus + DashScope** RAG 系统作为语义匹配的高性能支撑。

### 启动 RAG 演示

```bash
# 一键启动（包括容器清理）
chmod +x clean_and_restart.sh
./clean_and_restart.sh

# 快速演示（容器已启动）
chmod +x quick_start.sh
./quick_start.sh

# 运行完整项目
chmod +x run_mas.sh
./run_mas.sh
```

### RAG 访问地址

- **Attu**（向量管理）：http://localhost:8000
- **MinIO**（对象存储）：http://localhost:9001
- **Milvus gRPC**：localhost:19530

### RAG 核心模块

| 文件 | 用途 |
|------|------|
| `mas/rag/config.py` | RAG 配置 |
| `mas/rag/embedding_model.py` | DashScope 向量化 |
| `mas/rag/milvus_db.py` | Milvus 驱动 |
| `mas/rag/rag_database.py` | RAG 数据库接口 |

---

## 📊 核心模块说明

### `consensus.py` - 多节点语义共识引擎

```python
from mas.consensus.consensus import ConsensusEngine

engine = ConsensusEngine(similarity_method="bm25")

# 多节点全量共识评估
result = engine.evaluate_consensus(node_records)
# → { avg_similarity, pairwise, utility, decision, ... }

# 权重学习更新
engine.update_weights(result["utility"])
```

### `stackelberg.py` - Stackelberg 博弈

```python
from mas.consensus.stackelberg import StackelbergConsensusGame

game = StackelbergConsensusGame(
    leader_port=8000, 
    num_agents=3, 
    lambda_c=1.0  # 共识-激励权衡
)

result = game.run_game_rounds(bids, num_rounds=10)
```

### `generator.py` - 数据生成

```python
from mas.data import load_or_generate

# 有缓存就加载，否则调用 DeepSeek 生成
dataset = load_or_generate(n_nodes=3)

# 强制重新生成
dataset = load_or_generate(force_regenerate=True)
```

---

## 📚 理论参考

1. **Von Stackelberg, H.** (1934). Marktform und Gleichgewicht
2. **Olfati-Saber et al.** (2007). Consensus and Cooperation in Networked Multi-Agent Systems
3. **Liu et al.** (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment

---

## 📝 论文内容概览

- **Stackelberg 博弈论**：Leader-Follower 机制设计
- **分布式共识控制**：无中央协调的参数收敛
- **语义匹配**：多种相似度方法的融合与学习
- **权重学习**：自适应加权机制的理论与实验

---

## 🔗 相关文件

- **快速开始**：见 [QUICK_START.md](QUICK_START.md)
- **RAG 配置**：见 `mas/rag/config.py`
- **博弈参数**：见 `mas/config.py`

---

**项目状态**：论文核心实现 ✅ | RAG 系统集成 ✅ | 三大实验就绪 ✅

*最后更新：2026-03-17*
