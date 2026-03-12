# MAS: Game-Driven Decentralized Semantic Mechanism

## 📖 项目概览

本项目实现了一个融合 **Stackelberg 博弈论** 与 **分布式共识控制** 的多智能体语义共识框架。

系统由多个独立节点（进程）组成，每个节点自主生成 AEIC 决策记录，节点间通过 HTTP 互相通信，共识引擎对全量节点对计算语义相似度矩阵，驱动分布式参数收敛。

---

## 📐 三个核心公式

| # | 公式 | 含义 |
|---|------|------|
| 1 | `E = (1/2) Σ w_ij (θ_i - θ_j)²` | 共识能量，衡量全网节点参数一致性，目标 E→0 |
| 2 | `min_θ [-(ΣU_i) + λ·E]` | Leader 优化目标，λ 权衡激励与一致性 |
| 3 | `θ_i^{t+1} = θ_i^t + η∇U_i - γΣ_j w_ij(θ_i-θ_j)` | 分布式学习动态，无中央控制 |

---

## 🏗️ 项目结构详解

```
MAS/
├── start.py                        # 实验入口（三大实验）
├── requirements.txt                # Python 依赖包列表
├── run_mas.sh                      # 运行脚本（支持 conda 环境）
├── QUICK_START.md                  # 快速参考手册
├── mas/                           # 核心模块目录
│   ├── __init__.py
│   ├── config.py                  # API 配置（百炼 Embedding + DeepSeek）
│   ├── agent_node.py              # 分布式节点进程（FastAPI 服务）
│   ├── registry_center.py         # 服务发现注册中心（任务公告板）
│   ├── coordination_engine.py     # 多节点协调编排引擎
│   ├── memory.py                  # 节点内存管理器
│   ├── expert_recruiter.py        # 专家招募模块
│   ├── task_planner.py            # 任务规划器
│   ├── data/                      # 数据生成与管理
│   │   ├── __init__.py
│   │   ├── generator.py           # DeepSeek 动态生成多节点 AEIC 数据
│   │   └── __pycache__/
│   ├── consensus/                 # 共识与博弈核心
│   │   ├── __init__.py
│   │   ├── consensus.py           # 多节点语义共识引擎（核心）
│   │   ├── stackelberg.py         # Stackelberg Consensus Game 实现
│   │   ├── hybrid_semantic_engine.py  # 阿里云百炼 Embedding 引擎
│   │   ├── agentverse.py          # 四阶段协作编排框架
│   │   └── __pycache__/
│   └── eval/                      # 评测模块
│       ├── __init__.py
│       ├── export_to_evalscope.py # 导出评测数据
│       └── run_eval.py            # 运行评测任务
├── evalscope/                     # EvalScope 评测工具
│   ├── __init__.py
│   ├── exporter.py
│   └── run_eval.py
├── results/                       # 实验输出目录
│   ├── generated_dataset.json     # DeepSeek 生成的数据缓存
│   ├── exp1_convergence.csv       # 收敛性实验结果
│   ├── exp2_semantic_comparison.csv # 语义方法对比结果
│   ├── exp3_baseline_comparison.csv # 基线对比结果
│   └── figures/                   # 生成的图表文件
└── __pycache__/
```

### 核心模块职责

- **start.py**: 统一实验入口，支持三大实验（收敛性验证、语义方法对比、基线对比），提供命令行参数控制节点数、轮数等。

- **config.py**: 集中管理所有API密钥和配置，包括阿里云百炼Embedding、DeepSeek生成模型、LLM评判等。

- **agent_node.py**: 实现分布式智能体节点，每个节点作为独立的FastAPI服务，支持加入/退出网络、发起任务、响应任务等功能。

- **registry_center.py**: 分布式系统的服务发现中心，维护节点注册表，提供任务公告板功能，支持节点动态发现和任务分发。

- **coordination_engine.py**: 多节点协调编排引擎，负责任务分配、进度跟踪、结果聚合等分布式协调工作。

- **consensus/consensus.py**: 核心共识引擎，实现多节点语义相似度计算，支持5种相似度方法（字符级、词级TF-IDF、BM25、Sentence-BERT、LLM评判）。

- **consensus/stackelberg.py**: Stackelberg博弈实现，包含共识能量计算、分布式学习更新、Leader-Follower优化等。

- **data/generator.py**: 数据生成器，使用DeepSeek API动态生成多节点AEIC决策记录，支持多种业务场景。

---

## 🤖 智能体分布式设计

### 架构概述

系统采用 **去中心化P2P架构**，每个智能体节点都是独立的进程，通过HTTP API进行通信：

```
[Registry Center] ←─── 节点注册/发现
       │
       ├── [Agent Node 1] ── HTTP ── [Agent Node 2]
       │         │                       │
       │         └─── 本地共识引擎        └─── 本地共识引擎
       │
       └── [Agent Node N] ── HTTP ── [Coordination Engine]
```

### 节点生命周期

1. **启动阶段**: 节点向Registry Center注册，声明自身能力（模型类型、角色、容量等）
2. **发现阶段**: 通过Registry获取其他活跃节点列表，建立P2P通信拓扑
3. **协作阶段**: 参与任务执行，进行博弈交互，计算共识
4. **退出阶段**: 优雅退出网络，清理资源

### 通信机制

- **同步通信**: RESTful API调用，用于任务邀请、提案提交、反馈传递
- **异步通信**: 基于FastAPI的Background Tasks，支持长时间运行的任务
- **容错机制**: 节点故障检测、心跳保活、自动重连

### 角色分工

| 角色 | 职责 | 示例模型 |
|------|------|----------|
| **Solver** | 执行具体任务，生成AEIC决策 | DeepSeek, Qwen |
| **Reviewer** | 评审提案，提供反馈 | GPT-4, Claude |
| **Coordinator** | 协调多节点协作 | 专用协调模型 |

---

## 🎮 博弈论设计

### Stackelberg Consensus Game (SCG)

系统实现 **Stackelberg领导者-跟随者博弈**，结合共识约束：

- **领导者 (Leader)**: 平台/协调者，设置激励参数 θ_i，目标是最大化全局效用同时保持共识
- **跟随者 (Followers)**: 各智能体节点，根据 θ_i 最大化自身效用，同时受共识约束影响

### 核心公式实现

#### 1. 共识能量 (Consensus Energy)
```python
def consensus_energy(self, states: np.ndarray) -> float:
    """E = (1/2) Σ_{i<j} w_ij (θ_i - θ_j)²"""
    energy = 0.0
    for i in range(self.num_agents):
        for j in range(i + 1, self.num_agents):
            energy += self.W[i, j] * (states[i] - states[j]) ** 2
    return energy / 2.0
```

#### 2. 领导者优化目标
```python
# min_θ [-(Σ U_i) + λ·E]
# 其中 U_i = q_i * θ_i - c_i  (质量承诺收益 - 成本)
```

#### 3. 分布式学习更新
```python
def update_theta(self, utilities: np.ndarray) -> np.ndarray:
    """θ_i^{t+1} = θ_i^t + η∇U_i - γΣ_j w_ij(θ_i - θ_j)"""
    gradients_u = self._compute_utility_gradients(utilities)
    consensus_forces = self._compute_consensus_forces()
    return self.leader_params + self.eta * gradients_u - self.gamma * consensus_forces
```

### 博弈流程

1. **初始化**: 随机设置初始 θ_i ∈ [0.5, 1.0]
2. **跟随者响应**: 各节点根据当前 θ_i 计算最优策略
3. **效用计算**: U_i = 相似度得分 × θ_i - 任务成本
4. **领导者更新**: 使用梯度下降优化公式2
5. **共识约束**: 通过公式3施加参数一致性压力
6. **收敛判断**: 当 E < ε 或达到最大轮数时停止

### 参数调优

- **λ (lambda_c)**: 共识-激励权衡系数
  - λ=0: 纯Stackelberg，无共识约束
  - λ=1: 平衡共识与激励（推荐）
  - λ>1: 强共识约束

- **η (eta)**: 学习率，控制更新步长
- **γ (gamma)**: 共识约束强度

---

## 🔗 共识机制设计

### 多节点语义共识引擎

不同于传统两节点对比，本系统支持 **N节点全量共识评估**：

```python
# 主接口：N节点全量评估
result = engine.evaluate_consensus(node_records)
# 返回：平均相似度、两两相似度矩阵、共识决策等
```

### AEIC 数据格式

每个节点生成结构化决策记录：

```python
{
    "node_id": "node_1",
    "assumptions": "申请人信用评分良好，收入稳定",
    "evidence": ["征信报告", "银行流水", "收入证明"],
    "inference": "各项指标满足准入条件，无不良记录",
    "conclusion": "批准贷款申请"
}
```

### 相似度计算方法

支持5种相似度计算方法，按精度和速度排序：

| 方法 | 实现原理 | 速度 | 精度 | 适用场景 |
|------|----------|------|------|----------|
| `char_jaccard` | 字符集合Jaccard相似度 | ⚡⚡⚡ | ☆ | 基准测试 |
| `word_tfidf` | jieba分词 + TF-IDF余弦相似度 | ⚡⚡ | ★★★ | 大规模离线处理 |
| `bm25` | Okapi BM25算法 | ⚡⚡ | ★★★★ | **默认推荐** |
| `sentence_bert` | 阿里云百炼text-embedding-v4 | ⚡ | ★★★★★ | 高精度需求 |
| `llm_judge` | DeepSeek/Qwen作为评判者 | 🐢 | ★★★★★ | 最高精度评测 |

### 加权相似度计算

对AEIC四个维度进行加权融合：

```python
sim_total = 0.2 * sim(A) + 0.3 * sim(E) + 0.2 * sim(I) + 0.3 * sim(C)
```

其中 A=Assumptions, E=Evidence, I=Inference, C=Conclusion

### 共识决策逻辑

基于相似度矩阵输出决策：

- **ESS_Consensus**: 平均相似度 > 0.8，全网高度一致
- **Audit_Required**: 相似度 0.6-0.8，需要人工审核
- **Reject**: 相似度 < 0.6，存在重大分歧

---

## 🔑 AEIC 数据格式

每个节点对同一任务独立生成一份 AEIC 决策记录：

| 字段 | 含义 | 示例 |
|------|------|------|
| `assumptions` | 前提假设 | "申请人信用评分良好" |
| `evidence` | 支撑证据（列表） | ["征信报告", "银行流水", "收入证明"] |
| `inference` | 推理过程 | "各项指标满足准入条件" |
| `conclusion` | 最终结论 | "批准贷款" |

共识引擎计算所有节点对的四层加权相似度：

```
sim_total = 0.2·sim(A) + 0.3·sim(E) + 0.2·sim(I) + 0.3·sim(C)
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API

编辑 `mas/config.py`（已内置默认配置）：

```python
# 阿里云百炼 Embedding（text-embedding-v4）
API_KEY   = "your-dashscope-key"
API_BASE  = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# DeepSeek（数据生成 + LLM-as-Judge）
DEEPSEEK_API_KEY  = "your-deepseek-key"
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
```

### 3. 运行实验

```bash
# 全部实验（首次自动调用 DeepSeek 生成数据并缓存）
python start.py

# 指定实验
python start.py --exp 1          # 收敛性验证
python start.py --exp 2          # 语义方法对比
python start.py --exp 3          # 基线对比

# 生成论文图表
python start.py --exp all --plot

# 强制重新生成数据（如需更换场景）
python start.py --regen

# 调整节点数（默认3）
python start.py --nodes 4
```

### 4. 启动分布式节点

```bash
# 先启动注册中心
python mas/registry_center.py --port 9000

# 分别启动各节点进程（不同终端）
python mas/agent_node.py --port 8001 --model deepseek --role solver
python mas/agent_node.py --port 8002 --model qwen    --role reviewer
python mas/agent_node.py --port 8003 --model deepseek --role solver
```

---

## 🔬 三大实验

### Exp-1：收敛性验证

验证公式3的收敛性，对多个 λ 值运行迭代，观察共识能量 E 单调递减。

```
预期输出（λ=1.0）：
  Round  1: E=0.1423  ΔΘ=0.0891
  Round  5: E=0.0614  ΔΘ=0.0312
  Round 10: E=0.0128  ΔΘ=0.0047  ← 收敛
  Round 15: E=0.0031  ΔΘ=0.0009  ✅ 单调递减
```

输出：`results/exp1_convergence.csv`

### Exp-2：语义相似度方法对比

在 DeepSeek 动态生成的多节点数据集上，对比以下方法：

| 方法 | 原理 | 速度 | 精度 |
|------|------|------|------|
| `char_jaccard` | 字符集合 Jaccard | ⚡⚡⚡ | ☆ |
| `word_tfidf` | jieba 词级 TF-IDF | ⚡⚡ | ★★★ |
| `bm25` | Okapi BM25 | ⚡⚡ | ★★★★ |
| `sentence_bert` | 百炼 text-embedding-v4 | ⚡ | ★★★★★ |

评估指标：MAE、三分类准确率、ESS 率、平均效用 U

输出：`results/exp2_semantic_comparison.csv`

### Exp-3：基线对比

对比四种任务分配方案的社会福利 Social Welfare：

| 方案 | 说明 |
|------|------|
| Random | 随机分配 |
| Pure-Stackelberg | λ=0，无共识约束 |
| Pure-Consensus | 按容量均分，无博弈激励 |
| **SCG (Ours)** | λ=1，共识+博弈 |

20 次随机试验 + scipy t 检验显著性分析。

输出：`results/exp3_baseline_comparison.csv`

---

## 🧩 核心模块说明

### `consensus.py` — 多节点语义共识引擎

```python
from mas.consensus.consensus import ConsensusEngine

engine = ConsensusEngine(similarity_method="bm25")  # 或 sentence_bert

# 主接口：N 个节点全量共识评估
node_records = [
    {"node_id": "node_0", "assumptions": "...", "evidence": [...], ...},
    {"node_id": "node_1", "assumptions": "...", "evidence": [...], ...},
    {"node_id": "node_2", "assumptions": "...", "evidence": [...], ...},
]
result = engine.evaluate_consensus(node_records)
# → { avg_similarity, pairwise, similarity_matrix, utility, decision, ... }

# 分布式学习更新
theta = engine.update_theta(result["utility"])
```

### `generator.py` — 多节点数据生成

```python
from mas.data import load_or_generate

# 有缓存就加载，否则调 DeepSeek 生成
dataset = load_or_generate(n_nodes=3)

# 获取所有轮次的节点记录
node_recs_list = dataset.get_node_records()
# → [[node_0_dict, node_1_dict, node_2_dict], [...], ...]

# 强制重新生成（更换场景 / 增加数量）
dataset = load_or_generate(force_regenerate=True, n_nodes=4)

# 只生成某些领域
from mas.data import TASK_SCENARIOS
finance = [s for s in TASK_SCENARIOS if s.domain == "finance"]
dataset = load_or_generate(scenarios=finance, n_per_scenario=2)
```

### `stackelberg.py` — 博弈分配

```python
from mas.consensus.stackelberg import StackelbergConsensusGame, AgentBid

bids = [
    AgentBid("node_0", 8001, "deepseek", "solver",
             quality_promise=0.85, cost_per_task=20, capacity=0.6),
    AgentBid("node_1", 8002, "qwen",     "reviewer",
             quality_promise=0.90, cost_per_task=25, capacity=0.4),
]

game = StackelbergConsensusGame(leader_port=8000, num_agents=2, lambda_c=1.0)

# 多轮迭代（验证收敛）
result = game.run_game_rounds(bids, total_workload=1.0, num_rounds=10)

# 单轮执行
result = game.execute_stackelberg_game(bids, total_workload=1.0)
```

---

## 📊 数据集说明

### 任务场景（21 个内置，可自由扩充）

| 领域 | 场景数 | 相似度分布 |
|------|--------|-----------|
| finance | 5 | high×2, medium×2, low×1 |
| medical | 5 | high×2, medium×2, low×1 |
| admin | 4 | high×2, medium×1, low×1 |
| legal | 4 | high×1, medium×1, low×2 |
| supply_chain | 3 | high×1, medium×1, low×1 |

### 扩充自定义场景

```python
from mas.data import TaskScenario, TASK_SCENARIOS, load_or_generate

my_scenario = TaskScenario(
    domain="custom",
    name="我的场景",
    description="多个节点对同一问题独立分析，部分节点意见有分歧",
    similarity_target="medium",   # high / medium / low
)

dataset = load_or_generate(
    scenarios=TASK_SCENARIOS + [my_scenario],
    n_per_scenario=2,
    force_regenerate=True,
)
```

---

## 🔧 API 配置说明

```python
# mas/config.py

# ── 阿里云百炼 Embedding ─────────────────────────────
API_KEY   = "sk-xxx"
API_BASE  = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_MODEL = "text-embedding-v4"

# ── DeepSeek（数据生成 + LLM-as-Judge）──────────────
DEEPSEEK_API_KEY  = "sk-xxx"
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL    = "deepseek-chat"

# ── LLM-as-Judge（复用 DeepSeek）────────────────────
LLM_API_KEY = DEEPSEEK_API_KEY
LLM_API_URL = DEEPSEEK_API_BASE + "/chat/completions"
LLM_MODEL   = "deepseek-chat"
```

---

## 📚 理论参考

1. **Von Stackelberg, H.** (1934). Marktform und Gleichgewicht
2. **Olfati-Saber et al.** (2007). Consensus and Cooperation in Networked Multi-Agent Systems. *Proceedings of the IEEE*
3. **Liu et al.** (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment. *EMNLP*

---

*最后更新：2025年*
