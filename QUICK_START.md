# 🚀 博弈驱动的多智能体语义共识 - 快速开始

## 📋 前置条件

- Python 3.9+
- Docker & Docker Compose（可选，仅用于 RAG 模块）
- API Keys：DashScope、DeepSeek

## ⚡ 快速开始（5 分钟）

### 1. 安装依赖

```bash
cd /Users/spike/code/MAS
pip install -r requirements.txt
```

### 2. 配置 API

编辑 `mas/config.py`：

```python
# 阿里云百炼 Embedding（用于语义匹配）
API_KEY = "sk-xxx"
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# DeepSeek（用于数据生成和 LLM 评判）
DEEPSEEK_API_KEY = "sk-xxx"
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
```

### 3. 运行三大实验

```bash
# 运行全部实验
python start.py

# 或指定实验
python start.py --exp 1          # Exp-1: 收敛性验证
python start.py --exp 2          # Exp-2: 语义方法对比
python start.py --exp 3          # Exp-3: 基线对比

# 生成论文图表
python start.py --exp all --plot
```

---

## 🎯 核心实验说明

### Exp-1：共识收敛性验证 ⭐

验证公式 3（分布式学习动态）的收敛性：

```
θ_i^{t+1} = θ_i^t + η∇U_i - γΣ_j w_ij(θ_i - θ_j)
```

**预期输出**：共识能量 E 单调递减

```
λ=1.0 (推荐):
  Round  1: E=0.1423  ΔΘ=0.0891
  Round  5: E=0.0614  ΔΘ=0.0312
  Round 10: E=0.0128  ΔΘ=0.0047  ✅ 收敛
  Round 15: E=0.0031  ΔΘ=0.0009
```

**输出文件**：`results/exp1_convergence.csv`

### Exp-2：语义相似度方法对比

在多节点数据上对比 5 种相似度计算方法：

| 方法 | 原理 | 速度 | 精度 | 场景 |
|------|------|------|------|------|
| `char_jaccard` | 字符集合 Jaccard | ⚡⚡⚡ | ☆ | 基准 |
| `word_tfidf` | TF-IDF（中文分词） | ⚡⚡ | ★★★ | 大规模 |
| `bm25` | Okapi BM25 | ⚡⚡ | ★★★★ | **推荐** |
| `sentence_bert` | 百炼 text-embedding-v4 | ⚡ | ★★★★★ | **高精度** |
| `llm_judge` | DeepSeek 评判 | 🐢 | ★★★★★ | 验证 |

**预期结果**：`bm25` 和 `sentence_bert` 精度最高

**输出文件**：`results/exp2_semantic_comparison.csv`

### Exp-3：基线对比

对比 4 种任务分配方案在社会福利上的表现：

```
Social Welfare = Σ U_i - λ·E
```

| 方案 | 说明 | 预期社会福利 |
|------|------|-------------|
| Random | 随机分配 | 低 |
| Pure-Stackelberg | λ=0（无共识约束） | 中 |
| Pure-Consensus | 容量均分（无激励） | 中 |
| **SCG (Ours)** | λ=1（共识+博弈） | **高** ✅ |

**统计检验**：20 次随机试验 + scipy t 检验

**输出文件**：`results/exp3_baseline_comparison.csv`

---

## 🔧 参数调优指南

### 共识-激励权衡（λ）

```
λ = 0   : 纯 Stackelberg，无共识约束，参数可能分散
λ = 0.5 : 平衡但较弱的共识
λ = 1.0 : **推荐**，共识与激励平衡
λ > 1   : 强共识约束，激励可能不足
```

### 学习率（η）和约束强度（γ）

```python
# 默认参数（已在代码中优化）
eta = 0.01       # 领导者梯度下降步长
gamma = 0.1      # 共识约束强度
learning_rate = 0.01  # 权重学习率
```

---

## ✨ 可选：启动 RAG 系统

RAG 系统（Milvus + DashScope）作为辅助支撑模块，用于高性能语义匹配。

### 启动容器

```bash
# 一键启动（包括清理旧数据）
chmod +x clean_and_restart.sh
./clean_and_restart.sh

# 或快速启动（容器已运行）
chmod +x quick_start.sh
./quick_start.sh
```

### 访问 RAG 界面

- **Attu**（向量数据库管理）：http://localhost:8000
- **MinIO**（对象存储控制台）：http://localhost:9001

---

## 📊 数据集说明

项目包含 21 个内置任务场景：

| 领域 | 场景数 | 相似度分布 |
|------|--------|-----------|
| finance | 5 | 高×2, 中×2, 低×1 |
| medical | 5 | 高×2, 中×2, 低×1 |
| admin | 4 | 高×2, 中×1, 低×1 |
| legal | 4 | 高×1, 中×1, 低×2 |
| supply_chain | 3 | 高×1, 中×1, 低×1 |

### 扩充自定义场景

```python
from mas.data import TaskScenario, TASK_SCENARIOS, load_or_generate

my_scenario = TaskScenario(
    domain="custom",
    name="我的场景",
    description="多个节点对同一问题的分析",
    similarity_target="medium",  # high / medium / low
)

dataset = load_or_generate(
    scenarios=TASK_SCENARIOS + [my_scenario],
    n_per_scenario=2,
    force_regenerate=True,
)
```

---

## 💡 核心概念

### AEIC 决策记录

每个节点对同一任务独立生成：

```json
{
  "node_id": "node_1",
  "assumptions": "申请人信用评分良好，收入稳定",
  "evidence": ["征信报告", "银行流水", "收入证明"],
  "inference": "各项指标满足准入条件，无不良记录",
  "conclusion": "批准贷款申请"
}
```

### 加权相似度

```
sim_total = 0.2·sim(A) + 0.3·sim(E) + 0.2·sim(I) + 0.3·sim(C)
```

### 共识决策

- **ESS_Consensus**：avg_sim > 0.8，全网高度一致 ✅
- **Audit_Required**：0.6 < avg_sim < 0.8，需人工审核 ⚠️
- **Reject**：avg_sim < 0.6，存在重大分歧 ❌

---

## 🚀 分布式节点启动

启动多个智能体节点进行实时协作（可选）：

```bash
# 终端1：启动注册中心
python mas/registry_center.py --port 9000

# 终端2：启动 Node 1
python mas/agent_node.py --port 8001 --model deepseek --role solver

# 终端3：启动 Node 2
python mas/agent_node.py --port 8002 --model qwen --role reviewer

# 终端4：启动 Node 3
python mas/agent_node.py --port 8003 --model deepseek --role solver
```

---

## 📈 权重学习机制 ✨ NEW

从 2026-03-13 起支持自适应权重学习：

```python
# 自动学习权重（在实验中）
engine.update_weights(utility, learning_rate=0.01)

# 查看权重演化
weights_evolution = engine.get_weights_evolution()
# → {"A": [...], "E": [...], "I": [...], "C": [...]}
```

**权重更新规则**：

```
w_logits(t+1) = w_logits(t) + η × normalized_utility(t)
w(t) = softmax(w_logits(t))
```

---

## 🔍 常见问题

### Q：第一次运行很慢？
**A**：DeepSeek 首次生成数据集需要调用 API，约 5-10 分钟，生成后会缓存在 `results/generated_dataset.json`。

### Q：如何强制重新生成数据？
**A**：
```bash
python start.py --regen
```

### Q：如何调整节点数？
**A**：
```bash
python start.py --nodes 5  # 改为 5 个节点
```

### Q：RAG 系统是必要的吗？
**A**：否。RAG 系统是可选的辅助模块，用于高性能语义匹配。核心实验不依赖它。

---

## 📚 输出文件说明

实验完成后，查看 `results/` 目录：

```
results/
├── generated_dataset.json           # 缓存的数据集
├── exp1_convergence.csv            # 收敛性实验结果
├── exp2_semantic_comparison.csv    # 语义方法对比
├── exp3_baseline_comparison.csv    # 基线对比
└── figures/                        # 生成的图表（--plot）
    ├── convergence_curve.png
    ├── semantic_methods.png
    └── baseline_comparison.png
```

---

## 🎓 论文核心

### 三个核心公式

**1. 共识能量**
```
E = (1/2) Σ_{i<j} w_ij (θ_i - θ_j)²
```
衡量全网参数一致性，目标 E → 0

**2. Leader 优化目标**
```
min_θ [-(Σ U_i) + λ·E]
```
权衡社会福利与共识

**3. 分布式学习更新**
```
θ_i^{t+1} = θ_i^t + η∇U_i - γΣ_j w_ij(θ_i-θ_j)
```
无中央协调的参数收敛

---

## 🚀 现在开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API（mas/config.py）

# 3. 运行实验
python start.py

# 4. 查看结果
ls results/
```

**预计耗时**：5-15 分钟（取决于 API 响应）

---

**更新日期**：2026-03-17 | **版本**：v1.0
