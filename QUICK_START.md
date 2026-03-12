# 快速参考 — MAS 多节点语义共识

## 🎯 一句话描述

N 个独立节点各自生成 AEIC 决策记录 → 共识引擎计算全量两两语义相似度矩阵 → Stackelberg 博弈驱动参数收敛

---

## 📐 三个核心公式

```
① 共识能量    E = (1/2) Σ w_ij (θ_i - θ_j)²       → 目标 E→0
② Leader目标  min_θ [-(ΣU_i) + λ·E]                → λ 权衡激励与一致性
③ 分布式学习  θ_i^{t+1} = θ_i^t + η∇U_i - γΣw_ij(θ_i-θ_j)
```

---

## 🚀 最小可运行示例

```python
from mas.consensus.consensus import ConsensusEngine
from mas.data import load_or_generate

# 加载/生成多节点数据集（有缓存自动读缓存）
dataset = load_or_generate()
node_recs_list = dataset.get_node_records()
# → [[node_0, node_1, node_2], [...], ...]  每项为 AEIC 字典

# 多节点全量共识评估
engine = ConsensusEngine(similarity_method="bm25")
result = engine.evaluate_consensus(node_recs_list[0])
print(result["avg_similarity"])  # 全网平均相似度
print(result["pairwise"])        # 所有节点对的相似度
print(result["decision"])        # ESS_Consensus / Audit_Required / Reject
```

---

## 📊 相似度方法速查

| 方法 | 速度 | 精度 | 何时用 |
|------|------|------|--------|
| `char_jaccard` | ⚡⚡⚡ | ☆ | 仅作 baseline |
| `word_tfidf` | ⚡⚡ | ★★★ | 大规模离线 |
| `bm25` | ⚡⚡ | ★★★★ | **默认推荐** |
| `sentence_bert` | ⚡ | ★★★★★ | 百炼 API 已配置时 |
| `llm_judge` | 🐢 | ★★★★★ | 最高精度，有 API 开销 |

```python
engine = ConsensusEngine(similarity_method="bm25")          # 默认
engine = ConsensusEngine(similarity_method="sentence_bert",  # 百炼 Embedding
                         api_key="...", api_base="...")
```

---

## 🏃 实验命令速查

```bash
python start.py                    # 全部实验
python start.py --exp 1            # 收敛性验证
python start.py --exp 2            # 语义方法对比
python start.py --exp 3            # 基线对比
python start.py --exp all --plot   # 全部 + 生成论文图表
python start.py --regen            # 强制重新生成数据
python start.py --nodes 4          # 每轮 4 个节点（默认3）
python start.py --rounds 20        # 迭代20轮（默认15）
```

---

## 🗂️ 数据集操作

```python
from mas.data import load_or_generate, TASK_SCENARIOS

# 标准加载（有缓存秒读，无缓存调 DeepSeek 生成）
ds = load_or_generate()

# 调节参数
ds = load_or_generate(n_nodes=4, n_per_scenario=2, force_regenerate=True)

# 只取某领域
finance_only = [s for s in TASK_SCENARIOS if s.domain == "finance"]
ds = load_or_generate(scenarios=finance_only)

# 常用接口
ds.get_node_records()   # [[node_0, node_1, ...], ...]  ← 传给共识引擎
ds.ground_truths()      # [0.87, 0.42, 0.15, ...]
ds.labels()             # ["high", "medium", "low", ...]
ds.summary()            # 统计摘要字典
ds.to_dataframe()       # pandas DataFrame（含所有节点字段）
```

---

## 🌐 分布式节点启动

```bash
# 注册中心（先启动）
python mas/registry_center.py --port 9000

# 各节点（分别在不同终端）
python mas/agent_node.py --port 8001 --model deepseek --role solver   --registry http://127.0.0.1:9000
python mas/agent_node.py --port 8002 --model qwen    --role reviewer  --registry http://127.0.0.1:9000
python mas/agent_node.py --port 8003 --model deepseek --role solver   --registry http://127.0.0.1:9000

# 查看网络状态
curl http://127.0.0.1:9000/stats
curl http://127.0.0.1:9000/discover
```

---

## 🧩 核心类速查

### ConsensusEngine
```python
engine.evaluate_consensus(node_records)   # 主接口：N节点全量评估 → 相似度矩阵+决策
engine.evaluate_pair(rec_i, rec_j)        # 两节点对比（兼容用）
engine.update_theta(utility)              # 分布式学习更新 θ
```

### StackelbergConsensusGame
```python
game.run_game_rounds(bids, total_workload, num_rounds)  # 多轮迭代（验证收敛）
game.execute_stackelberg_game(bids, total_workload)      # 单轮执行
game.optimize_allocation(bids, workload)                 # 纯优化，无迭代
```

### GeneratedDataset
```python
ds.get_node_records()     # 节点记录列表（共识引擎入参格式）
ds.all_rounds             # ConsensusRound 对象列表
ds.by_domain("finance")   # 按领域筛选
ds.by_label("high")       # 按相似度等级筛选
ds.save(path)             # 保存缓存
GeneratedDataset.load(path)  # 从缓存加载
```

---

## 📁 结果文件说明

| 文件 | 内容 |
|------|------|
| `results/generated_dataset.json` | DeepSeek 生成的多节点 AEIC 数据缓存 |
| `results/exp1_convergence.csv` | 各 λ 值下的能量/参数变化曲线 |
| `results/exp2_semantic_comparison.csv` | 各相似度方法的 MAE/Acc/ESS/AvgU |
| `results/exp3_baseline_comparison.csv` | 四种分配方案的社会福利对比 |
| `results/figures/fig1_convergence.png` | 收敛曲线图 |
| `results/figures/fig2_semantic_comparison.png` | 方法对比柱状图 |
| `results/figures/fig3_baseline_comparison.png` | 基线对比箱线图 |

---

## ⚙️ config.py 关键字段

```python
API_KEY            # 百炼 Embedding key
API_BASE           # https://dashscope.aliyuncs.com/compatible-mode/v1
API_MODEL          # text-embedding-v4
DEEPSEEK_API_KEY   # DeepSeek key（数据生成）
DEEPSEEK_MODEL     # deepseek-chat
LLM_MODEL          # deepseek-chat（llm_judge 方法）
```

---

## 🧪 evalscope 评测

```bash
pip install evalscope

# 一键导出数据 + 运行两个评测任务
python mas/eval/run_eval.py

# 只跑分类准确率（MCQ，不需要 judge）
python mas/eval/run_eval.py --task mcq

# 只跑 QA 分析质量（LLM judge 打分）
python mas/eval/run_eval.py --task qa

# 换成其他模型来测试（evalscope service 模式）
python mas/eval/run_eval.py \
  --model qwen2.5-72b-instruct \
  --api-url https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions

# 只测少量样本（调试）
python mas/eval/run_eval.py --limit 5
```

也可以直接从 Python 调用：

```python
from mas.data import load_or_generate

ds = load_or_generate()
paths = ds.export_evalscope()   # 导出到 evalscope_data/
# 然后调用 run_eval.py 或自己配 TaskConfig
```

| 任务 | 格式 | 指标 | 说明 |
|------|------|------|------|
| `consensus_mcq` | general_mcq | AverageAccuracy | 三分类（high/medium/low） |
| `consensus_qa` | general_qa | LLM judge 1-5分 | 准确性/专业性/完整性 |

---

## 💡 常见问题

| 问题 | 解决 |
|------|------|
| 首次运行很慢 | 正在调 DeepSeek 生成数据，生成后缓存到 `results/generated_dataset.json` |
| 想换更多节点 | `--nodes 4` 或 `load_or_generate(n_nodes=4, force_regenerate=True)` |
| sentence_bert 不工作 | 检查 `config.py` 中的 `API_KEY` 是否填写 |
| 想加自定义场景 | 在 `mas/data/generator.py` 的 `TASK_SCENARIOS` 列表末尾追加 `TaskScenario(...)` |
| 清除缓存重新生成 | `python start.py --regen` 或删除 `results/generated_dataset.json` |
