# 文件索引

## 📂 完整文件列表

### 核心代码（src/）

```
src/
├── registry_center.py          Registry服务（基础设施）
│   ├── 服务发现（Agent注册/查询）
│   ├── 任务公告板（发布/查询任务）
│   └── 历史存储（共识记录）
│
├── agent_node.py               Agent节点（分布式）
│   ├── 加入/退出网络
│   ├── 发起任务（临时协调者）
│   ├── 响应任务（Solver/Reviewer）
│   └── 集成所有子模块
│
├── coordination_engine.py      协调引擎
│   ├── 编排AgentVerse四阶段
│   ├── 专家招募协调
│   ├── 任务规划执行
│   └── 共识博弈管理
│
├── expert_recruiter.py         专家招募
│   ├── 任务需求分析
│   ├── 从Registry发现Agent
│   └── 基于信任分选择
│
├── memory.py                   记忆系统
│   ├── 短期记忆（工作记忆）
│   ├── 长期记忆（持久化）
│   ├── 共识记忆（已达成）
│   └── 情景记忆（交互历史）
│
├── task_planner.py             任务规划
│   ├── 任务分解
│   ├── 依赖图构建
│   └── 执行计划生成
│
└── consensus/                  共识模块
    ├── consensus.py            共识引擎（四层语义算子）
    │   ├── SimHash（前提层）
    │   ├── MinHash（证据层）
    │   ├── NCD（推理层）
    │   └── Cosine（结论层）
    │
    ├── refined_analysis.py     精炼分析
    │   ├── 逻辑冲突检测
    │   └── 反义词惩罚
    │
    ├── agentverse.py           AgentVerse流程
    │   └── （可选扩展）
    │
    └── stackelberg.py          Stackelberg博弈
        └── （可选扩展）
```

### 示例和脚本（examples/）

```
examples/
├── start_registry.sh           启动Registry的Bash脚本
├── start_registry.py           启动Registry的Python脚本
├── start_agent.sh              启动Agent的Bash脚本
├── quick_start.py              快速开始示例
└── test_all.py                 系统测试脚本
```

### 文档（根目录）

```
.
├── README.md                   项目总览和使用说明
├── USAGE.md                    详细使用指南
├── REFACTOR_SUMMARY.md         重构总结
├── requirements.txt            依赖列表
└── FILE_INDEX.md              本文件
```

### 数据文件（src/consensus/）

```
src/consensus/
├── agent_a.csv                 Agent A的测试数据
├── agent_b.csv                 Agent B的测试数据
├── data.csv                    分析数据
└── simulation_output.csv       模拟输出
```

---

## 🎯 文件用途说明

### 必需文件（运行系统）

| 文件 | 作用 | 是否必需 |
|------|------|---------|
| `registry_center.py` | Registry服务 | ✅ 必需 |
| `agent_node.py` | Agent节点 | ✅ 必需 |
| `coordination_engine.py` | 协调引擎 | ✅ 必需 |
| `consensus.py` | 共识算法 | ✅ 必需 |
| `memory.py` | 记忆系统 | ✅ 必需 |
| `expert_recruiter.py` | 专家招募 | ✅ 必需 |
| `task_planner.py` | 任务规划 | ✅ 必需 |

### 辅助文件（增强功能）

| 文件 | 作用 | 是否必需 |
|------|------|---------|
| `refined_analysis.py` | 逻辑冲突检测 | 可选 |
| `agentverse.py` | AgentVerse扩展 | 可选 |
| `stackelberg.py` | Stackelberg扩展 | 可选 |

### 示例文件（学习使用）

| 文件 | 作用 | 推荐度 |
|------|------|--------|
| `quick_start.py` | 快速开始 | ⭐⭐⭐ |
| `test_all.py` | 系统测试 | ⭐⭐⭐ |
| `start_registry.py` | 启动脚本 | ⭐⭐ |
| `start_agent.sh` | 启动脚本 | ⭐⭐ |

### 文档文件（阅读理解）

| 文件 | 作用 | 阅读顺序 |
|------|------|---------|
| `README.md` | 项目总览 | 1️⃣ |
| `USAGE.md` | 使用指南 | 2️⃣ |
| `REFACTOR_SUMMARY.md` | 重构总结 | 3️⃣ |
| `FILE_INDEX.md` | 文件索引 | 4️⃣ |

---

## 🔗 文件依赖关系

### agent_node.py 依赖

```
agent_node.py
    ├── coordination_engine.py
    │   ├── task_planner.py
    │   ├── expert_recruiter.py
    │   └── memory.py
    ├── consensus/consensus.py
    └── memory.py
```

### coordination_engine.py 依赖

```
coordination_engine.py
    ├── task_planner.py
    ├── expert_recruiter.py
    └── memory.py
```

### expert_recruiter.py 依赖

```
expert_recruiter.py
    └── (无内部依赖，仅依赖httpx)
```

### 其他模块

```
registry_center.py   → 独立模块（FastAPI）
consensus.py         → 独立模块（纯算法）
memory.py            → 独立模块（数据结构）
task_planner.py      → 独立模块（图算法）
```

---

## 📝 修改指南

### 如果要修改共识算法

编辑：`src/consensus/consensus.py`

```python
# 调整权重
self.w = {'A': 0.1, 'E': 0.4, 'I': 0.2, 'C': 0.3}

# 调整收益参数
self.R = 100  # 奖励
self.C = 25   # 成本
```

### 如果要添加新的Agent角色

编辑：`src/expert_recruiter.py`

```python
self.role_templates["new_role"] = {
    "name": "新角色名称",
    "description": "角色描述",
    "required_capability": "capability_name"
}
```

### 如果要修改协作流程

编辑：`src/coordination_engine.py`

```python
# 修改最大轮次
MAX_ROUNDS = 5

# 修改ESS阈值
ESS_THRESHOLD = 55
```

### 如果要自定义记忆系统

编辑：`src/memory.py`

```python
# 修改记忆容量
def __init__(self, max_short_term=50, max_episodic=100):
    ...
```

---

## 🗂️ 文件大小统计

| 文件 | 行数（估计） | 复杂度 |
|------|------------|--------|
| `registry_center.py` | ~300 | 中 |
| `agent_node.py` | ~400 | 高 |
| `coordination_engine.py` | ~300 | 高 |
| `expert_recruiter.py` | ~200 | 中 |
| `memory.py` | ~250 | 中 |
| `task_planner.py` | ~200 | 中 |
| `consensus.py` | ~100 | 低 |
| **总计** | **~1,750** | - |

---

## 🔍 快速查找

### 想要...

**启动系统？**
→ 查看 `USAGE.md` 或运行 `examples/start_registry.py`

**理解架构？**
→ 查看 `README.md` 和 `REFACTOR_SUMMARY.md`

**修改算法？**
→ 编辑 `src/consensus/consensus.py`

**添加功能？**
→ 阅读源代码注释，从 `agent_node.py` 开始

**测试系统？**
→ 运行 `examples/test_all.py`

**查看示例？**
→ 运行 `examples/quick_start.py`

---

## ✅ 验收清单

检查所有文件是否存在：

- [x] src/registry_center.py
- [x] src/agent_node.py
- [x] src/coordination_engine.py
- [x] src/expert_recruiter.py
- [x] src/memory.py
- [x] src/task_planner.py
- [x] src/consensus/consensus.py
- [x] examples/start_registry.py
- [x] examples/start_agent.sh
- [x] examples/quick_start.py
- [x] examples/test_all.py
- [x] README.md
- [x] USAGE.md
- [x] REFACTOR_SUMMARY.md
- [x] requirements.txt

---

**文件索引更新日期：2026-02-03**
