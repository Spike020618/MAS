# 分布式多智能体语义共识系统

基于AgentVerse框架和Stackelberg博弈理论的分布式多智能体协作系统。

## 🎯 核心特性

- **去中心化架构**：Registry仅作服务发现，不做中心化决策
- **动态专家招募**：根据任务需求和Agent信誉自动招募
- **四层语义算子**：SimHash、MinHash、NCD、Cosine相似度
- **博弈论共识**：基于演化稳定策略(ESS)的收敛机制
- **记忆系统**：短期/长期/共识记忆，支持经验积累
- **P2P协作**：Agent间直接通信，减少中心节点压力

## 📁 项目结构

```
paper/
├── src/
│   ├── registry_center.py      # Registry服务（基础设施）
│   ├── agent_node.py           # Agent节点（分布式）
│   ├── coordination_engine.py  # 协调引擎
│   ├── expert_recruiter.py     # 专家招募
│   ├── memory.py               # 记忆系统
│   ├── task_planner.py         # 任务规划
│   └── consensus/
│       ├── consensus.py        # 共识引擎（四层语义算子）
│       ├── agentverse.py       # AgentVerse流程
│       └── stackelberg.py      # Stackelberg博弈
├── examples/
│   ├── start_registry.sh       # 启动Registry脚本
│   ├── start_agent.sh          # 启动Agent脚本
│   └── quick_start.py          # 快速开始示例
├── requirements.txt
└── README.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动Registry（必需）

**开启服务的人必须先运行Registry**：

```bash
# 终端1
python src/registry_center.py --port 9000
```

或使用脚本：

```bash
chmod +x examples/start_registry.sh
./examples/start_registry.sh
```

### 3. 启动Agent节点

**其他人可以启动Agent加入网络**：

```bash
# 终端2 - Agent A (Solver)
python src/agent_node.py --port 8001 --model qwen --role solver

# 终端3 - Agent B (Reviewer)
python src/agent_node.py --port 8002 --model deepseek --role reviewer

# 终端4 - Agent C (Initiator)
python src/agent_node.py --port 8003 --model qwen --role initiator
```

或使用脚本：

```bash
chmod +x examples/start_agent.sh
./examples/start_agent.sh 8001 qwen solver
./examples/start_agent.sh 8002 deepseek reviewer
./examples/start_agent.sh 8003 qwen initiator
```

### 4. 运行演示

```bash
# 终端5 - 快速开始示例
python examples/quick_start.py
```

## 📖 使用指南

### 基本概念

#### Registry Center
- **职责**：服务发现、任务公告、历史存储
- **不负责**：任务调度、Agent通信路由、共识仲裁
- **运行者**：开启服务的人（类似区块链的创世节点）

#### Agent Node
- **职责**：执行任务、参与博弈、维护记忆
- **角色**：
  - `solver`：提出解决方案
  - `reviewer`：评审和反馈
  - `initiator`：可发起任务（临时协调者）

#### 工作流程

```
用户 → Agent(Initiator) → Registry(发现其他Agent) → P2P协作 → 共识 → 结果
```

### 发起任务

任何Agent都可以发起任务并成为临时协调者：

```python
from agent_node import DistributedAgent
import asyncio

async def main():
    # 创建Agent
    agent = DistributedAgent(
        port=8001,
        model="qwen",
        role="initiator",
        registry_url="http://127.0.0.1:9000"
    )
    
    # 加入网络
    await agent.join_network()
    
    # 发起任务
    result = await agent.initiate_task(
        task_desc="审核企业贷款",
        task_data={
            "assumptions": "基础模型",
            "evidence": ["身份证", "营业执照"],
            "inference": "标准流程",
            "conclusion": "待定"
        },
        goal="达成ESS共识"
    )
    
    print(f"收益U: {result['result']['utility']:.2f}")
    print(f"轮次: {result['result']['rounds']}")

asyncio.run(main())
```

### 查看统计

```bash
# 查看Registry统计
curl http://127.0.0.1:9000/stats

# 查看Agent统计
curl http://127.0.0.1:8001/stats
```

## 🎮 AgentVerse四阶段流程

### Stage 1: 专家招募
- 根据任务需求生成所需角色
- 从Registry发现匹配的Agent
- 基于信任分优先选择

### Stage 2: 协作决策
- **垂直结构**：Solver提方案 → Reviewer评审 → Solver修正
- 循环迭代直到达成ESS或超过最大轮次

### Stage 3: 行动执行
- Agent并行处理子任务
- P2P通信交换结果

### Stage 4: 评估与反馈
- 计算语义共识收益U
- 更新Agent信任分
- 存储共识到Registry

## 🧮 语义共识算法

### 四层语义算子

1. **前提层(A)** - SimHash距离：`1 - hamming_distance / 64`
2. **证据层(E)** - MinHash Jaccard：`|E1 ∩ E2| / |E1 ∪ E2|`
3. **推理层(I)** - NCD相似度：`1 - (Z(I1+I2) - min(Z(I1),Z(I2))) / max(Z(I1),Z(I2))`
4. **结论层(C)** - Cosine相似度：`(V1 · V2) / (||V1|| × ||V2||)`

### 收益函数

```
F = s_a × 0.1 + s_e × 0.4 + s_i × 0.2 + s_c × 0.3
U = F × 100 - 25
```

- U > 55：达成ESS共识
- 0 < U < 55：需要继续审计
- U < 0：不一致，拒绝

## 🔬 理论基础

### Stackelberg博弈
- **Leader**：任务发起者（动态角色）
- **Follower**：响应者（其他Agent）
- **均衡**：通过逆向归纳求解
- **存在性**：引用Von Stackelberg 1934经典定理

### 演化稳定策略(ESS)
- 收益函数U作为适应度
- 迭代过程类似复制者动态
- 收敛到ESS表示达成共识

## 📊 实验与评估

### 对照实验

| 实验组 | 配置 | 指标 |
|--------|------|------|
| 单Agent | 无协作 | 基线准确率 |
| 垂直结构 | Solver-Reviewer | 收敛轮次、U值 |
| +记忆 | 启用记忆系统 | 避免重复错误 |
| +招募 | 动态专家招募 | 任务成功率 |

### 评估指标

- **任务成功率**：U > 55的比例
- **收敛速度**：平均达到ESS的轮次
- **协调得分**：通信有效性
- **信任分变化**：Agent信誉演化

## 🛠️ 高级配置

### 集成LLM

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4",
    openai_api_key="sk-xxx"
)

agent = DistributedAgent(
    port=8001,
    model="gpt4",
    role="initiator",
    registry_url="http://127.0.0.1:9000",
    llm=llm  # 启用智能分析
)
```

### 分布式部署

```bash
# 机器1（Registry）
python src/registry_center.py --host 0.0.0.0 --port 9000

# 机器2（Agent A）
python src/agent_node.py --port 8001 --model qwen --role solver \
  --registry http://192.168.1.100:9000

# 机器3（Agent B）
python src/agent_node.py --port 8002 --model deepseek --role reviewer \
  --registry http://192.168.1.100:9000
```

### 记忆持久化

```python
# 导出记忆
agent.memory_manager.export_memory("memory.json")

# 加载记忆
agent.memory_manager.load_memory("memory.json")
```

## 🐛 故障排查

### Registry无法连接
```bash
# 检查Registry是否运行
curl http://127.0.0.1:9000/health

# 检查端口占用
lsof -i :9000
```

### Agent无响应
```bash
# 检查Agent健康状态
curl http://127.0.0.1:8001/health

# 查看Agent日志
# Agent会在终端输出详细日志
```

### 共识无法达成
- 检查Agent角色配置（至少1个Solver + 1个Reviewer）
- 调整ESS阈值（默认55）
- 增加最大轮次（默认5）

## 📚 论文引用

如果您在研究中使用此系统，请引用：

```bibtex
@article{your-paper-2026,
  title={分布式多智能体语义共识系统},
  author={Your Name},
  year={2026}
}
```

## 📄 License

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📧 联系方式

- 作者：Spike
- 邮箱：your-email@example.com
