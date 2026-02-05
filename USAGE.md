# 快速使用指南

## 🚀 5分钟快速开始

### 前置要求

- Python 3.8+
- 终端工具

### 步骤1：安装依赖（1分钟）

```bash
cd /path/to/paper
pip install -r requirements.txt
```

### 步骤2：启动Registry（必需）

**开启服务的人运行：**

```bash
# 终端1
cd src
python registry_center.py
```

看到以下输出表示成功：
```
🌐 Registry Center 启动
   端口: 9000
   📋 等待Agent加入网络...
```

### 步骤3：启动Agent节点（至少2个）

**其他人可以加入网络：**

```bash
# 终端2 - Solver
cd src
python agent_node.py --port 8001 --model qwen --role solver

# 终端3 - Reviewer
cd src
python agent_node.py --port 8002 --model deepseek --role reviewer

# 终端4 - Initiator（可选）
cd src
python agent_node.py --port 8003 --model qwen --role initiator
```

看到以下输出表示成功：
```
✓ 成功加入网络
  网络规模: 2 个Agent
```

### 步骤4：运行示例

```bash
# 终端5
cd examples
python quick_start.py
```

选择 `1` 运行基本工作流程。

---

## 📖 详细使用说明

### 使用场景1：本地测试（单机多Agent）

所有Agent都在同一台机器上：

```bash
# 1. Registry
python src/registry_center.py

# 2-4. 三个Agent
python src/agent_node.py --port 8001 --model qwen --role solver
python src/agent_node.py --port 8002 --model deepseek --role reviewer
python src/agent_node.py --port 8003 --model qwen --role initiator
```

### 使用场景2：分布式部署（多台机器）

**机器A（192.168.1.100）- Registry：**
```bash
python src/registry_center.py --host 0.0.0.0 --port 9000
```

**机器B - Agent 1：**
```bash
python src/agent_node.py --port 8001 --model qwen --role solver \
  --registry http://192.168.1.100:9000
```

**机器C - Agent 2：**
```bash
python src/agent_node.py --port 8002 --model deepseek --role reviewer \
  --registry http://192.168.1.100:9000
```

### 使用场景3：编程方式发起任务

创建 `my_task.py`：

```python
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agent_node import DistributedAgent

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
        task_desc="审核企业贷款（100万额度）",
        task_data={
            "assumptions": "基础评估模型",
            "evidence": ["营业执照", "财务报表"],
            "inference": "初步验证",
            "conclusion": "待定"
        }
    )
    
    # 查看结果
    print(f"\n任务完成！")
    print(f"收益U: {result['result']['utility']:.2f}")
    print(f"轮次: {result['result']['rounds']}")
    print(f"决策: {result['result']['decision']}")

if __name__ == "__main__":
    asyncio.run(main())
```

运行：
```bash
python my_task.py
```

---

## 🔍 验证系统是否正常

### 方法1：运行测试脚本

```bash
cd examples
python test_all.py
```

应该看到：
```
✓ 所有测试通过！系统运行正常。
```

### 方法2：手动检查

#### 检查Registry
```bash
curl http://127.0.0.1:9000/health
```

应该返回：
```json
{
  "status": "healthy",
  "agents_online": 2
}
```

#### 检查Agent
```bash
curl http://127.0.0.1:8001/health
```

应该返回：
```json
{
  "status": "healthy",
  "port": 8001,
  "model": "qwen",
  "role": "solver"
}
```

#### 查看统计
```bash
# Registry统计
curl http://127.0.0.1:9000/stats

# Agent统计
curl http://127.0.0.1:8001/stats
```

---

## 🛠️ 常见问题解决

### 问题1：Registry连接失败

**错误信息：**
```
❌ 无法连接到Registry: Connection refused
```

**解决方法：**
1. 确保Registry已启动
2. 检查端口是否正确（默认9000）
3. 防火墙是否允许访问

### 问题2：Agent无法加入网络

**错误信息：**
```
⚠️  Registry连接失败
```

**解决方法：**
1. 确认Registry地址正确
2. 如果是远程Registry，使用完整URL：`--registry http://IP:PORT`
3. 检查网络连通性：`ping IP`

### 问题3：共识无法达成

**现象：**
任务一直在循环，达到最大轮次才停止

**解决方法：**
1. 确保至少有1个Solver + 1个Reviewer
2. 检查任务数据是否合理
3. 调整ESS阈值（在agent_node.py中修改）

### 问题4：端口被占用

**错误信息：**
```
OSError: [Errno 48] Address already in use
```

**解决方法：**
1. 更换端口：`--port 8005`
2. 或杀死占用进程：
   ```bash
   lsof -i :8001
   kill -9 <PID>
   ```

---

## 📊 监控与日志

### 实时查看日志

所有Agent和Registry都会在终端输出详细日志：

```
[Solver 8001] 正在分析任务...
   收益U = 45.32
   → 继续优化...

[Reviewer 8002] 正在评审方案...
   收益U = 58.67
   ✓ 达成ESS共识！
```

### 查看历史记录

```bash
# 查看Registry的共识历史
curl http://127.0.0.1:9000/consensus_history
```

### 导出记忆

```python
# 在任务完成后
agent.memory_manager.export_memory("memory.json")
```

---

## 🎯 最佳实践

### 1. Agent数量配置

| 任务复杂度 | 推荐配置 |
|-----------|---------|
| 简单 | 1 Solver + 1 Reviewer |
| 中等 | 1 Solver + 2 Reviewers |
| 复杂 | 2 Solvers + 3 Reviewers |

### 2. 角色分配

- **Initiator**：可发起任务，同时具备Solver和Reviewer能力
- **Solver**：专注于提出方案
- **Reviewer**：专注于评审反馈

### 3. 性能优化

- 减少最大轮次以加快响应：在`coordination_engine.py`修改`MAX_ROUNDS`
- 调整ESS阈值以放宽共识条件：修改`ESS_THRESHOLD`
- 增加Agent并行处理能力

---

## 📚 进阶使用

### 集成自定义LLM

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4",
    openai_api_key="sk-xxx",
    openai_api_base="https://api.openai.com/v1"
)

agent = DistributedAgent(
    port=8001,
    model="gpt4",
    role="initiator",
    registry_url="http://127.0.0.1:9000",
    llm=llm
)
```

### 自定义共识算法

修改 `src/consensus/consensus.py`：

```python
# 调整权重
self.w = {'A': 0.2, 'E': 0.3, 'I': 0.2, 'C': 0.3}

# 调整收益参数
self.R = 200  # 提高奖励
self.C = 50   # 提高成本
```

### 添加新的Agent角色

在 `expert_recruiter.py` 中添加：

```python
self.role_templates["your_role"] = {
    "name": "你的角色名",
    "description": "角色描述",
    "required_capability": "capability_name"
}
```

---

## 🎓 学习资源

### 推荐阅读顺序

1. `README.md` - 系统总览
2. `REFACTOR_SUMMARY.md` - 重构详情
3. `USAGE.md`（本文件）- 使用指南
4. 源代码注释 - 实现细节

### 关键文件理解

| 文件 | 作用 | 优先级 |
|------|------|--------|
| `agent_node.py` | 核心Agent逻辑 | ⭐⭐⭐ |
| `coordination_engine.py` | 协调流程 | ⭐⭐⭐ |
| `consensus.py` | 共识算法 | ⭐⭐⭐ |
| `memory.py` | 记忆系统 | ⭐⭐ |
| `expert_recruiter.py` | 招募逻辑 | ⭐⭐ |

---

## 💡 使用技巧

### 技巧1：快速重启

创建脚本 `restart_all.sh`：

```bash
#!/bin/bash
pkill -f "registry_center.py"
pkill -f "agent_node.py"
sleep 2
python src/registry_center.py &
sleep 1
python src/agent_node.py --port 8001 --model qwen --role solver &
python src/agent_node.py --port 8002 --model deepseek --role reviewer &
```

### 技巧2：批量启动Agent

```bash
for port in {8001..8005}; do
    python src/agent_node.py --port $port --model qwen --role solver &
done
```

### 技巧3：使用tmux管理多终端

```bash
# 创建session
tmux new -s agents

# 分屏
Ctrl+b %  # 垂直分屏
Ctrl+b "  # 水平分屏

# 切换
Ctrl+b 方向键
```

---

## ✅ 检查清单

部署前确认：

- [ ] Python版本 >= 3.8
- [ ] 依赖已安装 `pip install -r requirements.txt`
- [ ] Registry已启动
- [ ] 至少2个Agent在线
- [ ] 测试脚本通过 `python examples/test_all.py`
- [ ] 端口未被占用
- [ ] 防火墙允许通信（分布式部署）

---

## 📞 获取帮助

如有问题，请：

1. 查看日志输出
2. 运行测试脚本诊断
3. 查阅文档和代码注释
4. 提交Issue到GitHub

**祝您使用愉快！** 🎉
