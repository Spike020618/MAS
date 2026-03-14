# RAG系统 第3步 - 跨Agent通信（广播+收集机制）

## 📋 目录

1. [概述](#概述)
2. [通信架构](#通信架构)
3. [核心模块](#核心模块)
4. [API文档](#api文档)
5. [快速开始](#快速开始)
6. [消息类型](#消息类型)

---

## 概述

### 什么是第3步？

第3步是RAG系统的**多Agent通信层**，提供：
- ✅ 跨Agent通信协议
- ✅ 广播机制（一对多通信）
- ✅ 收集机制（多对一通信）
- ✅ 异步消息队列
- ✅ Agent目录管理

### 关键特性

```
完整的通信架构:
  1. 广播机制 - 发送消息给多个Agent
  2. 收集机制 - 等待并收集响应
  3. 异步队列 - 异步消息处理
  4. 消息追踪 - 追踪待响应消息
  5. Agent目录 - 维护Agent信息
```

---

## 通信架构

### 广播-收集模式

```
主Agent (协调者)
  │
  ├─→ [广播] → 任务请求消息
  │
  ├→ Agent1 (审查专家)
  │   ├─ 处理任务
  │   └─→ [响应] → 方案1 (成功率:85%)
  │
  ├→ Agent2 (规划专家)
  │   ├─ 处理任务
  │   └─→ [响应] → 方案2 (成功率:80%)
  │
  └→ Agent3 (开发专家)
      ├─ 处理任务
      └─→ [响应] → 方案3 (成功率:90%)

[收集阶段]
  ↓
主Agent收集所有响应 (最多5秒)
  ↓
分析并选择最佳方案 (Agent3)
  ↓
返回最终分配结果
```

### 消息流

```
发送方                          接收方
  │
  ├─ 创建消息
  │   ├─ message_id (唯一ID)
  │   ├─ message_type (消息类型)
  │   ├─ payload (数据内容)
  │   └─ priority (优先级)
  │
  ├─ 添加到发送队列
  │
  ├─→ 网络传输 (模拟)
  │                          ├─ 接收消息
  │                          ├─ 验证格式
  │                          ├─ 处理内容
  │                          └─ 生成响应
  │
  │                          ├─ 创建响应消息
  │                          │  ├─ reply_to (原消息ID)
  │                          │  ├─ payload (结果)
  │                          │  └─ timestamp
  │                          │
  │                          ├─ 添加到响应队列
  │
  ←───────────────────────
  │
  └─ 收集响应
     ├─ 等待回复
     ├─ 超时管理
     └─ 响应聚合
```

---

## 核心模块

### 1. AgentMessage

标准化的Agent通信消息

```python
@dataclass
class AgentMessage:
    message_id: str           # 唯一ID
    message_type: MessageType # 消息类型
    sender_id: int           # 发送Agent
    receiver_id: Optional[int] # 接收Agent
    receiver_ids: List[int]   # 多播列表
    payload: Dict            # 消息数据
    timestamp: float          # 发送时间
    status: MessageStatus    # 消息状态
    priority: int            # 优先级 (0-10)
    ttl: float               # 生存时间 (秒)
    reply_to: Optional[str]  # 回复的消息ID
```

### 2. RAGSyncManager

同步管理器，负责消息的发送和接收

```python
class RAGSyncManager:
    # 广播机制
    async def broadcast_task_request(task, targets) → message_id
    async def broadcast_to_all(message) → message_id
    
    # 收集机制
    async def collect_responses(message_id, timeout) → [messages]
    async def wait_for_response(message_id, timeout) → message
    
    # Agent目录
    async def register_agent(agent_id, name, task_types)
    async def list_agents_for_task(task_type) → [agent_ids]
    
    # 消息处理
    async def process_incoming_message(message)
    async def handle_task_broadcast(message)
    async def handle_task_response(message)
```

### 3. MultiAgentCoordinator

多Agent协调器，整合工作流和同步管理器

```python
class MultiAgentCoordinator:
    # 主方法
    async def allocate_task_with_sync(task_request) → result
    
    # 远程处理
    async def _handle_remote_allocation(task_request) → result
    async def _analyze_remote_responses(responses) → best_solution
    
    # 反馈
    async def send_feedback_to_agent(agent_id, score)
    
    # 监控
    async def get_system_status() → stats
    async def health_check() → health_info
```

---

## API文档

### RAGSyncManager API

#### 广播消息

```python
# 广播任务请求给特定Agent列表
message_id = await sync_manager.broadcast_task_request(
    task_request={"task_id": "...", "description": "..."},
    target_agents=[1, 2, 3]  # None表示广播给所有
)
```

#### 收集响应

```python
# 等待响应
responses = await sync_manager.collect_responses(
    message_id=message_id,
    timeout=5.0,          # 超时时间（秒）
    min_responses=1       # 最少响应数
)

# 等待单个响应
response = await sync_manager.wait_for_response(
    message_id=message_id,
    timeout=10.0
)
```

#### Agent目录

```python
# 注册Agent
await sync_manager.register_agent(
    agent_id=1,
    agent_name="ReviewExpert",
    task_types=["review", "analysis"],
    success_rate=0.85
)

# 获取支持特定任务的Agent
agents = await sync_manager.list_agents_for_task("review")
# 返回: [1, 3]
```

### MultiAgentCoordinator API

#### 任务分配

```python
# 执行多Agent任务分配
result = await coordinator.allocate_task_with_sync(
    task_request={
        "task_id": "task_001",
        "task_type": "review",
        "description": "代码审查",
    },
    task_embedding=[...],      # 可选
    enable_remote=True         # 是否启用远程请求
)

# 返回
{
    "type": "local" | "remote",
    "selected_agents": [1, 2],
    "solution": {...},
    "timestamp": 1234567890.0
}
```

#### 反馈

```python
# 发送反馈给Agent
await coordinator.send_feedback_to_agent(
    target_agent_id=1,
    record_id="rec_001",
    success_score=0.95,
    feedback_text="完成得很好"
)
```

---

## 快速开始

### 1. 初始化同步管理器

```python
from mas.rag import RAGSyncManager

sync_manager = RAGSyncManager(
    agent_id=0,
    agent_name="CoordinatorAgent"
)
```

### 2. 注册Agent

```python
# 注册所有Agent
await sync_manager.register_agent(
    agent_id=1,
    agent_name="ReviewExpert",
    task_types=["review", "analysis"],
    success_rate=0.85
)
```

### 3. 广播任务请求

```python
# 广播给支持"review"的Agent
agents = await sync_manager.list_agents_for_task("review")
message_id = await sync_manager.broadcast_task_request(
    task_request={"task_id": "...", "description": "..."},
    target_agents=agents
)
```

### 4. 收集响应

```python
# 等待响应
responses = await sync_manager.collect_responses(
    message_id=message_id,
    timeout=5.0,
    min_responses=1
)

# 分析响应
for response in responses:
    print(f"Agent {response.sender_id}: {response.payload}")
```

### 5. 多Agent协调

```python
from mas.rag import MultiAgentCoordinator

coordinator = MultiAgentCoordinator(
    agent_id=0,
    agent_name="CoordinatorAgent",
    rag_workflow=workflow,
    sync_manager=sync_manager
)

# 执行任务分配
result = await coordinator.allocate_task_with_sync(
    task_request=task_request,
    enable_remote=True
)
```

---

## 消息类型

### TaskRequestMessage (任务请求)

```python
TaskRequestMessage(
    sender_id=0,
    sender_name="Coordinator",
    task_request={
        "task_id": "task_001",
        "task_type": "review",
        "description": "代码审查"
    }
)
```

### TaskResponseMessage (任务响应)

```python
TaskResponseMessage(
    sender_id=1,
    sender_name="ReviewExpert",
    reply_to=message_id,
    response_data={
        "solution": "使用静态分析工具",
        "success_rate": 0.85,
        "confidence": 0.9
    }
)
```

### FeedbackMessage (反馈)

```python
FeedbackMessage(
    sender_id=0,
    sender_name="Coordinator",
    target_agent_id=1,
    record_id="rec_001",
    success_score=0.95,
    feedback_text="完成得很好"
)
```

---

## 性能指标

| 操作 | 时间 |
|------|------|
| 创建消息 | <1ms |
| 广播消息 | <5ms |
| 收集响应 | ~5s (含超时) |
| 处理消息 | <2ms |

### 可扩展性

- ✅ 支持数百个Agent
- ✅ 异步消息处理
- ✅ 消息优先级队列
- ✅ 自动超时管理

---

## 文件结构

```
mas/rag/
├── agent_message.py         (消息定义)
├── rag_sync_manager.py      (同步管理器)
├── multi_agent_coordinator.py (多Agent协调)
├── demo_step3.py            (演示脚本)
└── __init__.py              (已更新)
```

---

**版本**: Step 3 Final  
**完成日期**: 2026-03-14  
**代码行数**: ~1200行  
