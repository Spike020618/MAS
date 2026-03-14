# RAG系统 第4步 - 权重学习集成（与consensus.py整合）

## 📋 目录

1. [概述](#概述)
2. [权重学习机制](#权重学习机制)
3. [核心模块](#核心模块)
4. [API文档](#api文档)
5. [快速开始](#快速开始)
6. [与Consensus集成](#与consensus集成)

---

## 概述

### 什么是第4步？

第4步是RAG系统的**自适应学习层**，提供：
- ✅ 反馈驱动的权重学习
- ✅ AEIC四层权重自动优化
- ✅ 梯度上升学习算法
- ✅ 动量和衰减机制
- ✅ 与consensus.py的集成

### 关键特性

```
完整的权重学习系统:
  1. 反馈收集 - 从任务分配中获取成功评分
  2. 梯度计算 - 基于成功分数计算AEIC梯度
  3. 权重更新 - 使用梯度上升更新权重
  4. 历史追踪 - 保存权重变化历史
  5. 性能监控 - 实时监控学习效果
```

---

## 权重学习机制

### AEIC四层权重

```
权重向量: w = {w_A, w_E, w_I, w_C}

A (Ability) - Agent的基础能力
  ├─ 来源: Agent注册时的历史数据
  └─ 敏感度: 1.2x (最敏感)

E (Evidence) - 历史成功率
  ├─ 来源: 过往任务的成功记录
  └─ 敏感度: 1.0x (标准)

I (Inference) - 任务匹配度
  ├─ 来源: 任务-方案向量相似度
  └─ 敏感度: 0.8x (中等)

C (Conclusion) - 综合评分
  ├─ 来源: 上述三层的综合
  └─ 敏感度: 0.6x (最稳定)
```

### 学习算法

```
梯度上升优化:

1. 计算梯度:
   gradient_i = sign(success_score - 0.5) × |success_score - 0.5| × sensitivity_i
   
   其中:
   - success_score: 任务成功评分 (0.0-1.0)
   - threshold: 0.5 (成功/失败分界点)
   - sensitivity: 权重敏感度 (A>E>I>C)

2. 应用动量:
   velocity_i = momentum × velocity_i + gradient_i
   
   其中:
   - momentum: 0.9 (梯度平滑系数)
   - velocity: 速度累计

3. 更新权重:
   w_i^(t+1) = w_i^t + learning_rate × velocity_i
   
   其中:
   - learning_rate: 0.01 (步长)
   - 范围限制: [0.0, 1.0]

4. 归一化:
   w_i = w_i / Σw_j  (确保和为1)
```

### 学习效果

```
高成功评分 (> 0.5):
  → 增加权重
  → 梯度为正
  → 强化有效的权重

低成功评分 (< 0.5):
  → 减少权重
  → 梯度为负
  → 弱化无效的权重

随时间推移:
  → 权重逐渐收敛到最优值
  → 成功率不断提高
  → 系统自适应改进
```

---

## 核心模块

### 1. WeightLearner

权重学习器核心类

```python
class WeightLearner:
    async def update_weights_from_feedback(
        success_score,
        agent_scores,
        feedback_text
    ) → updated_weights
    
    async def batch_learn(feedback_samples) → final_weights
    
    async def get_convergence_metrics() → metrics
    
    async def save_history(filepath) → None
```

### 2. WeightLearningIntegration

集成器，连接工作流和学习器

```python
class WeightLearningIntegration:
    async def execute_task_with_learning(
        task_request,
        task_embedding
    ) → result
    
    async def process_feedback_with_learning(
        record_id,
        success_score,
        feedback_text
    ) → learning_result
    
    async def get_learning_status() → status
    
    async def evaluate_with_consensus(
        records,
        use_consensus
    ) → consensus_result
```

---

## API文档

### WeightLearner API

#### 更新权重

```python
# 单次反馈学习
updated_weights = await weight_learner.update_weights_from_feedback(
    success_score=0.9,          # 成功评分
    agent_scores={1: 0.9},      # Agent评分
    feedback_text="完成得很好",
    metadata={"task_id": "..."}
)

# 返回
{
    "w_A": 0.28,
    "w_E": 0.26,
    "w_I": 0.23,
    "w_C": 0.23
}
```

#### 批量学习

```python
# 批量处理多个反馈
final_weights = await weight_learner.batch_learn([
    {
        "success_score": 0.95,
        "agent_scores": {...},
        "feedback_text": "..."
    },
    {
        "success_score": 0.85,
        "agent_scores": {...},
        "feedback_text": "..."
    }
])
```

#### 收敛指标

```python
# 获取学习收敛状态
metrics = await weight_learner.get_convergence_metrics()

# 返回
{
    "samples": 10,                      # 样本数
    "weight_variance": 0.0001,         # 权重变化方差
    "weight_change_mean": 0.005,       # 权重变化均值
    "gradient_norm_mean": 0.01,        # 梯度范数均值
    "learning_stability": 0.98         # 学习稳定性 (0-1)
}
```

### WeightLearningIntegration API

#### 执行任务并学习

```python
result = await integration.execute_task_with_learning(
    task_request={
        "task_id": "task_001",
        "task_type": "review",
        "description": "代码审查"
    },
    task_embedding=[...]  # 可选
)

# 返回
{
    "workflow_state": {...},
    "allocated_agents": [1],
    "allocation_decision": "local_with_scoring",
    "success": True
}
```

#### 处理反馈并学习

```python
feedback_result = await integration.process_feedback_with_learning(
    record_id="rec_001",
    success_score=0.95,
    agent_ids=[1],
    feedback_text="审查质量优秀",
    agent_scores={1: 0.95}
)

# 返回
{
    "success": True,
    "updated_weights": {...},
    "performance_metrics": {
        "total_tasks": 5,
        "successful_tasks": 4,
        "avg_success_score": 0.91,
        "learning_iterations": 5
    }
}
```

#### 获取学习状态

```python
status = await integration.get_learning_status()

# 返回
{
    "current_weights": {...},
    "performance_metrics": {...},
    "learner_stats": {...},
    "convergence": {...},
    "system_health": {
        "total_tasks": 5,
        "successful_tasks": 4,
        "success_rate": 0.8,
        "status": "good"
    }
}
```

---

## 快速开始

### 1. 初始化

```python
from mas.rag import (
    LocalRAGDatabase,
    RAGWorkflow,
    WeightLearningIntegration
)

# 创建基础设施
rag_db = LocalRAGDatabase()
await rag_db.initialize()

workflow = RAGWorkflow(rag_db)

# 创建学习集成
learner = WeightLearningIntegration(
    rag_database=rag_db,
    rag_workflow=workflow,
    learning_rate=0.01
)
```

### 2. 执行任务和反馈

```python
# 执行任务
result = await learner.execute_task_with_learning(
    task_request={
        "task_id": "task_001",
        "task_type": "review",
        "description": "代码审查"
    }
)

# 处理反馈（自动触发权重学习）
feedback = await learner.process_feedback_with_learning(
    record_id="rec_001",
    success_score=0.95,
    agent_ids=result["allocated_agents"],
    feedback_text="完成得很好"
)
```

### 3. 监控学习

```python
# 获取学习状态
status = await learner.get_learning_status()

print(f"当前权重: {status['current_weights']}")
print(f"成功率: {status['system_health']['success_rate']:.2%}")
print(f"学习稳定性: {status['convergence']['learning_stability']:.4f}")
```

---

## 与Consensus集成

### 评估共识

```python
# 使用consensus.py评估记录
result = await learner.evaluate_with_consensus(
    records=[
        {
            "w_A": 0.25, "w_E": 0.25, "w_I": 0.25, "w_C": 0.25,
            "success_score": 0.95
        }
    ],
    use_consensus=True
)

# 返回
{
    "success": True,
    "consensus_result": {
        "consensus_energy": 0.05,
        "similarities": [[1.0, 0.92], [0.92, 1.0]],
        ...
    }
}
```

### 权重与共识

```
权重学习 (AEIC) ↔ Consensus (博弈论)

权重学习:
  - 单智能体学习
  - 基于个体反馈
  - 速度快，收敛快

Consensus:
  - 多节点共识
  - 基于节点比较
  - 全局最优性更强

集成优势:
  - 权重学习快速适应
  - Consensus保证一致性
  - 两者相辅相成
```

---

## 性能指标

| 指标 | 含义 | 范围 |
|------|------|------|
| weight_variance | 权重变化方差 | [0, 0.1] |
| weight_change_mean | 权重变化均值 | [0, 0.1] |
| gradient_norm_mean | 梯度范数均值 | [0, 0.2] |
| learning_stability | 学习稳定性 | [0, 1] |

### 收敛判断

```
学习稳定性 > 0.95 → 已收敛
学习稳定性 > 0.80 → 接近收敛
学习稳定性 > 0.50 → 学习中
学习稳定性 < 0.50 → 不稳定
```

---

## 文件结构

```
mas/rag/
├── weight_learner.py              (权重学习器)
├── weight_learning_integration.py (学习集成)
├── demo_step4.py                  (演示脚本)
└── __init__.py                    (已更新)
```

---

**版本**: Step 4 Final  
**完成日期**: 2026-03-14  
**代码行数**: ~1400行  
