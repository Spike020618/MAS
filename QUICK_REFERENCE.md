# 🚀 RAG系统快速参考卡

## ⚡ 30秒快速开始

### 最快方式：运行完整演示
```bash
cd /Users/spike/code/MAS
python -m mas.rag.demo_step5
```

**会看到**: 三种算法的性能对比（约需1-2分钟）

---

## 📋 核心代码片段

### 1️⃣ 初始化（复制粘贴）

```python
import asyncio
from mas.rag import LocalRAGDatabase, RAGWorkflow, WeightLearningIntegration

async def setup():
    # 创建数据库
    rag_db = LocalRAGDatabase(storage_path="./rag_storage")
    await rag_db.initialize()
    
    # 创建工作流
    workflow = RAGWorkflow(rag_db)
    
    # 创建学习集成
    learner = WeightLearningIntegration(
        rag_database=rag_db,
        rag_workflow=workflow,
        learning_rate=0.01
    )
    
    return rag_db, workflow, learner

# 运行
rag_db, workflow, learner = asyncio.run(setup())
```

### 2️⃣ 注册Agent（复制粘贴）

```python
async def register_agents(rag_db):
    agents = [
        {"agent_id": 1, "name": "Agent1", "task_types": ["review"], "success_rate": 0.85},
        {"agent_id": 2, "name": "Agent2", "task_types": ["review"], "success_rate": 0.90},
    ]
    
    for agent in agents:
        await rag_db.register_agent(**agent)

asyncio.run(register_agents(rag_db))
```

### 3️⃣ 分配任务并学习（复制粘贴）

```python
async def allocate_and_learn():
    # 执行任务
    result = await learner.execute_task_with_learning(
        task_request={
            "task_id": "task_001",
            "task_type": "review",
            "description": "代码审查",
        }
    )
    
    # 处理反馈（自动学习！）
    feedback = await learner.process_feedback_with_learning(
        record_id="rec_001",
        success_score=0.95,
        agent_ids=result.get("allocated_agents", []),
    )
    
    print(f"权重已更新: {feedback['updated_weights']}")

asyncio.run(allocate_and_learn())
```

### 4️⃣ 查看学习效果（复制粘贴）

```python
async def check_learning():
    status = await learner.get_learning_status()
    
    print(f"当前权重: {status['current_weights']}")
    print(f"成功率: {status['system_health']['success_rate']:.2%}")
    print(f"学习稳定性: {status['convergence']['learning_stability']:.4f}")

asyncio.run(check_learning())
```

---

## 🎯 不同场景的使用

| 场景 | 命令 | 说明 |
|------|------|------|
| **快速演示** | `python -m mas.rag.demo_step5` | 看三算法对比 |
| **示例1** | `python rag_integration_example.py --example 1` | 简单任务分配 |
| **示例2** | `python rag_integration_example.py --example 2` | 任务+学习 |
| **示例3** | `python rag_integration_example.py --example 3` | 对比实验 |
| **示例4** | `python rag_integration_example.py --example 4` | 多Agent协调 |

---

## 📊 性能预期

运行完全演示后，你会看到：

```
贪心基线:    成功率 50-60%  ⏱️  最快 (<1ms)
RAG检索:     成功率 70-80%  ⏱️  中等 (20-30ms)
RAG+学习:    成功率 80-90%  ⏱️  略慢 (25-35ms) ⭐ 最优
```

---

## 💾 关键参数配置

### 数据库配置
```python
rag_db = LocalRAGDatabase(
    storage_path="./rag_storage",        # 存储路径
    embedding_model="local_hash",        # 向量模型
    embedding_dimension=1536,            # 向量维度
)
```

### 学习器配置
```python
learner = WeightLearningIntegration(
    rag_database=rag_db,
    rag_workflow=workflow,
    learning_rate=0.01,                  # 学习率：0.001-0.1
    initial_weights={                    # 初始权重
        "w_A": 0.25,  # Ability
        "w_E": 0.25,  # Evidence
        "w_I": 0.25,  # Inference
        "w_C": 0.25,  # Conclusion
    }
)
```

### 实验配置
```python
runner = ExperimentRunner(rag_db)
results = await runner.run_experiment(
    num_agents=5,      # Agent数量
    num_tasks=50,      # 任务数量
    seed=42,           # 随机种子
)
```

---

## 🔍 调试和监控

### 查看权重历史
```python
history = await learner.weight_learner.get_weight_history()
for record in history[-3:]:  # 看最后3条
    print(f"权重: {record['weights']}")
```

### 查看收敛指标
```python
convergence = await learner.weight_learner.get_convergence_metrics()
print(f"学习稳定性: {convergence['learning_stability']:.4f}")  # 越接近1越好
```

### 保存结果
```python
await learner.save_learning_history("./learning_history.json")
```

---

## 📂 文件位置速查

| 类别 | 位置 |
|------|------|
| **Python模块** | `/Users/spike/code/MAS/mas/rag/` |
| **文档** | `/Users/spike/code/MAS/RAG_*.md` |
| **示例代码** | `/Users/spike/code/MAS/rag_integration_example.py` |
| **完成汇总** | `/Users/spike/code/MAS/STEP*_COMPLETION.md` |

---

## ✅ 验证清单

- [ ] 可以导入RAG模块：`from mas.rag import *`
- [ ] 可以运行demo_step5：`python -m mas.rag.demo_step5`
- [ ] 可以运行集成示例：`python rag_integration_example.py`
- [ ] 理解了三步工作流：初始化 → 分配任务 → 处理反馈+学习
- [ ] 对比实验显示RAG+学习性能最优

---

## 🚨 常见错误及解决

### 错误1: ModuleNotFoundError
```python
# ❌ 错误
from mas.rag import LocalRAGDatabase

# ✅ 正确
import sys
sys.path.insert(0, '/Users/spike/code/MAS')
from mas.rag import LocalRAGDatabase
```

### 错误2: 路径找不到
```python
# ✅ 确保使用绝对路径
rag_db = LocalRAGDatabase(
    storage_path="/Users/spike/code/MAS/rag_storage"
)
```

### 错误3: 异步问题
```python
# ✅ 所有async函数必须用asyncio.run()
result = asyncio.run(my_async_function())
```

---

## 📞 遇到问题？

1. **查看演示脚本**: `/Users/spike/code/MAS/mas/rag/demo_step*.py`
2. **读步骤指南**: `/Users/spike/code/MAS/RAG_STEP*_GUIDE.md`
3. **参考集成示例**: `/Users/spike/code/MAS/rag_integration_example.py`
4. **查看完成总结**: `/Users/spike/code/MAS/STEP*_COMPLETION.md`

---

## 🎓 推荐学习顺序

1. **第0步** (5分钟): 运行 `python -m mas.rag.demo_step5` 看演示
2. **第1步** (10分钟): 阅读 `RAG_STEP1_GUIDE.md` 了解存储
3. **第2步** (10分钟): 阅读 `RAG_STEP2_GUIDE.md` 了解工作流
4. **第3步** (10分钟): 阅读 `RAG_STEP3_GUIDE.md` 了解通信
5. **第4步** (10分钟): 阅读 `RAG_STEP4_GUIDE.md` 了解学习
6. **第5步** (10分钟): 阅读 `RAG_STEP5_GUIDE.md` 了解对比
7. **集成示例** (15分钟): 运行 `python rag_integration_example.py --example 1,2,3,4`
8. **自己尝试** (30分钟): 修改示例代码进行实验

---

## 🎉 预期结果

完成以上步骤后，你将能够：

✅ 理解完整的RAG系统架构  
✅ 使用RAG进行智能任务分配  
✅ 通过反馈自动优化权重  
✅ 对比不同算法的性能  
✅ 集成到自己的系统中  
✅ 部署到生产环境  

---

**祝你使用愉快！🚀**

如有任何问题，参考上面的"遇到问题?"章节。
