# 🎯 RAG系统安装后的下一步

恭喜！你的 Milvus + DashScope RAG 系统已成功安装！🚀

## 现在的状态

✅ Docker 容器已启动：
- milvus-etcd ✓
- milvus-minio ✓
- milvus-standalone ✓
- milvus-attu ✓

✅ Python 依赖已安装：
- pymilvus ✓
- httpx ✓
- numpy ✓

✅ 网络配置已完成

---

## 📋 下一步操作（按顺序）

### 第1步：配置 DashScope API（5分钟）

```bash
# 进入项目目录
cd /Users/spike/code/MAS

# 创建 .env 文件（如果还没有）
cp .env.example .env

# 编辑 .env 文件
vi .env
```

**确保以下配置正确**：
```
DASHSCOPE_API_KEY=sk-f771855105fe43b28584a0f4d68fb5e9
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=text-embedding-v4
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

**或者直接运行这个命令设置环境变量**（不用编辑文件）：
```bash
export DASHSCOPE_API_KEY="sk-f771855105fe43b28584a0f4d68fb5e9"
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_MODEL="text-embedding-v4"
```

---

### 第2步：验证系统配置（2分钟）

```bash
cd /Users/spike/code/MAS

python3 << 'EOF'
import sys
sys.path.insert(0, '/Users/spike/code/MAS')

print("🔍 验证系统配置...\n")

# 1. 检查 Milvus 连接
print("1️⃣  检查 Milvus 连接...")
try:
    from pymilvus import connections
    connections.connect(host="localhost", port=19530)
    print("   ✅ Milvus 连接成功\n")
    connections.disconnect(alias="default")
except Exception as e:
    print(f"   ❌ Milvus 连接失败: {e}\n")
    sys.exit(1)

# 2. 检查 DashScope 配置
print("2️⃣  检查 DashScope 配置...")
try:
    from mas.rag.config import RAGConfig
    if RAGConfig.validate():
        print("   ✅ DashScope 配置有效\n")
    else:
        print("   ❌ DashScope 配置无效\n")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ 配置检查失败: {e}\n")
    sys.exit(1)

# 3. 检查 RAG 模块导入
print("3️⃣  检查 RAG 模块...")
try:
    from mas.rag.rag_database import RAGDatabase
    from mas.rag.embedding_model import EmbeddingModel
    from mas.rag.milvus_db import MilvusDatabase
    print("   ✅ 所有 RAG 模块导入成功\n")
except Exception as e:
    print(f"   ❌ 模块导入失败: {e}\n")
    sys.exit(1)

print("=" * 50)
print("✅ 所有检查通过！系统已就绪！")
print("=" * 50)
EOF
```

---

### 第3步：运行演示脚本（5-10分钟）

这是最重要的一步！会看到完整的 RAG 系统工作。

```bash
python3 mas/rag/demo_rag_milvus.py
```

**预期看到的输出**：
```
🚀 RAG 系统演示 - 使用 Milvus + DashScope (生产级)

[步骤1] 初始化 RAG 数据库...
✓ RAG 数据库初始化成功

[步骤2] 注册 Agent...
✓ 注册了 3 个 Agent

[步骤3] 添加任务（自动生成语义向量）...
✓ 添加了 5 个任务

[步骤4] 演示语义相似度搜索...
搜索：'我需要进行代码审查和质量检查'

找到 5 个相似任务:
  [1] 🟢 task_001: 代码审查：检查代码质量和安全性
      相似度: 0.8542
  [2] 🟢 task_002: 代码评审：验证代码功能和性能
      相似度: 0.8234
```

---

### 第4步：打开 Attu 管理界面（3分钟）

这是向量数据库的可视化管理工具。

```bash
# 在浏览器中打开
open http://localhost:8000

# 或手动输入：
# http://localhost:8000
```

**在 Attu 中你可以**：
1. 连接到 `standalone:19530`
2. 查看 `rag_system` 数据库
3. 查看 `tasks` 集合中的所有向量
4. 搜索向量
5. 查看数据库统计

---

### 第5步：验证数据已存储（2分钟）

演示脚本会自动在 Milvus 中创建数据。你可以在 Attu 中查看。

在 Attu 中：
1. 选择数据库：`rag_system`
2. 选择集合：`tasks`
3. 点击 "Query" 查询数据
4. 应该看到 5 个任务的向量数据

---

## 🎓 完整工作流理解

```
你的代码
   ↓
RAGDatabase (我们创建的)
   ↓
EmbeddingModel (使用 DashScope API 生成向量)
   ↓
向量 (1536 维，语义向量)
   ↓
MilvusDatabase (存储到 Milvus)
   ↓
Milvus Docker 容器
   ↓
持久化存储到磁盘
```

**关键特性**：
- 自动向量化（DashScope API）
- 自动存储（Milvus）
- 自动搜索（向量相似度）
- 可视化监控（Attu）

---

## 🚀 快速命令总结

```bash
# 进入项目目录
cd /Users/spike/code/MAS

# 设置环境变量（可选，如果没用.env文件）
export DASHSCOPE_API_KEY="sk-f771855105fe43b28584a0f4d68fb5e9"

# 验证系统
python3 << 'EOF'
import sys
sys.path.insert(0, '/Users/spike/code/MAS')
from pymilvus import connections
connections.connect(host="localhost", port=19530)
print("✅ Milvus 已连接")
EOF

# 运行演示（最重要！）
python3 mas/rag/demo_rag_milvus.py

# 打开 Attu（在浏览器中）
# http://localhost:8000
```

---

## 📊 系统检查清单

完成上面的所有步骤后，检查：

- [ ] .env 文件已配置
- [ ] 环境变量已设置
- [ ] Milvus 连接验证成功
- [ ] DashScope 配置验证成功
- [ ] 演示脚本运行成功
- [ ] Attu 能打开 (http://localhost:8000)
- [ ] 能在 Attu 中看到向量数据

所有项都打勾？恭喜！你的生产级 RAG 系统已就绪！🎉

---

## 💡 接下来可以做什么？

### 选项1：在自己的代码中使用 RAG

```python
import asyncio
from mas.rag.rag_database import RAGDatabase

async def main():
    # 初始化
    rag = RAGDatabase()
    await rag.initialize()
    
    # 添加任务
    await rag.add_task(
        task_id="my_task",
        task_type="review",
        description="我的任务",
    )
    
    # 搜索
    results = await rag.search_tasks("搜索文本", top_k=5)
    
    await rag.close()

asyncio.run(main())
```

### 选项2：集成到现有项目

修改你的项目代码，使用 `RAGDatabase` 替代原来的 `LocalRAGDatabase`。

### 选项3：监控和优化

使用 Attu 监控向量数据库的性能。

---

## 🔧 故障排除

### 问题1：连接 Milvus 失败

```bash
# 检查容器状态
docker-compose -f docker-compose-milvus.yml ps

# 查看日志
docker logs milvus-standalone

# 重启服务
docker-compose -f docker-compose-milvus.yml restart
```

### 问题2：DashScope 错误

```
Error: DashScope API error: 401
```

**原因**：API 密钥错误或过期

**解决**：
1. 检查 .env 文件中的 API 密钥
2. 确保环境变量已设置
3. 访问 DashScope 官网验证密钥有效性

### 问题3：演示脚本卡住

通常是等待 DashScope API 响应。稍等片刻或检查网络连接。

---

## 📚 更多文档

- **快速开始**：`QUICKSTART_MILVUS.md`
- **完整指南**：`RAG_MILVUS_DASHSCOPE_GUIDE.md`
- **升级总结**：`RAG_UPGRADE_COMPLETE.md`

---

## 🎯 核心命令速查

```bash
# 1. 验证配置
python3 << 'EOF'
from mas.rag.config import RAGConfig
RAGConfig.validate()
EOF

# 2. 运行演示
python3 mas/rag/demo_rag_milvus.py

# 3. 打开 Attu
open http://localhost:8000

# 4. 查看 Milvus 日志
docker logs -f milvus-standalone

# 5. 停止服务
docker-compose -f docker-compose-milvus.yml down

# 6. 重启服务
docker-compose -f docker-compose-milvus.yml up -d
```

---

**现在就开始第1步吧！** 🚀
