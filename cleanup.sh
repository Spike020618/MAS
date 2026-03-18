#!/bin/bash

# 清理多余文件脚本

cd /Users/spike/code/MAS

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                 🧹 清理多余文件和代码                                     ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 删除 RAG 存储目录
echo "删除 RAG 存储目录..."
rm -rf rag_storage_milvus 2>/dev/null || true
rm -rf rag_storage_step5 2>/dev/null || true
echo "✅ 完成"
echo ""

# 删除 Docker 卷数据
echo "删除 Docker 卷数据..."
rm -rf volumes 2>/dev/null || true
echo "✅ 完成"
echo ""

# 删除旧的演示脚本
echo "删除旧的演示脚本..."
rm -f mas/rag/demo_step1.py 2>/dev/null || true
rm -f mas/rag/demo_step1_real_rag.py 2>/dev/null || true
rm -f mas/rag/demo_step2.py 2>/dev/null || true
rm -f mas/rag/demo_step3.py 2>/dev/null || true
rm -f mas/rag/demo_step4.py 2>/dev/null || true
rm -f mas/rag/demo_step5.py 2>/dev/null || true
echo "✅ 完成"
echo ""

# 删除旧的权重学习和实验模块
echo "删除旧的权重学习模块..."
rm -f mas/rag/weight_learner.py 2>/dev/null || true
rm -f mas/rag/weight_learning_integration.py 2>/dev/null || true
rm -f mas/rag/experiment_runner.py 2>/dev/null || true
rm -f mas/rag/results_analyzer.py 2>/dev/null || true
echo "✅ 完成"
echo ""

# 删除旧的工作流模块
echo "删除旧的工作流模块..."
rm -f mas/rag/rag_workflow.py 2>/dev/null || true
rm -f mas/rag/rag_sync_manager.py 2>/dev/null || true
rm -f mas/rag/workflow_nodes.py 2>/dev/null || true
rm -f mas/rag/workflow_state.py 2>/dev/null || true
rm -f mas/rag/multi_agent_coordinator.py 2>/dev/null || true
echo "✅ 完成"
echo ""

# 删除其他不相关的模块
echo "删除其他不相关的模块..."
rm -f mas/rag/faiss_index.py 2>/dev/null || true
rm -f mas/rag/local_rag_database.py 2>/dev/null || true
rm -f mas/rag/greedy_baseline.py 2>/dev/null || true
rm -f mas/rag/dataset_generator.py 2>/dev/null || true
rm -f mas/rag/agent_message.py 2>/dev/null || true
echo "✅ 完成"
echo ""

# 删除不相关的目录
echo "删除不相关的目录..."
rm -rf mas/eval 2>/dev/null || true
rm -rf mas/evalscope 2>/dev/null || true
rm -rf mas/consensus 2>/dev/null || true
rm -rf mas/data 2>/dev/null || true
echo "✅ 完成"
echo ""

# 删除不相关的主模块文件
echo "删除不相关的主模块文件..."
rm -f mas/agent_node.py 2>/dev/null || true
rm -f mas/coordination_engine.py 2>/dev/null || true
rm -f mas/expert_recruiter.py 2>/dev/null || true
rm -f mas/task_planner.py 2>/dev/null || true
rm -f mas/registry_center.py 2>/dev/null || true
rm -f mas/memory.py 2>/dev/null || true
echo "✅ 完成"
echo ""

# 删除旧的集成示例
echo "删除旧的文件..."
rm -f rag_integration_example.py 2>/dev/null || true
rm -f start.py 2>/dev/null || true
echo "✅ 完成"
echo ""

# 清理 Python 缓存
echo "清理 Python 缓存..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
echo "✅ 完成"
echo ""

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                      ✅ 清理完成！                                        ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

echo "📁 项目结构已精简，现在只保留核心代码："
echo ""
echo "mas/rag/  核心模块:"
echo "  ├─ config.py           - 配置管理"
echo "  ├─ embedding_model.py   - DashScope 向量化"
echo "  ├─ milvus_db.py        - Milvus 驱动"
echo "  ├─ rag_database.py      - RAG 数据库（主接口）"
echo "  └─ demo_rag_milvus.py   - 演示脚本"
echo ""
echo "🚀 现在可以运行演示了："
echo "  ./clean_and_restart.sh"
echo ""
