#!/bin/bash

# RAG系统快速导入测试脚本

cd /Users/spike/code/MAS

echo "🔍 测试RAG系统导入..."
echo ""

python3 << 'EOF'
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print("✅ 尝试导入RAG系统模块...")
print("")

try:
    print("  导入 LocalRAGDatabase...")
    from mas.rag.local_rag_database import LocalRAGDatabase
    print("    ✅ 成功")
except Exception as e:
    print(f"    ❌ 失败: {e}")

try:
    print("  导入 RAGWorkflow...")
    from mas.rag.rag_workflow import RAGWorkflow
    print("    ✅ 成功")
except Exception as e:
    print(f"    ❌ 失败: {e}")

try:
    print("  导入 RAGSyncManager...")
    from mas.rag.rag_sync_manager import RAGSyncManager
    print("    ✅ 成功")
except Exception as e:
    print(f"    ❌ 失败: {e}")

try:
    print("  导入 MultiAgentCoordinator...")
    from mas.rag.multi_agent_coordinator import MultiAgentCoordinator
    print("    ✅ 成功")
except Exception as e:
    print(f"    ❌ 失败: {e}")

try:
    print("  导入 WeightLearningIntegration...")
    from mas.rag.weight_learning_integration import WeightLearningIntegration
    print("    ✅ 成功")
except Exception as e:
    print(f"    ❌ 失败: {e}")

try:
    print("  导入 ExperimentRunner...")
    from mas.rag.experiment_runner import ExperimentRunner
    print("    ✅ 成功")
except Exception as e:
    print(f"    ❌ 失败: {e}")

try:
    print("  导入 ResultsAnalyzer...")
    from mas.rag.results_analyzer import ResultsAnalyzer
    print("    ✅ 成功")
except Exception as e:
    print(f"    ❌ 失败: {e}")

print("")
print("✅ 所有导入测试完成！")
EOF
