"""
RAG 系统 - 任务分配向量数据库（本地FAISS版）

第1步: 基础RAG存储 ✅
  - LocalRAGDatabase (本地存储+FAISS索引)
  - EmbeddingModel (文本向量化)
  - FAISSIndex (向量索引)

第2步: LangGraph工作流 ✅
  - RAGWorkflow (任务分配工作流)
  - WorkflowState (工作流状态)
  - WorkflowNodes (工作流节点)

第3步: 跨Agent通信 ✅
  - RAGSyncManager (同步管理器)
  - AgentMessage (通信消息)
  - MultiAgentCoordinator (多Agent协调器)

第4步: 权重学习集成 ✅
  - WeightLearner (权重学习器)
  - WeightLearningIntegration (学习集成)

第5步: 对比实验 ✅
  - GreedyBaseline (贪心基线)
  - DatasetGenerator (数据集生成)
  - ExperimentRunner (实验运行)
  - ResultsAnalyzer (结果分析)
"""

__version__ = "5.0.0"

from .local_rag_database import LocalRAGDatabase
from .embedding_model import EmbeddingModel
from .faiss_index import FAISSIndex
from .rag_workflow import RAGWorkflow
from .workflow_state import WorkflowState, AllocationScore
from .workflow_nodes import WorkflowNodes
from .rag_sync_manager import RAGSyncManager
from .agent_message import (
    AgentMessage,
    MessageType,
    MessageStatus,
    TaskRequestMessage,
    TaskResponseMessage,
    FeedbackMessage,
)
from .multi_agent_coordinator import MultiAgentCoordinator
from .weight_learner import WeightLearner, WeightSnapshot
from .weight_learning_integration import WeightLearningIntegration
from .greedy_baseline import GreedyBaseline
from .dataset_generator import DatasetGenerator
from .experiment_runner import ExperimentRunner
from .results_analyzer import ResultsAnalyzer

__all__ = [
    # Step 1
    "LocalRAGDatabase",
    "EmbeddingModel",
    "FAISSIndex",
    # Step 2
    "RAGWorkflow",
    "WorkflowState",
    "AllocationScore",
    "WorkflowNodes",
    # Step 3
    "RAGSyncManager",
    "AgentMessage",
    "MessageType",
    "MessageStatus",
    "TaskRequestMessage",
    "TaskResponseMessage",
    "FeedbackMessage",
    "MultiAgentCoordinator",
    # Step 4
    "WeightLearner",
    "WeightSnapshot",
    "WeightLearningIntegration",
    # Step 5
    "GreedyBaseline",
    "DatasetGenerator",
    "ExperimentRunner",
    "ResultsAnalyzer",
]
