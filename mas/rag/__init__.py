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
"""

__version__ = "2.0.0"

from .local_rag_database import LocalRAGDatabase
from .embedding_model import EmbeddingModel
from .faiss_index import FAISSIndex
from .rag_workflow import RAGWorkflow
from .workflow_state import WorkflowState, AllocationScore
from .workflow_nodes import WorkflowNodes

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
]
