"""
LangGraph 任务分配工作流

完整的任务分配流程：
1. 预处理
2. 本地RAG搜索
3. 评估本地命中
4. 本地分配 或 远程请求
5. 最终化
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any

from .workflow_state import WorkflowState
from .workflow_nodes import WorkflowNodes

logger = logging.getLogger(__name__)


class RAGWorkflow:
    """LangGraph 任务分配工作流"""

    def __init__(self, rag_database):
        """
        初始化工作流

        Args:
            rag_database: LocalRAGDatabase 实例
        """
        self.rag = rag_database
        self.nodes = WorkflowNodes(rag_database)
        logger.info("✓ RAGWorkflow initialized")

    # ────────────────────────────────────────────────────────
    # 主工作流入口
    # ────────────────────────────────────────────────────────

    async def allocate_task(
        self,
        task_request: Dict[str, Any],
        task_embedding: Optional[List[float]] = None,
    ) -> WorkflowState:
        """
        执行任务分配工作流

        Args:
            task_request: 任务请求
                {
                    "task_id": "task_001",
                    "task_type": "review",
                    "description": "代码审查",
                    ...
                }
            task_embedding: 任务描述的向量表示（可选）

        Returns:
            最终分配状态
        """
        # 初始化状态
        state = WorkflowState(
            task_request=task_request,
            task_embedding=task_embedding or [],
        )

        logger.info(f"\n{'='*70}")
        logger.info(f"Starting RAG Workflow: {state.workflow_id}")
        logger.info(f"Task Type: {task_request.get('task_type')}")
        logger.info(f"{'='*70}")

        try:
            # 如果没有提供embedding，自动生成
            if not state.task_embedding:
                description = task_request.get("description", "")
                if description:
                    state.task_embedding = await self.rag.embedding.embed(description)
                    logger.info(f"Generated embedding for task")

            # 节点1：预处理
            state = await self.nodes.preprocess(state)
            if state.error_message and "Missing" in state.error_message:
                return state

            # 节点2：本地RAG搜索
            state = await self.nodes.local_rag_search(state)

            # 节点3：评估本地命中（决策节点）
            decision = await self.nodes.evaluate_local_hit(state)

            if decision == "allocate_local":
                # 节点4a：本地分配
                state = await self.nodes.allocate_local(state)
            else:
                # 节点4b：远程请求
                state = await self.nodes.request_remote(state)

            # 节点5：最终化
            state = await self.nodes.finalize(state)

            logger.info(f"\n{'='*70}")
            logger.info(f"Workflow Result: {state.to_dict()}")
            logger.info(f"{'='*70}\n")

            return state

        except Exception as e:
            state.success = False
            state.error_message = str(e)
            import time
            state.end_time = time.time()
            logger.error(f"Workflow failed: {e}", exc_info=True)
            return state

    # ────────────────────────────────────────────────────────
    # 工作流分析和监控
    # ────────────────────────────────────────────────────────

    async def get_workflow_stats(self, state: WorkflowState) -> Dict[str, Any]:
        """
        获取工作流统计信息

        Args:
            state: 工作流状态

        Returns:
            统计信息字典
        """
        try:
            total_time = state.get_duration_ms()

            stats = {
                "workflow_id": state.workflow_id,
                "success": state.success,
                "allocation_decision": state.allocation_decision,
                "selected_agents": state.selected_agents,
                "total_time_ms": total_time,
                "step_timings": state.step_timings,
                "metrics": {
                    "local_results_count": len(state.local_search_results),
                    "best_agents_count": len(state.best_agents),
                    "hit_confidence": state.local_hit_confidence,
                },
            }

            if state.error_message:
                stats["error"] = state.error_message

            return stats

        except Exception as e:
            logger.error(f"Failed to get workflow stats: {e}")
            return {}

    # ────────────────────────────────────────────────────────
    # 反馈处理
    # ────────────────────────────────────────────────────────

    async def process_feedback(
        self,
        record_id: str,
        success_score: float,
        feedback_text: str = "",
    ) -> None:
        """
        处理分配反馈用于权重学习

        Args:
            record_id: 分配记录ID
            success_score: 成功评分 (0.0-1.0)
            feedback_text: 反馈文本
        """
        try:
            logger.info(
                f"Processing feedback for record: {record_id}, score={success_score}"
            )

            # 这里应该实现权重学习逻辑
            # 基于consensus.py中的权重学习机制

            # 获取当前权重
            weights = await self.rag.get_weights()

            # 基于反馈更新权重（简化版）
            # 在实际实现中应该使用梯度上升
            if success_score > 0.8:
                weights["w_A"] = min(weights["w_A"] + 0.01, 0.5)
                weights["w_E"] = min(weights["w_E"] + 0.01, 0.5)
            elif success_score < 0.5:
                weights["w_A"] = max(weights["w_A"] - 0.01, 0.1)
                weights["w_E"] = max(weights["w_E"] - 0.01, 0.1)

            # 归一化权重
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}

            await self.rag.set_weights(weights)
            logger.info(f"✓ Weights updated: {weights}")

        except Exception as e:
            logger.error(f"✗ Failed to process feedback: {e}")

    # ────────────────────────────────────────────────────────
    # 工作流监控
    # ────────────────────────────────────────────────────────

    async def get_system_stats(self) -> Dict[str, Any]:
        """获取系统级统计信息"""
        try:
            return await self.rag.get_stats()
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {}

    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查

        Returns:
            健康状态信息
        """
        try:
            stats = await self.rag.get_stats()
            agents = await self.rag.list_agents()

            health = {
                "status": "healthy" if agents else "degraded",
                "agents_count": len(agents),
                "tasks_count": stats.get("tasks", 0),
                "solutions_count": stats.get("solutions", 0),
                "records_count": stats.get("records", 0),
                "timestamp": __import__("datetime").datetime.now().isoformat(),
            }

            logger.info(f"Health check: {health['status']}")
            return health

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "error": str(e)}


# ────────────────────────────────────────────────────────
# 异步助手函数
# ────────────────────────────────────────────────────────


async def run_workflow(
    rag_workflow: RAGWorkflow,
    task_request: Dict[str, Any],
    task_embedding: Optional[List[float]] = None,
) -> WorkflowState:
    """
    运行工作流的异步包装

    Args:
        rag_workflow: RAGWorkflow 实例
        task_request: 任务请求
        task_embedding: 任务向量（可选）

    Returns:
        工作流最终状态
    """
    return await rag_workflow.allocate_task(
        task_request=task_request,
        task_embedding=task_embedding,
    )
