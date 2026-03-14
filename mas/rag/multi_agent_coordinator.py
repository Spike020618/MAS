"""
跨Agent协调器 - 整合工作流和同步管理器，实现完整的多Agent任务分配

功能：
- 本地任务分配
- 广播远程请求
- 收集远程响应
- 智能选择方案
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any

from .rag_workflow import RAGWorkflow
from .rag_sync_manager import RAGSyncManager
from .workflow_state import WorkflowState
from .agent_message import (
    TaskRequestMessage,
    TaskResponseMessage,
    FeedbackMessage,
)

logger = logging.getLogger(__name__)


class MultiAgentCoordinator:
    """多Agent协调器"""

    def __init__(
        self,
        agent_id: int,
        agent_name: str,
        rag_workflow: RAGWorkflow,
        sync_manager: RAGSyncManager,
    ):
        """
        初始化多Agent协调器

        Args:
            agent_id: 本Agent的ID
            agent_name: 本Agent的名称
            rag_workflow: RAG工作流实例
            sync_manager: 同步管理器实例
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.workflow = rag_workflow
        self.sync_manager = sync_manager

        logger.info(
            f"✓ MultiAgentCoordinator initialized for Agent {agent_id} ({agent_name})"
        )

    # ────────────────────────────────────────────────────────
    # 主协调方法
    # ────────────────────────────────────────────────────────

    async def allocate_task_with_sync(
        self,
        task_request: Dict[str, Any],
        task_embedding: Optional[List[float]] = None,
        enable_remote: bool = True,
    ) -> Dict[str, Any]:
        """
        执行任务分配，包含本地和远程选项

        Args:
            task_request: 任务请求
            task_embedding: 任务向量（可选）
            enable_remote: 是否启用远程Agent查询

        Returns:
            最终分配结果
        """
        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"Starting Multi-Agent Task Allocation")
            logger.info(f"Coordinator: Agent {self.agent_id} ({self.agent_name})")
            logger.info(f"{'='*70}")

            # 步骤1：执行本地工作流
            logger.info("\n[步骤1] 执行本地RAG工作流...")
            local_state = await self.workflow.allocate_task(
                task_request=task_request,
                task_embedding=task_embedding,
            )

            # 步骤2：检查是否需要远程协助
            needs_remote = (
                enable_remote
                and local_state.allocation_decision == "remote_fallback"
            )

            if needs_remote:
                logger.info("\n[步骤2] 广播远程任务请求...")
                final_result = await self._handle_remote_allocation(
                    task_request=task_request,
                    local_state=local_state,
                )
            else:
                logger.info("\n[步骤2] 使用本地分配结果")
                final_result = local_state.allocation_result

            logger.info(f"\n{'='*70}")
            logger.info("Task Allocation Complete")
            logger.info(f"Decision: {local_state.allocation_decision}")
            logger.info(f"Selected Agents: {local_state.selected_agents}")
            logger.info(f"{'='*70}\n")

            return final_result

        except Exception as e:
            logger.error(f"✗ Task allocation failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "agents": [],
            }

    # ────────────────────────────────────────────────────────
    # 远程分配处理
    # ────────────────────────────────────────────────────────

    async def _handle_remote_allocation(
        self,
        task_request: Dict[str, Any],
        local_state: WorkflowState,
    ) -> Dict[str, Any]:
        """
        处理远程Agent分配

        Args:
            task_request: 任务请求
            local_state: 本地工作流状态

        Returns:
            最终分配结果
        """
        try:
            task_type = task_request.get("task_type")

            # 获取支持该任务类型的远程Agent
            logger.info(f"\nSearching for remote agents supporting task type: {task_type}")
            remote_agents = await self.sync_manager.list_agents_for_task(task_type)

            if not remote_agents:
                logger.warning("No remote agents available, using local results")
                return local_state.allocation_result

            logger.info(f"Found {len(remote_agents)} remote agents")

            # 创建任务请求消息
            message = TaskRequestMessage(
                sender_id=self.agent_id,
                sender_name=self.agent_name,
                task_request=task_request,
            )

            # 广播任务请求
            logger.info(f"Broadcasting task request to {len(remote_agents)} agents...")
            message_id = await self.sync_manager.broadcast_to_all(message)

            # 收集响应
            logger.info("Waiting for remote responses (timeout: 5s)...")
            responses = await self.sync_manager.collect_responses(
                message_id=message_id,
                timeout=5.0,
                min_responses=1,
            )

            if not responses:
                logger.warning("No responses from remote agents, using local results")
                return local_state.allocation_result

            # 分析远程响应
            logger.info(f"Received {len(responses)} responses, analyzing...")
            best_solution = await self._analyze_remote_responses(
                responses=responses,
                task_type=task_type,
            )

            if best_solution:
                logger.info(f"Selected remote agent: {best_solution.get('agent_id')}")
                return {
                    "type": "remote",
                    "source": "remote_agents",
                    "selected_agents": [best_solution.get("agent_id")],
                    "solution": best_solution,
                    "timestamp": __import__("time").time(),
                }
            else:
                logger.warning("No suitable remote solution, using local results")
                return local_state.allocation_result

        except Exception as e:
            logger.error(f"✗ Remote allocation failed: {e}")
            return local_state.allocation_result

    async def _analyze_remote_responses(
        self,
        responses: List,
        task_type: str,
    ) -> Optional[Dict[str, Any]]:
        """
        分析远程Agent的响应

        Args:
            responses: 响应消息列表
            task_type: 任务类型

        Returns:
            最佳方案
        """
        try:
            best_solution = None
            best_score = 0.0

            for response in responses:
                payload = response.payload
                agent_id = response.sender_id
                success_rate = payload.get("success_rate", 0.0)

                # 计算评分
                score = success_rate

                if score > best_score:
                    best_score = score
                    best_solution = {
                        "agent_id": agent_id,
                        "agent_name": response.sender_name,
                        "success_rate": success_rate,
                        "score": score,
                        **payload,
                    }

            return best_solution

        except Exception as e:
            logger.error(f"✗ Response analysis failed: {e}")
            return None

    # ────────────────────────────────────────────────────────
    # 反馈处理
    # ────────────────────────────────────────────────────────

    async def send_feedback_to_agent(
        self,
        target_agent_id: int,
        record_id: str,
        success_score: float,
        feedback_text: str = "",
    ) -> None:
        """
        向其他Agent发送反馈

        Args:
            target_agent_id: 目标Agent ID
            record_id: 分配记录ID
            success_score: 成功评分
            feedback_text: 反馈文本
        """
        try:
            # 创建反馈消息
            message = FeedbackMessage(
                sender_id=self.agent_id,
                sender_name=self.agent_name,
                target_agent_id=target_agent_id,
                record_id=record_id,
                success_score=success_score,
                feedback_text=feedback_text,
            )

            # 发送反馈
            await self.sync_manager.outgoing_queue.put(message)

            logger.info(
                f"[Agent {self.agent_id}] Sent feedback to Agent {target_agent_id}: "
                f"score={success_score}"
            )

        except Exception as e:
            logger.error(f"✗ Failed to send feedback: {e}")

    # ────────────────────────────────────────────────────────
    # 监控和统计
    # ────────────────────────────────────────────────────────

    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            sync_stats = await self.sync_manager.get_stats()
            workflow_stats = await self.workflow.get_system_stats()

            return {
                "agent": {
                    "id": self.agent_id,
                    "name": self.agent_name,
                },
                "sync_manager": sync_stats,
                "workflow": workflow_stats,
                "timestamp": __import__("datetime").datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {}

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            sync_health = await self.sync_manager.health_check()
            workflow_health = await self.workflow.health_check()

            health = {
                "status": "healthy",
                "agent": {
                    "id": self.agent_id,
                    "name": self.agent_name,
                },
                "sync_manager": sync_health,
                "workflow": workflow_health,
            }

            # 如果有任何组件不健康，标记为降级
            if (
                sync_health.get("status") != "healthy"
                or workflow_health.get("status") != "healthy"
            ):
                health["status"] = "degraded"

            return health

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "error": str(e)}
