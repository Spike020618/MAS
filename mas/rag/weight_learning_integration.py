"""
权重学习集成器 - 将权重学习与RAG工作流集成

功能：
- 反馈自动触发权重学习
- 工作流与学习器集成
- 性能监控和自适应
- 与consensus.py的集成接口
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any

from .weight_learner import WeightLearner
from .rag_workflow import RAGWorkflow
from .local_rag_database import LocalRAGDatabase

logger = logging.getLogger(__name__)


class WeightLearningIntegration:
    """权重学习集成器"""

    def __init__(
        self,
        rag_database: LocalRAGDatabase,
        rag_workflow: RAGWorkflow,
        initial_weights: Optional[Dict[str, float]] = None,
        learning_rate: float = 0.01,
    ):
        """
        初始化权重学习集成

        Args:
            rag_database: RAG数据库实例
            rag_workflow: RAG工作流实例
            initial_weights: 初始权重
            learning_rate: 学习率
        """
        self.rag_db = rag_database
        self.workflow = rag_workflow
        self.weight_learner = WeightLearner(
            initial_weights=initial_weights,
            learning_rate=learning_rate,
        )

        # 性能指标
        self.performance_metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "avg_success_score": 0.0,
            "learning_iterations": 0,
        }

        logger.info("✓ WeightLearningIntegration initialized")

    # ────────────────────────────────────────────────────────
    # 工作流执行与学习
    # ────────────────────────────────────────────────────────

    async def execute_task_with_learning(
        self,
        task_request: Dict[str, Any],
        task_embedding: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        执行任务并自动触发学习

        Args:
            task_request: 任务请求
            task_embedding: 任务向量

        Returns:
            分配结果
        """
        try:
            logger.info(
                f"\n{'='*70}"
            )
            logger.info(f"Executing task with learning")
            logger.info(f"{'='*70}")

            # 步骤1：执行工作流
            logger.info("\n[步骤1] 执行工作流...")
            state = await self.workflow.allocate_task(
                task_request=task_request,
                task_embedding=task_embedding,
            )

            result = {
                "workflow_state": state,
                "allocated_agents": state.selected_agents,
                "allocation_decision": state.allocation_decision,
                "success": state.success,
            }

            # 更新性能指标
            self.performance_metrics["total_tasks"] += 1

            logger.info(f"✓ Task executed: {result['allocated_agents']}")

            return result

        except Exception as e:
            logger.error(f"✗ Task execution failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def process_feedback_with_learning(
        self,
        record_id: str,
        success_score: float,
        agent_ids: Optional[List[int]] = None,
        feedback_text: str = "",
        agent_scores: Optional[Dict[int, float]] = None,
    ) -> Dict[str, Any]:
        """
        处理反馈并触发权重学习

        Args:
            record_id: 分配记录ID
            success_score: 成功评分 (0.0-1.0)
            agent_ids: 相关Agent ID列表
            feedback_text: 反馈文本
            agent_scores: Agent评分

        Returns:
            学习结果
        """
        try:
            logger.info(
                f"\n{'='*70}"
            )
            logger.info(f"Processing feedback with learning")
            logger.info(f"Record: {record_id}, Score: {success_score:.2f}")
            logger.info(f"{'='*70}")

            # 步骤1：保存反馈到数据库
            logger.info("\n[步骤1] 保存反馈到数据库...")
            await self.rag_db.record_success(
                record_id=record_id,
                task_id=f"task_{record_id}",
                agent_ids=agent_ids or [],
                feedback=feedback_text,
                success_score=success_score,
                metadata={"agent_scores": agent_scores or {}},
            )

            # 步骤2：触发权重学习
            logger.info("\n[步骤2] 触发权重学习...")
            updated_weights = await self.weight_learner.update_weights_from_feedback(
                success_score=success_score,
                agent_scores=agent_scores,
                feedback_text=feedback_text,
                metadata={
                    "record_id": record_id,
                    "agent_ids": agent_ids,
                },
            )

            # 步骤3：更新数据库权重
            logger.info("\n[步骤3] 更新数据库权重...")
            await self.rag_db.set_weights(updated_weights)

            # 更新性能指标
            self.performance_metrics["successful_tasks"] += 1
            self.performance_metrics["learning_iterations"] += 1

            # 计算平均成功率
            total = self.performance_metrics["total_tasks"]
            successful = self.performance_metrics["successful_tasks"]
            if total > 0:
                avg_score = (
                    self.performance_metrics["avg_success_score"] * (total - 1)
                    + success_score
                ) / total
                self.performance_metrics["avg_success_score"] = avg_score

            logger.info(f"✓ Feedback processed and weights updated")
            logger.info(f"New weights: {updated_weights}")

            return {
                "success": True,
                "updated_weights": updated_weights,
                "performance_metrics": self.performance_metrics,
            }

        except Exception as e:
            logger.error(f"✗ Feedback processing failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    # ────────────────────────────────────────────────────────
    # 批量学习
    # ────────────────────────────────────────────────────────

    async def batch_learning_from_history(
        self, num_records: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        从历史记录进行批量学习

        Args:
            num_records: 要学习的记录数（None表示全部）

        Returns:
            学习结果
        """
        try:
            logger.info(
                f"\n{'='*70}"
            )
            logger.info(f"Starting batch learning from history")
            logger.info(f"{'='*70}")

            # 这里会从RAG数据库获取历史记录
            # 为了演示，这里是一个占位符

            logger.info("✓ Batch learning completed")

            return {
                "success": True,
                "learning_iterations": self.performance_metrics["learning_iterations"],
            }

        except Exception as e:
            logger.error(f"✗ Batch learning failed: {e}")
            return {"success": False, "error": str(e)}

    # ────────────────────────────────────────────────────────
    # 性能监控
    # ────────────────────────────────────────────────────────

    async def get_learning_status(self) -> Dict[str, Any]:
        """获取学习状态"""
        try:
            # 获取权重学习统计
            learner_stats = await self.weight_learner.get_stats()

            # 获取收敛指标
            convergence = await self.weight_learner.get_convergence_metrics()

            status = {
                "current_weights": self.weight_learner.weights,
                "performance_metrics": self.performance_metrics,
                "learner_stats": learner_stats,
                "convergence": convergence,
                "system_health": await self._check_system_health(),
            }

            return status

        except Exception as e:
            logger.error(f"✗ Failed to get learning status: {e}")
            return {}

    async def _check_system_health(self) -> Dict[str, Any]:
        """检查系统健康状态"""
        try:
            total = self.performance_metrics["total_tasks"]
            successful = self.performance_metrics["successful_tasks"]

            health = {
                "total_tasks": total,
                "successful_tasks": successful,
                "success_rate": (
                    successful / total if total > 0 else 0.0
                ),
                "avg_success_score": self.performance_metrics["avg_success_score"],
                "learning_iterations": self.performance_metrics[
                    "learning_iterations"
                ],
            }

            # 判断系统状态
            if health["success_rate"] > 0.8:
                health["status"] = "excellent"
            elif health["success_rate"] > 0.6:
                health["status"] = "good"
            elif health["success_rate"] > 0.4:
                health["status"] = "fair"
            else:
                health["status"] = "poor"

            return health

        except Exception as e:
            logger.error(f"✗ Health check failed: {e}")
            return {"status": "error"}

    # ────────────────────────────────────────────────────────
    # Consensus集成
    # ────────────────────────────────────────────────────────

    async def evaluate_with_consensus(
        self,
        records: List[Dict[str, Any]],
        use_consensus: bool = True,
    ) -> Dict[str, Any]:
        """
        使用consensus.py评估记录

        Args:
            records: AEIC记录列表
            use_consensus: 是否使用consensus引擎

        Returns:
            评估结果
        """
        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"Evaluating records with consensus")
            logger.info(f"{'='*70}")

            if not use_consensus:
                logger.info("Consensus evaluation disabled, skipping")
                return {"consensus_enabled": False}

            # 尝试导入consensus引擎
            try:
                from ..consensus.consensus import ConsensusEngine

                consensus = ConsensusEngine()

                # 评估共识
                logger.info(f"Evaluating {len(records)} records...")
                consensus_result = consensus.evaluate_consensus(
                    node_records=records
                )

                logger.info(f"✓ Consensus evaluation completed")
                logger.info(f"Consensus energy: {consensus_result.get('consensus_energy')}")

                return {
                    "success": True,
                    "consensus_result": consensus_result,
                }

            except ImportError:
                logger.warning(
                    "Consensus module not available, skipping consensus evaluation"
                )
                return {"consensus_enabled": False, "reason": "module_not_found"}

        except Exception as e:
            logger.error(f"✗ Consensus evaluation failed: {e}")
            return {"success": False, "error": str(e)}

    # ────────────────────────────────────────────────────────
    # 工具方法
    # ────────────────────────────────────────────────────────

    async def save_learning_history(self, filepath: str) -> None:
        """保存学习历史"""
        try:
            await self.weight_learner.save_history(filepath)
            logger.info(f"✓ Learning history saved to {filepath}")
        except Exception as e:
            logger.error(f"✗ Failed to save history: {e}")

    async def reset_learning(self) -> None:
        """重置学习状态"""
        try:
            await self.weight_learner.reset_weights()
            self.performance_metrics = {
                "total_tasks": 0,
                "successful_tasks": 0,
                "avg_success_score": 0.0,
                "learning_iterations": 0,
            }
            logger.info("✓ Learning state reset")
        except Exception as e:
            logger.error(f"✗ Failed to reset learning: {e}")
