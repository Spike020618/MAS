"""
基准贪心算法 - 对比实验的基线实现

简单贪心策略：
- 直接选择历史成功率最高的Agent
- 不使用向量检索
- 不使用权重学习
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class GreedyBaseline:
    """贪心基线算法"""

    def __init__(self, rag_database):
        """
        初始化贪心基线

        Args:
            rag_database: RAG数据库实例
        """
        self.rag_db = rag_database
        self.stats = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "avg_success_score": 0.0,
            "total_time": 0.0,
        }

        logger.info("✓ GreedyBaseline initialized")

    async def allocate_task(
        self,
        task_request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        使用贪心策略分配任务

        Args:
            task_request: 任务请求
                {
                    "task_id": "task_001",
                    "task_type": "review",
                    "description": "...",
                }

        Returns:
            分配结果
        """
        try:
            import time
            start_time = time.time()

            task_type = task_request.get("task_type")

            logger.info(
                f"[Greedy] Allocating task: {task_request.get('task_id')}, "
                f"type: {task_type}"
            )

            # 贪心策略：选择支持此任务类型且成功率最高的Agent
            agents = await self.rag_db.list_agents()

            best_agent = None
            best_success_rate = -1

            for agent_id, agent_info in agents.items():
                # 检查Agent是否支持此任务类型
                if task_type in agent_info.get("task_types", []):
                    success_rate = agent_info.get("success_rate", 0.0)

                    if success_rate > best_success_rate:
                        best_success_rate = success_rate
                        best_agent = agent_id

            if best_agent is None:
                logger.warning(
                    f"[Greedy] No agent found for task type {task_type}"
                )
                return {
                    "success": False,
                    "error": "No suitable agent found",
                    "allocated_agents": [],
                    "time_ms": (time.time() - start_time) * 1000,
                }

            # 记录分配
            self.stats["total_tasks"] += 1

            elapsed = time.time() - start_time

            logger.info(
                f"[Greedy] ✓ Allocated to Agent {best_agent} "
                f"(success_rate={best_success_rate:.2%})"
            )

            return {
                "success": True,
                "allocated_agents": [best_agent],
                "agent_scores": {best_agent: best_success_rate},
                "allocation_decision": "greedy_best_rate",
                "allocation_reason": f"Highest success rate: {best_success_rate:.2%}",
                "time_ms": elapsed * 1000,
            }

        except Exception as e:
            logger.error(f"[Greedy] ✗ Allocation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "allocated_agents": [],
            }

    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            total = self.stats["total_tasks"]
            successful = self.stats["successful_tasks"]

            return {
                "total_tasks": total,
                "successful_tasks": successful,
                "success_rate": successful / total if total > 0 else 0.0,
                "avg_success_score": self.stats["avg_success_score"],
                "avg_time_ms": (
                    self.stats["total_time"] / total if total > 0 else 0.0
                ),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    async def record_feedback(
        self,
        record_id: str,
        success_score: float,
        agent_ids: Optional[List[int]] = None,
    ) -> None:
        """记录反馈（贪心方案不学习）"""
        try:
            if success_score > 0.5:
                self.stats["successful_tasks"] += 1

            # 计算平均成功分数
            total = self.stats["total_tasks"]
            if total > 0:
                current_avg = self.stats["avg_success_score"]
                self.stats["avg_success_score"] = (
                    current_avg * (total - 1) + success_score
                ) / total

        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
