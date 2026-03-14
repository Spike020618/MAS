"""
实验运行器 - 协调三种算法的对比实验

功能：
- 准备实验环境
- 运行三种算法
- 收集和对比结果
"""

import logging
import asyncio
from typing import Dict, List, Any

from .greedy_baseline import GreedyBaseline
from .rag_workflow import RAGWorkflow
from .weight_learning_integration import WeightLearningIntegration
from .dataset_generator import DatasetGenerator, TaskSample
from .results_analyzer import ResultsAnalyzer

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """实验运行器"""

    def __init__(self, rag_database):
        """初始化实验运行器"""
        self.rag_db = rag_database
        self.analyzer = ResultsAnalyzer()
        logger.info("✓ ExperimentRunner initialized")

    async def run_experiment(
        self,
        num_agents: int = 5,
        num_tasks: int = 50,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        运行完整对比实验

        Args:
            num_agents: Agent数量
            num_tasks: 任务数量
            seed: 随机种子

        Returns:
            实验结果
        """
        try:
            logger.info("\n" + "=" * 80)
            logger.info("开始对比实验")
            logger.info("=" * 80)

            # 步骤1：生成数据集
            logger.info("\n[步骤1] 生成数据集...")
            dataset_gen = DatasetGenerator(seed=seed)
            agents = dataset_gen.generate_agents(num_agents)
            tasks = dataset_gen.generate_tasks(num_tasks, agents)
            feedback_samples = dataset_gen.generate_feedback_samples(tasks, agents)

            # 注册Agent到数据库
            for agent_config in agents:
                await self.rag_db.register_agent(
                    agent_id=agent_config.agent_id,
                    name=agent_config.name,
                    task_types=agent_config.task_types,
                    success_rate=agent_config.base_success_rate,
                )

            logger.info(f"✓ Generated {len(agents)} agents, {len(tasks)} tasks")

            # 步骤2：运行三种算法
            logger.info("\n[步骤2] 运行三种算法...")

            results = {}

            # 算法1：贪心基线
            logger.info("\n【算法1：贪心基线】")
            greedy_results = await self._run_greedy(tasks, feedback_samples)
            results["greedy"] = greedy_results

            # 算法2：RAG检索
            logger.info("\n【算法2：RAG检索】")
            rag_results = await self._run_rag(tasks, feedback_samples)
            results["rag"] = rag_results

            # 算法3：RAG+权重学习
            logger.info("\n【算法3：RAG+权重学习】")
            rag_learning_results = await self._run_rag_with_learning(
                tasks, feedback_samples
            )
            results["rag_learning"] = rag_learning_results

            logger.info(f"✓ All algorithms completed")

            return results

        except Exception as e:
            logger.error(f"✗ Experiment failed: {e}", exc_info=True)
            raise

    async def _run_greedy(
        self,
        tasks: List[TaskSample],
        feedback_samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """运行贪心基线算法"""
        try:
            greedy = GreedyBaseline(self.rag_db)
            results = []

            for i, task in enumerate(tasks, 1):
                # 分配任务
                allocation = await greedy.allocate_task(
                    {
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "description": task.description,
                    }
                )

                # 获取反馈
                matching_feedbacks = [
                    f for f in feedback_samples if f["task_id"] == task.task_id
                ]

                if matching_feedbacks:
                    # 使用第一个反馈
                    feedback = matching_feedbacks[0]
                    success_score = feedback["success_score"]
                    is_optimal = feedback["is_optimal"]

                    # 记录反馈
                    await greedy.record_feedback(
                        record_id=f"rec_greedy_{i:04d}",
                        success_score=success_score,
                        agent_ids=allocation.get("allocated_agents", []),
                    )

                    result = {
                        "task_id": task.task_id,
                        "allocated_agent_id": allocation.get("allocated_agents", [None])[0],
                        "success_score": success_score,
                        "is_optimal": is_optimal,
                        "time_ms": allocation.get("time_ms", 0),
                    }
                    results.append(result)

                if i % 10 == 0:
                    logger.info(f"  已完成: {i}/{len(tasks)}")

            stats = await greedy.get_stats()
            logger.info(f"✓ 贪心基线完成: 成功率={stats['success_rate']:.2%}")

            return {
                "algorithm": "greedy",
                "results": results,
                "stats": stats,
            }

        except Exception as e:
            logger.error(f"✗ Greedy experiment failed: {e}")
            raise

    async def _run_rag(
        self,
        tasks: List[TaskSample],
        feedback_samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """运行RAG检索算法"""
        try:
            workflow = RAGWorkflow(self.rag_db)
            results = []

            for i, task in enumerate(tasks, 1):
                # 向量化任务
                embedding = await self.rag_db.embedding.embed(task.description)

                # 分配任务
                state = await workflow.allocate_task(
                    task_request={
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "description": task.description,
                    },
                    task_embedding=embedding,
                )

                # 获取反馈
                matching_feedbacks = [
                    f for f in feedback_samples if f["task_id"] == task.task_id
                ]

                if matching_feedbacks and state.selected_agents:
                    feedback = matching_feedbacks[0]
                    success_score = feedback["success_score"]
                    is_optimal = (
                        feedback["allocated_agent_id"] in state.selected_agents
                    )

                    result = {
                        "task_id": task.task_id,
                        "allocated_agent_id": state.selected_agents[0],
                        "success_score": success_score,
                        "is_optimal": is_optimal,
                        "time_ms": state.get_duration_ms(),
                    }
                    results.append(result)

                if i % 10 == 0:
                    logger.info(f"  已完成: {i}/{len(tasks)}")

            # 计算统计
            total = len(results)
            successful = sum(1 for r in results if r["success_score"] > 0.5)
            avg_score = sum(r["success_score"] for r in results) / total if total > 0 else 0

            logger.info(f"✓ RAG算法完成: 成功率={successful/total:.2%}")

            return {
                "algorithm": "rag",
                "results": results,
                "stats": {
                    "total_tasks": total,
                    "successful_tasks": successful,
                    "success_rate": successful / total if total > 0 else 0,
                    "avg_success_score": avg_score,
                },
            }

        except Exception as e:
            logger.error(f"✗ RAG experiment failed: {e}")
            raise

    async def _run_rag_with_learning(
        self,
        tasks: List[TaskSample],
        feedback_samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """运行RAG+权重学习算法"""
        try:
            workflow = RAGWorkflow(self.rag_db)
            learner = WeightLearningIntegration(
                rag_database=self.rag_db,
                rag_workflow=workflow,
                learning_rate=0.01,
            )

            results = []

            for i, task in enumerate(tasks, 1):
                # 向量化任务
                embedding = await self.rag_db.embedding.embed(task.description)

                # 执行任务（带学习）
                result = await learner.execute_task_with_learning(
                    task_request={
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "description": task.description,
                    },
                    task_embedding=embedding,
                )

                # 获取反馈并触发学习
                matching_feedbacks = [
                    f for f in feedback_samples if f["task_id"] == task.task_id
                ]

                if matching_feedbacks and result.get("success"):
                    feedback = matching_feedbacks[0]
                    success_score = feedback["success_score"]
                    is_optimal = (
                        feedback["allocated_agent_id"]
                        in result.get("allocated_agents", [])
                    )

                    # 处理反馈（自动触发权重学习）
                    await learner.process_feedback_with_learning(
                        record_id=f"rec_learning_{i:04d}",
                        success_score=success_score,
                        agent_ids=result.get("allocated_agents", []),
                        feedback_text="Feedback",
                    )

                    result_item = {
                        "task_id": task.task_id,
                        "allocated_agent_id": result.get("allocated_agents", [None])[0],
                        "success_score": success_score,
                        "is_optimal": is_optimal,
                        "time_ms": 0,  # 在实际实现中应该记录
                    }
                    results.append(result_item)

                if i % 10 == 0:
                    logger.info(f"  已完成: {i}/{len(tasks)}")

            # 获取学习状态
            status = await learner.get_learning_status()

            # 计算统计
            total = len(results)
            successful = sum(1 for r in results if r["success_score"] > 0.5)
            avg_score = sum(r["success_score"] for r in results) / total if total > 0 else 0

            logger.info(
                f"✓ RAG+学习算法完成: 成功率={successful/total:.2%}, "
                f"权重稳定性={status.get('convergence', {}).get('learning_stability', 0):.4f}"
            )

            return {
                "algorithm": "rag_learning",
                "results": results,
                "stats": {
                    "total_tasks": total,
                    "successful_tasks": successful,
                    "success_rate": successful / total if total > 0 else 0,
                    "avg_success_score": avg_score,
                },
                "learning_metrics": status,
            }

        except Exception as e:
            logger.error(f"✗ RAG+Learning experiment failed: {e}")
            raise
