"""
第4步演示：权重学习集成 - 反馈驱动的自适应权重更新

演示内容：
1. 初始化权重学习集成
2. 执行多个任务分配
3. 收集反馈
4. 触发权重学习
5. 观察权重变化和性能改进
6. 获取收敛指标
"""

import asyncio
import logging
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from mas.rag.local_rag_database import LocalRAGDatabase
from mas.rag.rag_workflow import RAGWorkflow
from mas.rag.weight_learning_integration import WeightLearningIntegration


async def demo():
    """运行第4步演示"""

    logger.info("\n" + "=" * 80)
    logger.info("第4步演示：权重学习集成 - 反馈驱动的自适应权重更新")
    logger.info("=" * 80)

    try:
        # ────────────────────────────────────────────────────────
        # 第1步：初始化基础设施
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤1] 初始化基础设施...")

        # 创建RAG数据库
        rag_db = LocalRAGDatabase(
            storage_path="./rag_storage_step4",
            embedding_model="local_hash",
            embedding_dimension=1536,
        )
        await rag_db.initialize()

        # 创建工作流
        workflow = RAGWorkflow(rag_db)

        # 创建权重学习集成
        initial_weights = {
            "w_A": 0.25,
            "w_E": 0.25,
            "w_I": 0.25,
            "w_C": 0.25,
        }
        learner_integration = WeightLearningIntegration(
            rag_database=rag_db,
            rag_workflow=workflow,
            initial_weights=initial_weights,
            learning_rate=0.01,
        )

        logger.info("✓ 基础设施初始化完成")

        # 显示初始权重
        logger.info(f"\n初始权重: {learner_integration.weight_learner.weights}")

        # ────────────────────────────────────────────────────────
        # 第2步：注册Agent和数据
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤2] 注册Agent和数据...")

        agents = [
            {
                "agent_id": 1,
                "name": "ReviewExpert",
                "task_types": ["review"],
                "success_rate": 0.85,
            },
            {
                "agent_id": 2,
                "name": "PlanningMaster",
                "task_types": ["planning"],
                "success_rate": 0.80,
            },
            {
                "agent_id": 3,
                "name": "DeveloperBot",
                "task_types": ["development"],
                "success_rate": 0.90,
            },
        ]

        for agent in agents:
            await rag_db.register_agent(**agent)

        # 添加任务和方案
        tasks = [
            {
                "task_id": "task_review_001",
                "task_type": "review",
                "description": "代码审查",
            },
        ]

        for task in tasks:
            await rag_db.add_task(**task)

        solutions = [
            {
                "solution_id": "sol_review_001",
                "agent_id": 1,
                "task_type": "review",
                "solution_text": "静态分析工具审查",
                "success_rate": 0.85,
            },
        ]

        for solution in solutions:
            await rag_db.add_solution(**solution)

        logger.info(f"✓ 注册了 {len(agents)} 个Agent")

        # ────────────────────────────────────────────────────────
        # 第3步：执行多个任务并收集反馈
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤3] 执行任务并收集反馈...")

        # 定义任务和预期反馈
        task_feedbacks = [
            {
                "task": {
                    "task_id": "task_001",
                    "task_type": "review",
                    "description": "需要代码审查",
                },
                "success_score": 0.95,  # 非常成功
                "feedback": "审查质量优秀",
            },
            {
                "task": {
                    "task_id": "task_002",
                    "task_type": "review",
                    "description": "新项目代码审查",
                },
                "success_score": 0.90,  # 成功
                "feedback": "完成得很好",
            },
            {
                "task": {
                    "task_id": "task_003",
                    "task_type": "review",
                    "description": "紧急代码审查",
                },
                "success_score": 0.85,  # 成功
                "feedback": "及时完成",
            },
            {
                "task": {
                    "task_id": "task_004",
                    "task_type": "review",
                    "description": "复杂代码审查",
                },
                "success_score": 0.75,  # 一般成功
                "feedback": "完成但有遗漏",
            },
            {
                "task": {
                    "task_id": "task_005",
                    "task_type": "review",
                    "description": "快速审查",
                },
                "success_score": 0.88,  # 成功
                "feedback": "快速完成",
            },
        ]

        task_results = []

        for i, task_feedback in enumerate(task_feedbacks, 1):
            logger.info(f"\n【任务{i}】执行任务...")

            # 执行任务
            task_request = task_feedback["task"]
            result = await learner_integration.execute_task_with_learning(
                task_request=task_request,
                task_embedding=await rag_db.embedding.embed(
                    task_request["description"]
                ),
            )

            if result["success"]:
                logger.info(f"✓ 任务执行成功")

                # 获取分配结果
                record_id = f"rec_{i:03d}"
                agents = result["allocated_agents"]

                # 处理反馈并触发学习
                logger.info(f"  处理反馈（分数：{task_feedback['success_score']}）...")
                feedback_result = (
                    await learner_integration.process_feedback_with_learning(
                        record_id=record_id,
                        success_score=task_feedback["success_score"],
                        agent_ids=agents,
                        feedback_text=task_feedback["feedback"],
                        agent_scores={agents[0]: task_feedback["success_score"]}
                        if agents
                        else {},
                    )
                )

                if feedback_result["success"]:
                    logger.info(
                        f"✓ 反馈处理完成，权重已更新"
                    )

                    # 记录结果
                    task_results.append(
                        {
                            "task_id": task_request["task_id"],
                            "success_score": task_feedback["success_score"],
                            "weights_after": feedback_result.get("updated_weights"),
                        }
                    )

                    # 显示权重变化
                    logger.info(
                        f"  当前权重: {feedback_result['updated_weights']}"
                    )
                else:
                    logger.error(f"✗ 反馈处理失败: {feedback_result.get('error')}")

            else:
                logger.error(f"✗ 任务执行失败")

        # ────────────────────────────────────────────────────────
        # 第4步：分析学习过程
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤4] 分析学习过程...")

        # 获取权重历史
        history = await learner_integration.weight_learner.get_weight_history()
        logger.info(f"\n权重更新历史 ({len(history)} 条):")
        for i, record in enumerate(history[-3:], 1):  # 显示最后3条
            weights = record["weights"]
            logger.info(
                f"  [{i}] w_A={weights['w_A']:.3f}, "
                f"w_E={weights['w_E']:.3f}, "
                f"w_I={weights['w_I']:.3f}, "
                f"w_C={weights['w_C']:.3f}"
            )

        # 获取收敛指标
        convergence = await learner_integration.weight_learner.get_convergence_metrics()
        logger.info(f"\n收敛指标:")
        logger.info(f"  - 样本数: {convergence.get('samples')}")
        logger.info(
            f"  - 权重变化均值: {convergence.get('weight_change_mean', 0):.6f}"
        )
        logger.info(
            f"  - 梯度范数均值: {convergence.get('gradient_norm_mean', 0):.6f}"
        )
        logger.info(f"  - 学习稳定性: {convergence.get('learning_stability', 0):.4f}")

        # ────────────────────────────────────────────────────────
        # 第5步：获取学习状态
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤5] 获取最终学习状态...")

        status = await learner_integration.get_learning_status()

        logger.info(f"\n最终权重: {status.get('current_weights')}")
        logger.info(f"\n性能指标:")
        perf = status.get("performance_metrics", {})
        logger.info(f"  - 总任务数: {perf.get('total_tasks')}")
        logger.info(f"  - 成功任务数: {perf.get('successful_tasks')}")
        logger.info(f"  - 平均成功分数: {perf.get('avg_success_score', 0):.2f}")
        logger.info(f"  - 学习迭代数: {perf.get('learning_iterations')}")

        logger.info(f"\n系统健康状态:")
        health = status.get("system_health", {})
        logger.info(f"  - 成功率: {health.get('success_rate', 0):.2%}")
        logger.info(f"  - 系统状态: {health.get('status')}")

        # ────────────────────────────────────────────────────────
        # 第6步：保存学习历史
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤6] 保存学习历史...")

        await learner_integration.save_learning_history(
            "./rag_storage_step4/weight_learning_history.json"
        )

        logger.info("✓ 学习历史已保存")

        # ────────────────────────────────────────────────────────
        # 第7步：保存并关闭
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤7] 保存并关闭...")

        await rag_db.save()
        await rag_db.close()

        logger.info("✓ 数据库已保存并关闭")

        # ────────────────────────────────────────────────────────
        # 总结
        # ────────────────────────────────────────────────────────

        logger.info("\n" + "=" * 80)
        logger.info("第4步演示完成！")
        logger.info("=" * 80)
        logger.info("\n关键成果:")
        logger.info(f"  ✓ 执行了 {len(task_results)} 个任务")
        logger.info(f"  ✓ 收集了 {len(task_results)} 次反馈")
        logger.info(f"  ✓ 完成了 {perf.get('learning_iterations')} 次权重学习")
        logger.info(f"  ✓ 达到了 {health.get('success_rate', 0):.2%} 的成功率")
        logger.info(f"  ✓ 权重学习稳定性: {convergence.get('learning_stability', 0):.4f}")
        logger.info("\n" + "=" * 80 + "\n")

    except Exception as e:
        logger.error(f"\n✗ 演示过程中出错: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(demo())
