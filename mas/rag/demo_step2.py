"""
第2步演示：LangGraph工作流 - 任务分配有向无环图

演示内容：
1. 初始化RAG数据库和工作流
2. 注册Agent
3. 添加任务模板和方案
4. 执行多个任务分配工作流
5. 查看工作流执行结果
6. 处理反馈和权重更新
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

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


async def demo():
    """运行第2步演示"""

    logger.info("\n" + "=" * 80)
    logger.info("第2步演示：LangGraph工作流 - 任务分配有向无环图")
    logger.info("=" * 80)

    try:
        # ────────────────────────────────────────────────────────
        # 第1步：初始化数据库和工作流
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤1] 初始化RAG数据库和工作流...")

        rag_db = LocalRAGDatabase(
            storage_path="./rag_storage_step2",
            embedding_model="local_hash",
            embedding_dimension=1536,
        )

        await rag_db.initialize()

        # 初始化工作流
        workflow = RAGWorkflow(rag_db)

        logger.info("✓ RAG数据库和工作流初始化完成")

        # ────────────────────────────────────────────────────────
        # 第2步：注册Agent
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤2] 注册Agent...")

        agents = [
            {
                "agent_id": 1,
                "name": "ReviewExpert",
                "task_types": ["review", "analysis"],
                "success_rate": 0.85,
            },
            {
                "agent_id": 2,
                "name": "PlanningMaster",
                "task_types": ["planning", "design"],
                "success_rate": 0.80,
            },
            {
                "agent_id": 3,
                "name": "DeveloperBot",
                "task_types": ["development", "coding"],
                "success_rate": 0.90,
            },
        ]

        for agent in agents:
            await rag_db.register_agent(**agent)

        logger.info(f"✓ 注册了 {len(agents)} 个Agent")

        # ────────────────────────────────────────────────────────
        # 第3步：添加任务模板和方案
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤3] 添加任务模板和方案...")

        # 添加任务
        tasks = [
            {
                "task_id": "task_review_001",
                "task_type": "review",
                "description": "代码审查：检查代码质量、性能和安全性",
                "metadata": {"domain": "code", "priority": "high"},
            },
            {
                "task_id": "task_planning_001",
                "task_type": "planning",
                "description": "项目规划：制定项目时间表和资源分配",
                "metadata": {"domain": "project", "complexity": "high"},
            },
        ]

        for task in tasks:
            await rag_db.add_task(**task)

        # 添加方案
        solutions = [
            {
                "solution_id": "sol_review_001",
                "agent_id": 1,
                "task_type": "review",
                "solution_text": "使用静态分析工具进行代码审查",
                "success_rate": 0.85,
            },
            {
                "solution_id": "sol_planning_001",
                "agent_id": 2,
                "task_type": "planning",
                "solution_text": "使用敏捷方法论进行项目规划",
                "success_rate": 0.80,
            },
        ]

        for solution in solutions:
            await rag_db.add_solution(**solution)

        logger.info(f"✓ 添加了 {len(tasks)} 个任务和 {len(solutions)} 个方案")

        # ────────────────────────────────────────────────────────
        # 第4步：执行工作流
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤4] 执行任务分配工作流...")

        # 测试任务1
        task_request_1 = {
            "task_id": "task_001",
            "task_type": "review",
            "description": "需要进行代码质量检查和审查",
            "priority": "high",
        }

        logger.info("\n【任务1】执行工作流...")
        state_1 = await workflow.allocate_task(
            task_request=task_request_1,
            task_embedding=await rag_db.embedding.embed(task_request_1["description"]),
        )

        logger.info(f"\n任务1分配结果:")
        logger.info(f"  - 工作流ID: {state_1.workflow_id}")
        logger.info(f"  - 成功: {state_1.success}")
        logger.info(f"  - 分配决策: {state_1.allocation_decision}")
        logger.info(f"  - 选定Agent: {state_1.selected_agents}")
        logger.info(f"  - 执行时间: {state_1.get_duration_ms():.2f}ms")

        # 获取工作流统计
        workflow_stats_1 = await workflow.get_workflow_stats(state_1)
        logger.info(f"  - 本地匹配数: {workflow_stats_1['metrics']['local_results_count']}")
        logger.info(f"  - 最佳Agent数: {workflow_stats_1['metrics']['best_agents_count']}")
        logger.info(
            f"  - 置信度: {workflow_stats_1['metrics']['hit_confidence']:.2%}"
        )

        # 测试任务2
        task_request_2 = {
            "task_id": "task_002",
            "task_type": "planning",
            "description": "需要制定项目执行计划和里程碑",
            "priority": "medium",
        }

        logger.info("\n【任务2】执行工作流...")
        state_2 = await workflow.allocate_task(
            task_request=task_request_2,
            task_embedding=await rag_db.embedding.embed(task_request_2["description"]),
        )

        logger.info(f"\n任务2分配结果:")
        logger.info(f"  - 工作流ID: {state_2.workflow_id}")
        logger.info(f"  - 成功: {state_2.success}")
        logger.info(f"  - 分配决策: {state_2.allocation_decision}")
        logger.info(f"  - 选定Agent: {state_2.selected_agents}")
        logger.info(f"  - 执行时间: {state_2.get_duration_ms():.2f}ms")

        # ────────────────────────────────────────────────────────
        # 第5步：处理反馈
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤5] 处理反馈并更新权重...")

        # 任务1反馈
        if state_1.allocation_result.get("record_id"):
            logger.info(f"\n处理任务1反馈...")
            await workflow.process_feedback(
                record_id=state_1.allocation_result["record_id"],
                success_score=0.95,
                feedback_text="代码审查成功完成",
            )

        # 任务2反馈
        if state_2.allocation_result.get("record_id"):
            logger.info(f"处理任务2反馈...")
            await workflow.process_feedback(
                record_id=state_2.allocation_result["record_id"],
                success_score=0.85,
                feedback_text="项目规划完成",
            )

        # 获取更新后的权重
        updated_weights = await rag_db.get_weights()
        logger.info(f"\n更新后的权重: {updated_weights}")

        # ────────────────────────────────────────────────────────
        # 第6步：系统健康检查
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤6] 系统健康检查...")

        health = await workflow.health_check()
        logger.info(f"\n健康状态: {health['status']}")
        logger.info(f"  - Agent数: {health['agents_count']}")
        logger.info(f"  - 任务数: {health['tasks_count']}")
        logger.info(f"  - 方案数: {health['solutions_count']}")
        logger.info(f"  - 记录数: {health['records_count']}")

        # ────────────────────────────────────────────────────────
        # 第7步：保存并关闭
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤7] 保存并关闭数据库...")

        await rag_db.save()
        await rag_db.close()

        logger.info("✓ 数据库已保存并关闭")

        logger.info("\n" + "=" * 80)
        logger.info("第2步演示完成！")
        logger.info("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"\n✗ 演示过程中出错: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(demo())
