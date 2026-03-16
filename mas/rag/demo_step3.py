"""
第3步演示：跨Agent通信 - 广播和收集机制

演示内容：
1. 初始化多个Agent及其同步管理器
2. 注册Agent到目录
3. 发送广播任务请求
4. 收集远程Agent的响应
5. 执行多Agent任务分配
6. 发送反馈
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
from mas.rag.rag_sync_manager import RAGSyncManager
from mas.rag.multi_agent_coordinator import MultiAgentCoordinator


async def demo():
    """运行第3步演示"""

    logger.info("\n" + "=" * 80)
    logger.info("第3步演示：跨Agent通信 - 广播和收集机制")
    logger.info("=" * 80)

    try:
        # ────────────────────────────────────────────────────────
        # 第1步：初始化基础设施
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤1] 初始化基础设施...")

        # 创建共享的RAG数据库
        rag_db = LocalRAGDatabase(
            storage_path="./rag_storage_step3",
            embedding_model="local_hash",
            embedding_dimension=1536,
        )
        await rag_db.initialize()

        # 创建主Agent
        main_agent_id = 0
        main_agent_name = "CoordinatorAgent"
        main_workflow = RAGWorkflow(rag_db)
        main_sync = RAGSyncManager(main_agent_id, main_agent_name)
        coordinator = MultiAgentCoordinator(
            agent_id=main_agent_id,
            agent_name=main_agent_name,
            rag_workflow=main_workflow,
            sync_manager=main_sync,
        )

        logger.info("✓ 基础设施初始化完成")

        # ────────────────────────────────────────────────────────
        # 第2步：注册Agent和任务数据
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤2] 注册Agent和任务数据...")

        # 注册Agent
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
            # 也注册到同步管理器目录
            await main_sync.register_agent(
                agent_id=agent["agent_id"],
                agent_name=agent["name"],
                task_types=agent["task_types"],
                success_rate=agent["success_rate"],
            )

        logger.info(f"✓ 注册了 {len(agents)} 个Agent")

        # 添加任务模板
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
        # 第3步：测试同步管理器的广播机制
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤3] 测试广播机制...")

        # 获取支持"review"的Agent
        review_agents = await main_sync.list_agents_for_task("review")
        logger.info(f"支持'review'的Agent: {review_agents}")

        # 获取支持"planning"的Agent
        planning_agents = await main_sync.list_agents_for_task("planning")
        logger.info(f"支持'planning'的Agent: {planning_agents}")

        # ────────────────────────────────────────────────────────
        # 第4步：执行多Agent任务分配（任务1 - 代码审查）
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤4] 执行多Agent任务分配（任务1：代码审查）...")

        task_request_1 = {
            "task_id": "task_001",
            "task_type": "review",
            "description": "需要进行代码质量检查和安全审查",
            "priority": "high",
        }

        result_1 = await coordinator.allocate_task_with_sync(
            task_request=task_request_1,
            task_embedding=await rag_db.embedding.embed(task_request_1["description"]),
            enable_remote=True,
        )

        logger.info(f"\n任务1分配结果:")
        logger.info(f"  - 分配类型: {result_1.get('type')}")
        logger.info(f"  - 选定Agent: {result_1.get('selected_agents')}")
        if "solution" in result_1:
            logger.info(f"  - 方案: {result_1['solution'].get('agent_name')}")

        # ────────────────────────────────────────────────────────
        # 第5步：执行多Agent任务分配（任务2 - 项目规划）
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤5] 执行多Agent任务分配（任务2：项目规划）...")

        task_request_2 = {
            "task_id": "task_002",
            "task_type": "planning",
            "description": "需要制定项目执行计划和里程碑",
            "priority": "medium",
        }

        result_2 = await coordinator.allocate_task_with_sync(
            task_request=task_request_2,
            task_embedding=await rag_db.embedding.embed(task_request_2["description"]),
            enable_remote=True,
        )

        logger.info(f"\n任务2分配结果:")
        logger.info(f"  - 分配类型: {result_2.get('type')}")
        logger.info(f"  - 选定Agent: {result_2.get('selected_agents')}")
        if "solution" in result_2:
            logger.info(f"  - 方案: {result_2['solution'].get('agent_name')}")

        # ────────────────────────────────────────────────────────
        # 第6步：发送反馈
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤6] 发送反馈...")

        if result_1.get("selected_agents"):
            target_agent = result_1["selected_agents"][0]
            await coordinator.send_feedback_to_agent(
                target_agent_id=target_agent,
                record_id="rec_task1",
                success_score=0.95,
                feedback_text="代码审查完成，质量优秀",
            )

        if result_2.get("selected_agents"):
            target_agent = result_2["selected_agents"][0]
            await coordinator.send_feedback_to_agent(
                target_agent_id=target_agent,
                record_id="rec_task2",
                success_score=0.85,
                feedback_text="项目规划完成，计划合理",
            )

        logger.info("✓ 反馈已发送")

        # ────────────────────────────────────────────────────────
        # 第7步：获取系统状态
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤7] 获取系统状态...")

        status = await coordinator.get_system_status()
        logger.info(f"\n同步管理器统计:")
        sync_stats = status.get("sync_manager", {}).get("stats", {})
        logger.info(f"  - 消息发送: {sync_stats.get('broadcasts_sent', 0)}")
        logger.info(f"  - 消息接收: {sync_stats.get('messages_received', 0)}")
        logger.info(f"  - 响应收集: {sync_stats.get('responses_collected', 0)}")

        # ────────────────────────────────────────────────────────
        # 第8步：健康检查
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤8] 系统健康检查...")

        health = await coordinator.health_check()
        logger.info(f"\n健康检查结果: {health['status']}")
        logger.info(f"  - 注册Agent数: {health['sync_manager'].get('agents_online', 0)}")

        # ────────────────────────────────────────────────────────
        # 第9步：保存并关闭
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤9] 保存并关闭...")

        await rag_db.save()
        await rag_db.close()

        logger.info("✓ 数据库已保存并关闭")

        logger.info("\n" + "=" * 80)
        logger.info("第3步演示完成！")
        logger.info("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"\n✗ 演示过程中出错: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(demo())
