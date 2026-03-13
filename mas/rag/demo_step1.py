"""
第1步演示：基础RAG存储（本地FAISS+JSON）

演示内容：
1. 初始化RAG数据库
2. 注册Agent
3. 添加任务模板
4. 添加Agent方案
5. 搜索相似任务
6. 搜索最佳方案
7. 记录成功分配
8. 查询统计信息
"""

import asyncio
import logging
import sys
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from local_rag_database import LocalRAGDatabase


async def demo():
    """运行第1步演示"""

    logger.info("\n" + "=" * 80)
    logger.info("第1步演示：基础RAG存储（本地FAISS+JSON）")
    logger.info("=" * 80)

    try:
        # ────────────────────────────────────────────────────────
        # 第1步：初始化数据库
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤1] 初始化RAG数据库...")

        rag_db = LocalRAGDatabase(
            storage_path="./rag_storage",
            embedding_model="local_hash",  # 使用本地哈希（快速，无依赖）
            embedding_dimension=1536,
        )

        await rag_db.initialize()
        logger.info("✓ RAG数据库初始化完成")

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
        # 第3步：添加任务模板
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤3] 添加任务模板...")

        tasks = [
            {
                "task_id": "task_review_001",
                "task_type": "review",
                "description": "代码审查：检查代码质量、性能和安全性",
                "metadata": {"domain": "code", "priority": "high"},
            },
            {
                "task_id": "task_review_002",
                "task_type": "review",
                "description": "文档审查：验证API文档的准确性和完整性",
                "metadata": {"domain": "docs", "priority": "medium"},
            },
            {
                "task_id": "task_planning_001",
                "task_type": "planning",
                "description": "项目规划：制定项目时间表和资源分配",
                "metadata": {"domain": "project", "complexity": "high"},
            },
            {
                "task_id": "task_planning_002",
                "task_type": "planning",
                "description": "模块设计：设计新功能的系统架构",
                "metadata": {"domain": "architecture", "complexity": "medium"},
            },
            {
                "task_id": "task_dev_001",
                "task_type": "development",
                "description": "实现API端点：开发RESTful API",
                "metadata": {"domain": "backend", "language": "python"},
            },
        ]

        for task in tasks:
            await rag_db.add_task(**task)

        logger.info(f"✓ 添加了 {len(tasks)} 个任务模板")

        # ────────────────────────────────────────────────────────
        # 第4步：添加Agent方案
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤4] 添加Agent方案...")

        solutions = [
            {
                "solution_id": "sol_review_001",
                "agent_id": 1,
                "task_type": "review",
                "solution_text": "使用静态分析工具进行代码审查，包括SonarQube和ESLint",
                "success_rate": 0.85,
                "metadata": {"tools": ["SonarQube", "ESLint"]},
            },
            {
                "solution_id": "sol_review_002",
                "agent_id": 1,
                "task_type": "review",
                "solution_text": "人工代码审查流程，关注设计模式和最佳实践",
                "success_rate": 0.90,
                "metadata": {"approach": "manual", "focus": "design"},
            },
            {
                "solution_id": "sol_planning_001",
                "agent_id": 2,
                "task_type": "planning",
                "solution_text": "使用敏捷方法论，分2周Sprint进行规划",
                "success_rate": 0.80,
                "metadata": {"methodology": "Agile", "sprint_duration": "2weeks"},
            },
            {
                "solution_id": "sol_dev_001",
                "agent_id": 3,
                "task_type": "development",
                "solution_text": "使用FastAPI框架开发RESTful API",
                "success_rate": 0.90,
                "metadata": {"framework": "FastAPI", "language": "python"},
            },
        ]

        for solution in solutions:
            await rag_db.add_solution(**solution)

        logger.info(f"✓ 添加了 {len(solutions)} 个Agent方案")

        # ────────────────────────────────────────────────────────
        # 第5步：搜索相似任务
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤5] 搜索相似任务...")

        # 搜索查询1：代码审查
        query1 = "需要对新代码进行质量检查和审查"
        results1 = await rag_db.search_tasks(query1, task_type="review", top_k=3)

        logger.info(f"\n查询: '{query1}'")
        logger.info(f"找到 {len(results1)} 个相似任务:")
        for task in results1:
            logger.info(f"  - {task['task_id']}: {task['description'][:50]}...")

        # 搜索查询2：规划
        query2 = "需要制定项目的执行计划和里程碑"
        results2 = await rag_db.search_tasks(query2, task_type="planning", top_k=3)

        logger.info(f"\n查询: '{query2}'")
        logger.info(f"找到 {len(results2)} 个相似任务:")
        for task in results2:
            logger.info(f"  - {task['task_id']}: {task['description'][:50]}...")

        # ────────────────────────────────────────────────────────
        # 第6步：搜索最佳方案
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤6] 搜索最佳Agent方案...")

        # 搜索代码审查方案
        query3 = "需要进行代码质量审查"
        solutions1 = await rag_db.search_solutions(query3, task_type="review", top_k=3)

        logger.info(f"\n查询: '{query3}'")
        logger.info(f"找到 {len(solutions1)} 个方案:")
        for sol in solutions1:
            logger.info(
                f"  - Agent {sol['agent_id']}: {sol['solution_text'][:40]}... "
                f"(成功率: {sol.get('success_rate', 0):.2%})"
            )

        # ────────────────────────────────────────────────────────
        # 第7步：记录成功分配
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤7] 记录成功分配...")

        records = [
            {
                "record_id": "rec_001",
                "task_id": "task_review_001",
                "agent_ids": [1],
                "feedback": "代码审查成功完成，发现并修复了3个关键问题",
                "success_score": 0.95,
            },
            {
                "record_id": "rec_002",
                "task_id": "task_planning_001",
                "agent_ids": [2],
                "feedback": "项目规划完成，时间表清晰，资源分配合理",
                "success_score": 0.85,
            },
            {
                "record_id": "rec_003",
                "task_id": "task_dev_001",
                "agent_ids": [3],
                "feedback": "API实现完成，测试覆盖率90%，性能达标",
                "success_score": 0.90,
            },
        ]

        for record in records:
            await rag_db.record_success(**record)

        logger.info(f"✓ 记录了 {len(records)} 个成功分配")

        # ────────────────────────────────────────────────────────
        # 第8步：查询统计信息
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤8] 查询统计信息...")

        stats = await rag_db.get_stats()
        logger.info(f"\nRAG数据库统计:")
        logger.info(f"  - 任务模板数: {stats['tasks']}")
        logger.info(f"  - Agent方案数: {stats['solutions']}")
        logger.info(f"  - 成功记录数: {stats['records']}")
        logger.info(f"  - 注册Agent数: {stats['agents']}")
        logger.info(f"  - 当前权重: {stats['weights']}")

        # ────────────────────────────────────────────────────────
        # 第9步：修改权重
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤9] 修改权重...")

        current_weights = await rag_db.get_weights()
        logger.info(f"当前权重: {current_weights}")

        new_weights = {
            "w_A": 0.25,
            "w_E": 0.25,
            "w_I": 0.25,
            "w_C": 0.25,
        }
        await rag_db.set_weights(new_weights)

        updated_weights = await rag_db.get_weights()
        logger.info(f"更新后权重: {updated_weights}")

        # ────────────────────────────────────────────────────────
        # 第10步：保存和关闭
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤10] 保存并关闭数据库...")

        await rag_db.save()
        await rag_db.close()

        logger.info("✓ 数据库已保存并关闭")

        logger.info("\n" + "=" * 80)
        logger.info("演示完成！")
        logger.info("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"\n✗ 演示过程中出错: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(demo())
