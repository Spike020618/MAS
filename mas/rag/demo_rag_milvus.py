"""
RAG 系统演示 - 使用 Milvus + DashScope (生产级)

演示内容：
1. 连接到 Milvus 向量数据库
2. 使用 DashScope API 生成语义向量
3. 演示语义相似度搜索
4. 展示生产级 RAG 的优势
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

from mas.rag.rag_database import RAGDatabase


async def demo():
    """运行演示"""

    logger.info("\n" + "=" * 80)
    logger.info("🚀 RAG 系统演示 - 使用 Milvus + DashScope (生产级)")
    logger.info("=" * 80)

    # 配置信息
    logger.info("\n📋 系统配置:")
    logger.info("  向量数据库: Milvus")
    logger.info("  嵌入模型: DashScope (阿里云)")
    logger.info("  向量维度: 1024 (DashScope text-embedding-v4)")
    logger.info("  集合维度: 1024")
    logger.info("  配置维度: 1024")

    try:
        # ────────────────────────────────────────────────────────
        # 第1步：初始化 RAG 数据库
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤1] 初始化 RAG 数据库...")
        logger.info("  连接地址: localhost:19530 (Milvus)")

        rag_db = RAGDatabase(
            storage_path="./rag_storage_milvus",
            milvus_host="localhost",
            milvus_port=19530,
            # DashScope 配置会自动从环境变量或 config.py 读取
        )

        await rag_db.initialize()

        logger.info("✓ RAG 数据库初始化成功")

        # ────────────────────────────────────────────────────────
        # 第2步：注册 Agent
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤2] 注册 Agent...")

        agents = [
            {
                "agent_id": 1,
                "name": "CodeReviewExpert",
                "task_types": ["code_review", "code_analysis"],
                "success_rate": 0.85,
            },
            {
                "agent_id": 2,
                "name": "ProjectPlanner",
                "task_types": ["project_planning", "scheduling"],
                "success_rate": 0.80,
            },
            {
                "agent_id": 3,
                "name": "DeveloperBot",
                "task_types": ["development", "implementation"],
                "success_rate": 0.90,
            },
        ]

        for agent in agents:
            await rag_db.register_agent(**agent)

        logger.info(f"✓ 注册了 {len(agents)} 个 Agent")

        # ────────────────────────────────────────────────────────
        # 第3步：添加任务（自动生成向量）
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤3] 添加任务（自动生成语义向量）...")

        tasks = [
            {
                "task_id": "task_001",
                "task_type": "code_review",
                "description": "代码审查：检查代码质量和安全性",
            },
            {
                "task_id": "task_002",
                "task_type": "code_review",
                "description": "代码评审：验证代码功能和性能",
            },
            {
                "task_id": "task_003",
                "task_type": "project_planning",
                "description": "项目规划：制定时间表和资源分配",
            },
            {
                "task_id": "task_004",
                "task_type": "code_analysis",
                "description": "代码检查：静态分析和代码规范检查",
            },
            {
                "task_id": "task_005",
                "task_type": "development",
                "description": "功能开发：实现新功能和模块",
            },
        ]

        for task in tasks:
            await rag_db.add_task(**task)
            logger.info(f"  ✓ {task['task_id']}: {task['description']}")

        logger.info(f"✓ 添加了 {len(tasks)} 个任务")

        # ────────────────────────────────────────────────────────
        # 第4步：演示语义相似度搜索
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤4] 演示语义相似度搜索...")
        logger.info("  搜索：'我需要进行代码审查和质量检查'")

        results = await rag_db.search_tasks(
            query_text="我需要进行代码审查和质量检查",
            top_k=5,
        )

        logger.info(f"\n  找到 {len(results)} 个相似任务:")
        logger.info("  " + "-" * 70)

        for i, result in enumerate(results, 1):
            similarity = result["similarity"]
            emoji = "🟢" if similarity > 0.7 else "🟡" if similarity > 0.4 else "🔴"

            logger.info(
                f"  [{i}] {emoji} {result['task_id']}: {result['description']}"
            )
            logger.info(f"      相似度: {similarity:.4f}")

        # ────────────────────────────────────────────────────────
        # 第5步：按类型搜索
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤5] 按任务类型搜索...")
        logger.info("  搜索：'代码检查' (task_type=code_review)")

        results = await rag_db.search_tasks(
            query_text="代码检查",
            task_type="code_review",
            top_k=3,
        )

        logger.info(f"\n  找到 {len(results)} 个 code_review 类型的相似任务:")
        for i, result in enumerate(results, 1):
            logger.info(
                f"  [{i}] {result['task_id']}: {result['description']} "
                f"(相似度: {result['similarity']:.4f})"
            )

        # ────────────────────────────────────────────────────────
        # 第6步：显示系统统计
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤6] 系统统计...")

        stats = await rag_db.get_stats()

        logger.info("\n  向量数据库统计:")
        logger.info(f"    嵌入模型: {stats['embedding_model']}")
        logger.info(f"    向量维度: {stats['embedding_dimension']}")
        logger.info(f"    总 Agent 数: {stats['total_agents']}")
        logger.info(f"    总任务数: {stats['total_tasks']}")
        logger.info(f"    已生成向量数: {stats['embeddings_generated']}")

        if "milvus" in stats:
            milvus = stats["milvus"]
            logger.info(f"\n  Milvus 数据库:")
            logger.info(f"    状态: {milvus.get('status', 'unknown')}")
            logger.info(f"    集合: {milvus.get('collection_name', 'N/A')}")
            logger.info(f"    实体数: {milvus.get('num_entities', 0)}")

        # ────────────────────────────────────────────────────────
        # 第7步：保存并关闭
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤7] 保存并关闭...")

        await rag_db.save()
        await rag_db.close()

        logger.info("✓ 数据库已保存并关闭")

        logger.info("\n" + "=" * 80)
        logger.info("✅ 演示完成！")
        logger.info("=" * 80)

        logger.info("\n【系统优势】")
        logger.info("  ✅ Milvus: 高性能向量数据库，支持千万级向量")
        logger.info("  ✅ DashScope: 阿里云企业级嵌入服务")
        logger.info("  ✅ 语义理解: 能识别不同措辞但含义相同的任务")
        logger.info("  ✅ 生产就绪: 可直接用于生产环境")

        logger.info("\n【访问工具】")
        logger.info("  Attu 管理界面: http://localhost:8000")
        logger.info("  MinIO 控制台: http://localhost:9001")

        logger.info("\n" + "=" * 80 + "\n")

    except Exception as e:
        logger.error(f"\n✗ 演示过程中出错: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(demo())
