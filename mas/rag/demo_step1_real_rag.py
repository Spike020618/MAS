"""
第1步演示（升级版）：真正的RAG存储 - 使用SentenceTransformers语义向量

演示内容：
1. 初始化真正的RAG数据库（SentenceTransformers）
2. 向量化任务和方案
3. 语义相似度搜索
4. 展示语义向量的优势
"""

import asyncio
import logging
import sys
import os
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from mas.rag.local_rag_database import LocalRAGDatabase


def cosine_similarity(v1, v2):
    """计算余弦相似度"""
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


async def demo():
    """运行第1步演示（真正的RAG版本）"""

    logger.info("\n" + "=" * 80)
    logger.info("第1步演示（升级版）：真正的RAG存储 - 语义向量")
    logger.info("=" * 80)

    try:
        # ────────────────────────────────────────────────────────
        # 第1步：初始化真正的RAG数据库（使用SentenceTransformers）
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤1] 初始化真正的RAG数据库...")
        logger.info("  使用向量模型: SentenceTransformers（真正的语义向量）")

        rag_db = LocalRAGDatabase(
            storage_path="./rag_storage_real_rag",
            embedding_model="sentence-transformers",  # ✅ 真正的语义向量！
            embedding_dimension=384,
        )

        await rag_db.initialize()

        logger.info("✓ RAG数据库初始化完成")
        logger.info(f"  向量模型: {rag_db.embedding.model_name}")
        logger.info(f"  向量维度: {rag_db.embedding.get_dimension()}")

        # ────────────────────────────────────────────────────────
        # 第2步：注册Agent
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤2] 注册Agent...")

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

        logger.info(f"✓ 注册了 {len(agents)} 个Agent")

        # ────────────────────────────────────────────────────────
        # 第3步：演示语义相似度搜索
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤3] 演示语义相似度搜索...")
        logger.info("\n这是真正RAG的核心：语义向量能够识别相似含义的文本！")

        # 定义相关的任务
        test_tasks = [
            ("任务A", "代码审查：检查代码质量和安全性"),
            ("任务B", "代码评审：验证代码功能和性能"),  # 与A语义相似
            ("任务C", "项目规划：制定时间表和资源分配"),  # 与A语义不同
            ("任务D", "代码检查：静态分析和代码规范检查"),  # 与A语义相似
        ]

        # 向量化所有任务
        logger.info("\n向量化任务...")
        task_vectors = {}
        for name, description in test_tasks:
            vec = await rag_db.embedding.embed(description)
            task_vectors[name] = vec
            logger.info(f"  ✓ {name}: {description}")

        # 计算语义相似度
        logger.info("\n计算语义相似度（与'任务A'对比）...")
        logger.info("-" * 70)

        base_vec = task_vectors["任务A"]
        for name, description in test_tasks[1:]:
            similarity = cosine_similarity(base_vec, task_vectors[name])
            logger.info(f"  {name}: {similarity:.4f}")
            if similarity > 0.7:
                logger.info(f"    → 🟢 高度相似（语义接近）")
            elif similarity > 0.4:
                logger.info(f"    → 🟡 中等相似")
            else:
                logger.info(f"    → 🔴 低度相似（语义不同）")

        # ────────────────────────────────────────────────────────
        # 第4步：添加任务模板和方案
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤4] 添加任务模板和方案...")

        tasks = [
            {
                "task_id": "task_review_001",
                "task_type": "code_review",
                "description": "代码审查：检查代码质量、性能和安全性",
            },
            {
                "task_id": "task_planning_001",
                "task_type": "project_planning",
                "description": "项目规划：制定项目时间表和资源分配",
            },
        ]

        for task in tasks:
            await rag_db.add_task(**task)

        solutions = [
            {
                "solution_id": "sol_review_001",
                "agent_id": 1,
                "task_type": "code_review",
                "solution_text": "使用静态分析工具进行代码审查",
                "success_rate": 0.85,
            },
            {
                "solution_id": "sol_planning_001",
                "agent_id": 2,
                "task_type": "project_planning",
                "solution_text": "使用敏捷方法论进行项目规划",
                "success_rate": 0.80,
            },
        ]

        for solution in solutions:
            await rag_db.add_solution(**solution)

        logger.info(f"✓ 添加了 {len(tasks)} 个任务和 {len(solutions)} 个方案")

        # ────────────────────────────────────────────────────────
        # 第5步：演示真正的RAG搜索
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤5] 演示真正的RAG搜索...")

        test_query = "我需要进行代码审查和质量检查"
        logger.info(f"\n查询: '{test_query}'")

        # 搜索相似任务
        query_vec = await rag_db.embedding.embed(test_query)
        logger.info(f"\n搜索相似任务（基于语义向量）...")

        for task_dict in [{"task_id": t["task_id"], "description": t["description"]} for t in tasks]:
            task_vec = await rag_db.embedding.embed(task_dict["description"])
            similarity = cosine_similarity(query_vec, task_vec)
            logger.info(
                f"  {task_dict['task_id']}: {similarity:.4f} "
                f"({'✓ 匹配' if similarity > 0.5 else '✗ 不匹配'})"
            )

        # ────────────────────────────────────────────────────────
        # 第6步：性能对比
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤6] 演示RAG的优势...")
        logger.info("-" * 70)

        logger.info("\n❌ 使用哈希（当前默认）的问题:")
        logger.info("   • 无法理解文本语义")
        logger.info("   • '代码审查'和'代码评审'被视为完全不同的向量")
        logger.info("   • 只能精确匹配，不能找到相似文本")

        logger.info("\n✅ 使用SentenceTransformers（真正RAG）的优势:")
        logger.info("   • 理解文本语义")
        logger.info("   • '代码审查'和'代码评审'被识别为相似")
        logger.info("   • 可以找到语义相似的任务，即使措辞不同")
        logger.info("   • 更精准的任务分配")

        # ────────────────────────────────────────────────────────
        # 第7步：保存并关闭
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤7] 保存并关闭...")

        await rag_db.save()
        await rag_db.close()

        logger.info("✓ 数据库已保存并关闭")

        logger.info("\n" + "=" * 80)
        logger.info("第1步演示（真正RAG版本）完成！")
        logger.info("=" * 80)
        logger.info("\n关键收获:")
        logger.info("  ✅ 真正的RAG使用语义向量来理解文本含义")
        logger.info("  ✅ 可以找到语义相似的任务，即使措辞不同")
        logger.info("  ✅ 相比哈希方案，性能和准确度大幅提升")
        logger.info("\n" + "=" * 80 + "\n")

    except Exception as e:
        logger.error(f"\n✗ 演示过程中出错: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # 首先检查SentenceTransformers是否已安装
    try:
        import sentence_transformers
    except ImportError:
        logger.error("❌ SentenceTransformers 未安装")
        logger.error("\n请运行以下命令安装：")
        logger.error("  pip install sentence-transformers")
        logger.error("\n或运行升级脚本：")
        logger.error("  bash upgrade_to_real_rag.sh")
        sys.exit(1)

    asyncio.run(demo())
