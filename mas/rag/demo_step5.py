"""
第5步演示：对比实验 - 三种算法的性能对比

演示内容：
1. 初始化实验环境
2. 生成合成数据集
3. 运行三种算法：
   - 算法1：贪心基线 (不使用RAG，不使用学习)
   - 算法2：RAG检索 (使用RAG，不使用学习)
   - 算法3：RAG+权重学习 (使用RAG和学习)
4. 收集和分析结果
5. 生成对比报告
"""

import asyncio
import logging
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from local_rag_database import LocalRAGDatabase
from experiment_runner import ExperimentRunner
from results_analyzer import ResultsAnalyzer


async def demo():
    """运行第5步演示"""

    logger.info("\n" + "=" * 80)
    logger.info("第5步演示：对比实验 - 三种算法的性能对比")
    logger.info("=" * 80)

    try:
        # ────────────────────────────────────────────────────────
        # 第1步：初始化实验环境
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤1] 初始化实验环境...")

        # 创建RAG数据库
        rag_db = LocalRAGDatabase(
            storage_path="./rag_storage_step5",
            embedding_model="local_hash",
            embedding_dimension=1536,
        )
        await rag_db.initialize()

        logger.info("✓ 实验环境初始化完成")

        # ────────────────────────────────────────────────────────
        # 第2步：运行对比实验
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤2] 运行对比实验...")

        runner = ExperimentRunner(rag_db)

        # 运行实验（使用较小的数据集进行快速演示）
        experiment_results = await runner.run_experiment(
            num_agents=4,
            num_tasks=30,  # 较小的数据集用于演示
            seed=42,
        )

        logger.info("✓ 实验完成")

        # ────────────────────────────────────────────────────────
        # 第3步：分析结果
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤3] 分析结果...")

        analyzer = ResultsAnalyzer()

        # 为每个算法计算指标
        metrics_list = []

        for algo_name, algo_results in experiment_results.items():
            logger.info(f"\n分析算法: {algo_name}")

            results = algo_results["results"]
            metrics = analyzer.compute_metrics(results, algo_name)
            metrics_list.append(metrics)

        # ────────────────────────────────────────────────────────
        # 第4步：对比分析
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤4] 对比分析...")

        comparison = analyzer.compare_algorithms(metrics_list)

        # ────────────────────────────────────────────────────────
        # 第5步：生成报告
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤5] 生成报告...")

        report = analyzer.generate_report(metrics_list, comparison)

        # 显示报告
        logger.info("\n" + report)

        # ────────────────────────────────────────────────────────
        # 第6步：详细对比表格
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤6] 详细指标对比表...")

        logger.info("\n【各算法详细指标对比】")
        logger.info("-" * 100)

        # 表头
        header = (
            f"{'算法':<20} "
            f"{'成功率':<12} "
            f"{'平均分数':<12} "
            f"{'分配时间':<12} "
            f"{'最优率':<12} "
            f"{'稳定性':<12}"
        )
        logger.info(header)
        logger.info("-" * 100)

        # 数据行
        for metrics in metrics_list:
            row = (
                f"{metrics.algorithm_name:<20} "
                f"{metrics.success_rate:>11.2%} "
                f"{metrics.avg_success_score:>11.4f} "
                f"{metrics.avg_allocation_time_ms:>11.2f}ms "
                f"{metrics.optimal_allocation_rate:>11.2%} "
                f"{metrics.convergence_stability:>11.4f}"
            )
            logger.info(row)

        logger.info("-" * 100)

        # ────────────────────────────────────────────────────────
        # 第7步：关键发现
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤7] 关键发现...")

        logger.info("\n【关键发现】")
        logger.info("-" * 100)

        # 找出各指标的最优算法
        best_success_rate = max(metrics_list, key=lambda m: m.success_rate)
        best_score = max(metrics_list, key=lambda m: m.avg_success_score)
        best_time = min(metrics_list, key=lambda m: m.avg_allocation_time_ms)
        best_optimal = max(metrics_list, key=lambda m: m.optimal_allocation_rate)
        best_stability = max(metrics_list, key=lambda m: m.convergence_stability)

        logger.info(f"\n成功率最高: {best_success_rate.algorithm_name} "
                   f"({best_success_rate.success_rate:.2%})")
        logger.info(f"平均分数最高: {best_score.algorithm_name} "
                   f"({best_score.avg_success_score:.4f})")
        logger.info(f"分配速度最快: {best_time.algorithm_name} "
                   f"({best_time.avg_allocation_time_ms:.2f}ms)")
        logger.info(f"最优分配率: {best_optimal.algorithm_name} "
                   f"({best_optimal.optimal_allocation_rate:.2%})")
        logger.info(f"稳定性最好: {best_stability.algorithm_name} "
                   f"({best_stability.convergence_stability:.4f})")

        logger.info(f"\n🏆 总体赢家: {comparison['winner']}")

        # ────────────────────────────────────────────────────────
        # 第8步：保存报告
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤8] 保存报告...")

        report_path = "./rag_storage_step5/experiment_report.txt"
        analyzer.save_report(report, report_path)

        logger.info(f"✓ 报告已保存到: {report_path}")

        # ────────────────────────────────────────────────────────
        # 第9步：保存并关闭
        # ────────────────────────────────────────────────────────

        logger.info("\n[步骤9] 保存并关闭...")

        await rag_db.save()
        await rag_db.close()

        logger.info("✓ 数据库已保存并关闭")

        # ────────────────────────────────────────────────────────
        # 总结
        # ────────────────────────────────────────────────────────

        logger.info("\n" + "=" * 80)
        logger.info("第5步演示完成！")
        logger.info("=" * 80)

        logger.info("\n【实验总结】")
        logger.info(f"  ✓ 运行了3种算法")
        logger.info(f"  ✓ 测试了{best_success_rate.total_tasks}个任务")
        logger.info(f"  ✓ 对比了多个关键性能指标")
        logger.info(f"  ✓ 确定了最优算法: {comparison['winner']}")

        logger.info("\n【算法对比总结】")
        logger.info("  贪心基线:")
        logger.info("    ✓ 速度快 (无RAG开销)")
        logger.info("    ✗ 准确度低 (缺乏智能匹配)")
        logger.info("    ✗ 不能学习 (固定权重)")

        logger.info("  RAG检索:")
        logger.info("    ✓ 准确度中等 (向量匹配)")
        logger.info("    ✓ 速度适中")
        logger.info("    ✗ 不能自适应 (固定权重)")

        logger.info("  RAG+权重学习:")
        logger.info("    ✓ 准确度高 (向量匹配+智能评分)")
        logger.info("    ✓ 自适应学习 (权重自动优化)")
        logger.info("    ✓ 性能提升 (随时间改进)")

        logger.info("\n" + "=" * 80 + "\n")

    except Exception as e:
        logger.error(f"\n✗ 演示过程中出错: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(demo())
