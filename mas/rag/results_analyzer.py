"""
结果分析器 - 对比实验结果的统计分析

功能：
- 性能指标计算
- 统计对比
- 收敛分析
- 结果可视化数据生成
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExperimentMetrics:
    """实验指标"""
    total_tasks: int
    successful_tasks: int
    success_rate: float
    avg_success_score: float
    avg_allocation_time_ms: float
    optimal_allocation_rate: float  # 选择最优Agent的比例
    convergence_stability: float    # 收敛稳定性
    algorithm_name: str


class ResultsAnalyzer:
    """结果分析器"""

    def __init__(self):
        """初始化结果分析器"""
        logger.info("✓ ResultsAnalyzer initialized")

    def compute_metrics(
        self,
        results: List[Dict[str, Any]],
        algorithm_name: str,
    ) -> ExperimentMetrics:
        """
        计算实验指标

        Args:
            results: 实验结果列表
            algorithm_name: 算法名称

        Returns:
            实验指标
        """
        try:
            if not results:
                raise ValueError("Empty results")

            # 基础指标
            total_tasks = len(results)
            successful_tasks = sum(1 for r in results if r.get("success_score", 0) > 0.5)
            success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0

            # 平均成功分数
            success_scores = [r.get("success_score", 0) for r in results]
            avg_success_score = np.mean(success_scores) if success_scores else 0.0

            # 平均分配时间
            allocation_times = [r.get("time_ms", 0) for r in results]
            avg_allocation_time = np.mean(allocation_times) if allocation_times else 0.0

            # 最优分配率
            optimal_allocations = sum(1 for r in results if r.get("is_optimal", False))
            optimal_rate = optimal_allocations / total_tasks if total_tasks > 0 else 0.0

            # 收敛稳定性（成功分数的标准差越小越稳定）
            if len(success_scores) > 1:
                convergence_stability = 1.0 / (1.0 + np.std(success_scores))
            else:
                convergence_stability = 0.5

            metrics = ExperimentMetrics(
                total_tasks=total_tasks,
                successful_tasks=successful_tasks,
                success_rate=success_rate,
                avg_success_score=avg_success_score,
                avg_allocation_time_ms=avg_allocation_time,
                optimal_allocation_rate=optimal_rate,
                convergence_stability=convergence_stability,
                algorithm_name=algorithm_name,
            )

            logger.info(f"✓ Computed metrics for {algorithm_name}")
            logger.info(f"  - Success rate: {success_rate:.2%}")
            logger.info(f"  - Avg success score: {avg_success_score:.4f}")
            logger.info(f"  - Optimal allocation rate: {optimal_rate:.2%}")

            return metrics

        except Exception as e:
            logger.error(f"✗ Failed to compute metrics: {e}")
            raise

    def compare_algorithms(
        self,
        metrics_list: List[ExperimentMetrics],
    ) -> Dict[str, Any]:
        """
        对比多个算法

        Args:
            metrics_list: 多个算法的指标列表

        Returns:
            对比分析结果
        """
        try:
            comparison = {
                "algorithms": {},
                "winner": None,
                "rankings": {},
            }

            # 记录各算法指标
            for metrics in metrics_list:
                comparison["algorithms"][metrics.algorithm_name] = {
                    "success_rate": metrics.success_rate,
                    "avg_success_score": metrics.avg_success_score,
                    "avg_allocation_time_ms": metrics.avg_allocation_time_ms,
                    "optimal_allocation_rate": metrics.optimal_allocation_rate,
                    "convergence_stability": metrics.convergence_stability,
                }

            # 计算排名
            rankings = {
                "success_rate": sorted(
                    metrics_list,
                    key=lambda m: m.success_rate,
                    reverse=True,
                ),
                "avg_success_score": sorted(
                    metrics_list,
                    key=lambda m: m.avg_success_score,
                    reverse=True,
                ),
                "avg_allocation_time_ms": sorted(
                    metrics_list,
                    key=lambda m: m.avg_allocation_time_ms,
                ),  # 越小越好
                "optimal_allocation_rate": sorted(
                    metrics_list,
                    key=lambda m: m.optimal_allocation_rate,
                    reverse=True,
                ),
            }

            # 生成排名报告
            comparison["rankings"] = {
                metric_name: [m.algorithm_name for m in sorted_metrics]
                for metric_name, sorted_metrics in rankings.items()
            }

            # 确定总体赢家（基于多个指标的加权）
            scores = {}
            for metrics in metrics_list:
                # 加权评分：成功率(40%) + 平均分数(30%) + 最优率(20%) + 稳定性(10%)
                weighted_score = (
                    metrics.success_rate * 0.4
                    + metrics.avg_success_score * 0.3
                    + metrics.optimal_allocation_rate * 0.2
                    + metrics.convergence_stability * 0.1
                )
                scores[metrics.algorithm_name] = weighted_score

            comparison["winner"] = max(scores, key=scores.get)
            comparison["weighted_scores"] = scores

            logger.info(f"✓ Comparison analysis completed")
            logger.info(f"  - Winner: {comparison['winner']}")

            return comparison

        except Exception as e:
            logger.error(f"✗ Failed to compare algorithms: {e}")
            raise

    def generate_report(
        self,
        metrics_list: List[ExperimentMetrics],
        comparison: Dict[str, Any],
    ) -> str:
        """
        生成实验报告

        Args:
            metrics_list: 指标列表
            comparison: 对比结果

        Returns:
            报告文本
        """
        try:
            report = []
            report.append("=" * 80)
            report.append("对比实验结果报告")
            report.append("=" * 80)
            report.append("")

            # 各算法详细指标
            report.append("【各算法详细指标】")
            report.append("-" * 80)

            for metrics in metrics_list:
                report.append(f"\n算法: {metrics.algorithm_name}")
                report.append(f"  - 总任务数: {metrics.total_tasks}")
                report.append(f"  - 成功任务数: {metrics.successful_tasks}")
                report.append(f"  - 成功率: {metrics.success_rate:.2%}")
                report.append(f"  - 平均成功分数: {metrics.avg_success_score:.4f}")
                report.append(f"  - 平均分配时间: {metrics.avg_allocation_time_ms:.2f}ms")
                report.append(f"  - 最优分配率: {metrics.optimal_allocation_rate:.2%}")
                report.append(f"  - 收敛稳定性: {metrics.convergence_stability:.4f}")

            # 排名分析
            report.append("\n" + "=" * 80)
            report.append("【性能排名】")
            report.append("-" * 80)

            for metric_name, ranking in comparison["rankings"].items():
                report.append(f"\n{metric_name}排名:")
                for i, algo_name in enumerate(ranking, 1):
                    report.append(f"  {i}. {algo_name}")

            # 总体赢家
            report.append("\n" + "=" * 80)
            report.append("【总体评估】")
            report.append("-" * 80)
            report.append(f"\n总体赢家: {comparison['winner']}")
            report.append("\n加权评分:")
            for algo_name, score in comparison["weighted_scores"].items():
                report.append(f"  - {algo_name}: {score:.4f}")

            report.append("\n" + "=" * 80)

            return "\n".join(report)

        except Exception as e:
            logger.error(f"✗ Failed to generate report: {e}")
            raise

    def save_report(self, report: str, filepath: str) -> None:
        """保存报告到文件"""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"✓ Saved report to {filepath}")
        except Exception as e:
            logger.error(f"✗ Failed to save report: {e}")
            raise
