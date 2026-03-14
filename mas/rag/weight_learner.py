"""
权重学习器 - 基于反馈的AEIC四层权重自适应更新

功能：
- 反馈驱动的权重学习
- 梯度上升优化
- 权重历史追踪
- 学习率自适应
- 与consensus.py集成
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WeightSnapshot:
    """权重快照"""

    timestamp: float
    """时间戳"""

    weights: Dict[str, float]
    """权重值 {w_A, w_E, w_I, w_C}"""

    learning_rate: float
    """学习率"""

    feedback_data: Optional[Dict[str, Any]] = None
    """反馈数据"""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp,
            "weights": self.weights,
            "learning_rate": self.learning_rate,
            "feedback_data": self.feedback_data,
        }


class WeightLearner:
    """权重学习器 - AEIC四层权重的自适应学习"""

    def __init__(
        self,
        initial_weights: Optional[Dict[str, float]] = None,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        decay_rate: float = 0.99,
    ):
        """
        初始化权重学习器

        Args:
            initial_weights: 初始权重字典 {w_A, w_E, w_I, w_C}
            learning_rate: 学习率 (0.001 - 0.1 推荐)
            momentum: 动量系数 (梯度动量)
            decay_rate: 衰减率 (学习率衰减)
        """
        # 初始化权重
        self.weights = initial_weights or {
            "w_A": 0.25,  # Ability
            "w_E": 0.25,  # Evidence
            "w_I": 0.25,  # Inference
            "w_C": 0.25,  # Conclusion
        }

        # 学习参数
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay_rate = decay_rate

        # 梯度动量累计
        self.velocity = {k: 0.0 for k in self.weights.keys()}

        # 学习历史
        self.history: List[WeightSnapshot] = []
        self.feedback_history: List[Dict[str, Any]] = []

        # 统计信息
        self.stats = {
            "updates": 0,
            "total_feedback": 0,
            "convergence_metric": 0.0,
        }

        logger.info(f"✓ WeightLearner initialized with learning_rate={learning_rate}")

    # ────────────────────────────────────────────────────────
    # 权重更新方法
    # ────────────────────────────────────────────────────────

    async def update_weights_from_feedback(
        self,
        success_score: float,
        agent_scores: Optional[Dict[int, float]] = None,
        feedback_text: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        根据反馈更新权重

        Args:
            success_score: 成功评分 (0.0-1.0)
            agent_scores: Agent评分 {agent_id: score}
            feedback_text: 反馈文本
            metadata: 额外元数据

        Returns:
            更新后的权重
        """
        try:
            logger.info(
                f"Learning from feedback: success_score={success_score:.2f}"
            )

            # 计算梯度
            gradient = await self._compute_gradient(
                success_score=success_score,
                agent_scores=agent_scores,
            )

            # 应用梯度上升更新
            self._apply_gradient(gradient)

            # 归一化权重
            self._normalize_weights()

            # 记录历史
            await self._record_update(success_score, gradient, feedback_text, metadata)

            self.stats["updates"] += 1
            self.stats["total_feedback"] += 1

            logger.info(f"✓ Weights updated: {self.weights}")

            return self.weights.copy()

        except Exception as e:
            logger.error(f"✗ Failed to update weights: {e}")
            raise

    async def _compute_gradient(
        self,
        success_score: float,
        agent_scores: Optional[Dict[int, float]] = None,
    ) -> Dict[str, float]:
        """
        计算权重梯度

        使用策略梯度方法：
        ∇w_i ∝ sign(success_score - threshold) * magnitude
        """
        try:
            gradient = {}

            # 成功阈值（0-1之间的中点）
            threshold = 0.5

            # 梯度大小 (success_score 与 threshold 的偏差)
            magnitude = abs(success_score - threshold)

            # 梯度方向 (成功则增加权重，失败则减少)
            direction = 1.0 if success_score > threshold else -1.0

            # 计算每个权重的梯度
            # A (Ability): 主要影响因素，敏感度较高
            gradient["w_A"] = direction * magnitude * 1.2

            # E (Evidence): 次要影响因素
            gradient["w_E"] = direction * magnitude * 1.0

            # I (Inference): 依赖于任务匹配度
            gradient["w_I"] = direction * magnitude * 0.8

            # C (Conclusion): 综合权重，稳定性较强
            gradient["w_C"] = direction * magnitude * 0.6

            logger.info(f"Computed gradient: {gradient}")

            return gradient

        except Exception as e:
            logger.error(f"✗ Failed to compute gradient: {e}")
            raise

    def _apply_gradient(self, gradient: Dict[str, float]) -> None:
        """
        应用梯度上升（带动量）

        θ^{t+1} = θ^t + η * (momentum * v + gradient)
        """
        try:
            for key in self.weights.keys():
                # 更新动量
                self.velocity[key] = (
                    self.momentum * self.velocity[key] + gradient.get(key, 0.0)
                )

                # 更新权重
                self.weights[key] += self.learning_rate * self.velocity[key]

                # 确保权重在[0, 1]范围内
                self.weights[key] = max(0.0, min(1.0, self.weights[key]))

        except Exception as e:
            logger.error(f"✗ Failed to apply gradient: {e}")
            raise

    def _normalize_weights(self) -> None:
        """归一化权重使其和为1"""
        try:
            total = sum(self.weights.values())
            if total > 0:
                self.weights = {k: v / total for k, v in self.weights.items()}

        except Exception as e:
            logger.error(f"✗ Failed to normalize weights: {e}")
            raise

    # ────────────────────────────────────────────────────────
    # 批量学习
    # ────────────────────────────────────────────────────────

    async def batch_learn(
        self, feedback_samples: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        批量学习多个反馈样本

        Args:
            feedback_samples: 反馈样本列表
                [{
                    "success_score": 0.9,
                    "agent_scores": {...},
                    "feedback_text": "...",
                }]

        Returns:
            最终权重
        """
        try:
            logger.info(f"Starting batch learning with {len(feedback_samples)} samples")

            for i, sample in enumerate(feedback_samples):
                await self.update_weights_from_feedback(
                    success_score=sample.get("success_score", 0.5),
                    agent_scores=sample.get("agent_scores"),
                    feedback_text=sample.get("feedback_text", ""),
                    metadata=sample.get("metadata"),
                )

                logger.info(f"  [{i+1}/{len(feedback_samples)}] Updated weights")

            logger.info(
                f"✓ Batch learning completed. Final weights: {self.weights}"
            )

            return self.weights.copy()

        except Exception as e:
            logger.error(f"✗ Batch learning failed: {e}")
            raise

    # ────────────────────────────────────────────────────────
    # 历史和分析
    # ────────────────────────────────────────────────────────

    async def _record_update(
        self,
        success_score: float,
        gradient: Dict[str, float],
        feedback_text: str,
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """记录权重更新历史"""
        try:
            import time

            snapshot = WeightSnapshot(
                timestamp=time.time(),
                weights=self.weights.copy(),
                learning_rate=self.learning_rate,
                feedback_data={
                    "success_score": success_score,
                    "gradient": gradient,
                    "feedback_text": feedback_text,
                    "metadata": metadata,
                },
            )

            self.history.append(snapshot)
            self.feedback_history.append(
                {
                    "timestamp": snapshot.timestamp,
                    "success_score": success_score,
                    "gradient": gradient,
                }
            )

        except Exception as e:
            logger.error(f"✗ Failed to record update: {e}")

    async def get_weight_history(
        self, last_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        获取权重历史

        Args:
            last_n: 返回最近n条记录（None表示全部）

        Returns:
            权重历史列表
        """
        history = [s.to_dict() for s in self.history]
        if last_n:
            history = history[-last_n:]
        return history

    async def get_convergence_metrics(self) -> Dict[str, Any]:
        """
        获取收敛指标

        Returns:
            指标字典
        """
        try:
            if len(self.history) < 2:
                return {
                    "samples": len(self.history),
                    "weight_variance": 0.0,
                    "gradient_norm": 0.0,
                    "learning_stability": 0.0,
                }

            # 计算权重变化
            weight_changes = []
            for i in range(1, len(self.history)):
                prev_weights = self.history[i - 1].weights
                curr_weights = self.history[i].weights

                change = np.sqrt(
                    sum(
                        (curr_weights.get(k, 0) - prev_weights.get(k, 0)) ** 2
                        for k in ["w_A", "w_E", "w_I", "w_C"]
                    )
                )
                weight_changes.append(change)

            # 计算梯度范数
            gradient_norms = []
            for feedback in self.feedback_history:
                gradient = feedback.get("gradient", {})
                norm = np.sqrt(sum(v**2 for v in gradient.values()))
                gradient_norms.append(norm)

            metrics = {
                "samples": len(self.history),
                "weight_variance": np.var(weight_changes) if weight_changes else 0.0,
                "weight_change_mean": np.mean(weight_changes)
                if weight_changes
                else 0.0,
                "gradient_norm_mean": np.mean(gradient_norms)
                if gradient_norms
                else 0.0,
                "learning_stability": (
                    1.0 / (1.0 + np.var(weight_changes)) if weight_changes else 0.0
                ),
            }

            self.stats["convergence_metric"] = metrics["learning_stability"]

            return metrics

        except Exception as e:
            logger.error(f"✗ Failed to compute convergence metrics: {e}")
            return {}

    async def save_history(self, filepath: str) -> None:
        """保存学习历史到JSON文件"""
        try:
            history_data = {
                "current_weights": self.weights,
                "learning_stats": self.stats,
                "history": [s.to_dict() for s in self.history],
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)

            logger.info(f"✓ Saved learning history to {filepath}")

        except Exception as e:
            logger.error(f"✗ Failed to save history: {e}")

    # ────────────────────────────────────────────────────────
    # 统计和监控
    # ────────────────────────────────────────────────────────

    async def get_stats(self) -> Dict[str, Any]:
        """获取学习统计信息"""
        try:
            convergence = await self.get_convergence_metrics()

            return {
                "current_weights": self.weights,
                "learning_stats": self.stats,
                "convergence": convergence,
                "history_size": len(self.history),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"✗ Failed to get stats: {e}")
            return {}

    async def reset_weights(
        self, weights: Optional[Dict[str, float]] = None
    ) -> None:
        """
        重置权重

        Args:
            weights: 新权重，如果None则重置为均匀分布
        """
        try:
            if weights is None:
                self.weights = {
                    "w_A": 0.25,
                    "w_E": 0.25,
                    "w_I": 0.25,
                    "w_C": 0.25,
                }
            else:
                self.weights = weights.copy()
                self._normalize_weights()

            # 重置速度
            self.velocity = {k: 0.0 for k in self.weights.keys()}

            logger.info(f"✓ Weights reset to {self.weights}")

        except Exception as e:
            logger.error(f"✗ Failed to reset weights: {e}")
            raise
