"""
工作流状态定义 - 任务分配工作流的数据模型

定义了工作流各个阶段的状态和转移
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid


@dataclass
class WorkflowState:
    """任务分配工作流的完整状态"""

    # ──────────────────────────────────────────────────────
    # 输入信息
    # ──────────────────────────────────────────────────────
    task_request: Dict[str, Any] = field(default_factory=dict)
    """任务请求信息 {task_id, task_type, description, ...}"""

    task_embedding: List[float] = field(default_factory=list)
    """任务描述的向量表示"""

    # ──────────────────────────────────────────────────────
    # 中间结果
    # ──────────────────────────────────────────────────────
    local_search_results: List[Dict[str, Any]] = field(default_factory=list)
    """本地RAG搜索到的相似任务"""

    best_agents: List[Dict[str, Any]] = field(default_factory=list)
    """搜索到的最佳Agent列表（按能力排序）"""

    agent_scores: Dict[int, float] = field(default_factory=dict)
    """Agent综合评分 {agent_id: score}"""

    remote_requests: List[Dict[str, Any]] = field(default_factory=list)
    """发送给远程Agent的请求"""

    remote_responses: List[Dict[str, Any]] = field(default_factory=list)
    """远程Agent的响应"""

    # ──────────────────────────────────────────────────────
    # 决策过程
    # ──────────────────────────────────────────────────────
    allocation_decision: str = ""
    """分配决策 (local_direct, local_with_scoring, remote_fallback)"""

    allocation_reason: str = ""
    """分配决策的理由"""

    local_hit_confidence: float = 0.0
    """本地命中的置信度 (0.0-1.0)"""

    # ──────────────────────────────────────────────────────
    # 最终结果
    # ──────────────────────────────────────────────────────
    selected_agents: List[int] = field(default_factory=list)
    """最终选定的Agent ID列表"""

    allocation_result: Dict[str, Any] = field(default_factory=dict)
    """分配结果详情 {record_id, agents, scores, timestamp, ...}"""

    success: bool = False
    """工作流是否成功完成"""

    error_message: str = ""
    """错误信息（如果失败）"""

    # ──────────────────────────────────────────────────────
    # 元数据
    # ──────────────────────────────────────────────────────
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """工作流唯一ID"""

    start_time: float = 0.0
    """工作流开始时间"""

    end_time: float = 0.0
    """工作流结束时间"""

    step_timings: Dict[str, float] = field(default_factory=dict)
    """各步骤的执行时间 {step_name: duration_ms}"""

    # ──────────────────────────────────────────────────────
    # 性能追踪
    # ──────────────────────────────────────────────────────
    metrics: Dict[str, Any] = field(default_factory=dict)
    """性能指标 {search_count, agent_count, decision_time, ...}"""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        return {
            "workflow_id": self.workflow_id,
            "task_type": self.task_request.get("task_type"),
            "allocation_decision": self.allocation_decision,
            "selected_agents": self.selected_agents,
            "success": self.success,
            "duration_ms": (self.end_time - self.start_time) * 1000 if self.end_time else 0,
            "error": self.error_message if not self.success else None,
        }

    def get_duration_ms(self) -> float:
        """获取工作流执行时间（毫秒）"""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0


@dataclass
class AllocationScore:
    """Agent分配评分（AEIC四层模型）"""

    agent_id: int
    """Agent ID"""

    ability: float
    """A - Ability: Agent的基础能力评分 (0.0-1.0)"""

    evidence: float
    """E - Evidence: 历史成功率 (0.0-1.0)"""

    inference: float
    """I - Inference: 任务匹配度 (0.0-1.0)"""

    conclusion: float
    """C - Conclusion: 综合评分 (0.0-1.0)"""

    total_score: float = 0.0
    """最终得分（根据权重计算）"""

    reasoning: Dict[str, Any] = field(default_factory=dict)
    """评分理由"""

    def calculate_total(self, weights: Dict[str, float]) -> float:
        """
        根据权重计算最终得分

        Args:
            weights: 权重字典 {"w_A": 0.2, "w_E": 0.3, "w_I": 0.2, "w_C": 0.3}

        Returns:
            最终得分
        """
        self.total_score = (
            weights.get("w_A", 0.25) * self.ability
            + weights.get("w_E", 0.25) * self.evidence
            + weights.get("w_I", 0.25) * self.inference
            + weights.get("w_C", 0.25) * self.conclusion
        )
        return self.total_score

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "agent_id": self.agent_id,
            "ability": self.ability,
            "evidence": self.evidence,
            "inference": self.inference,
            "conclusion": self.conclusion,
            "total_score": self.total_score,
            "reasoning": self.reasoning,
        }
