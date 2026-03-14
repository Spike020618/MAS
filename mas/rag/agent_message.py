"""
Agent通信消息定义 - 跨Agent通信的标准化消息格式

定义了Agent之间通信的所有消息类型和结构
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import uuid


class MessageType(Enum):
    """消息类型枚举"""

    # 请求消息
    TASK_REQUEST = "task_request"  # 任务请求
    TASK_BROADCAST = "task_broadcast"  # 任务广播
    AGENT_QUERY = "agent_query"  # Agent查询
    SOLUTION_REQUEST = "solution_request"  # 方案请求

    # 响应消息
    TASK_RESPONSE = "task_response"  # 任务响应
    SOLUTION_RESPONSE = "solution_response"  # 方案响应
    AGENT_INFO = "agent_info"  # Agent信息

    # 同步消息
    SYNC_REQUEST = "sync_request"  # 同步请求
    SYNC_RESPONSE = "sync_response"  # 同步响应
    KNOWLEDGE_SHARE = "knowledge_share"  # 知识共享

    # 反馈消息
    FEEDBACK = "feedback"  # 反馈消息
    WEIGHT_UPDATE = "weight_update"  # 权重更新


class MessageStatus(Enum):
    """消息状态"""

    PENDING = "pending"  # 待处理
    PROCESSING = "processing"  # 处理中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    TIMEOUT = "timeout"  # 超时


@dataclass
class AgentMessage:
    """Agent通信消息基类"""

    # ──────────────────────────────────────────────────────
    # 消息标识
    # ──────────────────────────────────────────────────────
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """消息唯一ID"""

    message_type: MessageType = MessageType.TASK_REQUEST
    """消息类型"""

    # ──────────────────────────────────────────────────────
    # 发送方信息
    # ──────────────────────────────────────────────────────
    sender_id: int = 0
    """发送Agent的ID"""

    sender_name: str = ""
    """发送Agent的名称"""

    # ──────────────────────────────────────────────────────
    # 接收方信息
    # ──────────────────────────────────────────────────────
    receiver_id: Optional[int] = None
    """接收Agent的ID（None表示广播）"""

    receiver_ids: List[int] = field(default_factory=list)
    """接收Agent的ID列表（用于多播）"""

    # ──────────────────────────────────────────────────────
    # 消息内容
    # ──────────────────────────────────────────────────────
    payload: Dict[str, Any] = field(default_factory=dict)
    """消息负载（具体内容）"""

    # ──────────────────────────────────────────────────────
    # 时间戳和状态
    # ──────────────────────────────────────────────────────
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    """发送时间戳"""

    status: MessageStatus = MessageStatus.PENDING
    """消息状态"""

    # ──────────────────────────────────────────────────────
    # 元数据
    # ──────────────────────────────────────────────────────
    priority: int = 0
    """消息优先级（0-10）"""

    ttl: float = 30.0
    """消息生存时间（秒）"""

    reply_to: Optional[str] = None
    """回复的消息ID"""

    def is_broadcast(self) -> bool:
        """是否是广播消息"""
        return self.receiver_id is None and len(self.receiver_ids) == 0

    def is_multicast(self) -> bool:
        """是否是多播消息"""
        return len(self.receiver_ids) > 1

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "receiver_id": self.receiver_id,
            "receiver_ids": self.receiver_ids,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "status": self.status.value,
            "priority": self.priority,
            "ttl": self.ttl,
            "reply_to": self.reply_to,
        }


@dataclass
class TaskRequestMessage(AgentMessage):
    """任务请求消息"""

    def __init__(self, sender_id: int, sender_name: str, task_request: Dict):
        """
        初始化任务请求消息

        Args:
            sender_id: 发送Agent ID
            sender_name: 发送Agent名称
            task_request: 任务请求内容
        """
        super().__init__(
            message_type=MessageType.TASK_REQUEST,
            sender_id=sender_id,
            sender_name=sender_name,
            payload=task_request,
        )


@dataclass
class TaskResponseMessage(AgentMessage):
    """任务响应消息"""

    def __init__(
        self, sender_id: int, sender_name: str, reply_to: str, response_data: Dict
    ):
        """
        初始化任务响应消息

        Args:
            sender_id: 发送Agent ID
            sender_name: 发送Agent名称
            reply_to: 回复的消息ID
            response_data: 响应内容
        """
        super().__init__(
            message_type=MessageType.TASK_RESPONSE,
            sender_id=sender_id,
            sender_name=sender_name,
            reply_to=reply_to,
            payload=response_data,
        )


@dataclass
class SolutionResponseMessage(AgentMessage):
    """方案响应消息"""

    def __init__(
        self, sender_id: int, sender_name: str, reply_to: str, solution_data: Dict
    ):
        """
        初始化方案响应消息

        Args:
            sender_id: 发送Agent ID
            sender_name: 发送Agent名称
            reply_to: 回复的消息ID
            solution_data: 方案数据
        """
        super().__init__(
            message_type=MessageType.SOLUTION_RESPONSE,
            sender_id=sender_id,
            sender_name=sender_name,
            reply_to=reply_to,
            payload=solution_data,
        )


@dataclass
class FeedbackMessage(AgentMessage):
    """反馈消息"""

    def __init__(
        self,
        sender_id: int,
        sender_name: str,
        target_agent_id: int,
        record_id: str,
        success_score: float,
        feedback_text: str = "",
    ):
        """
        初始化反馈消息

        Args:
            sender_id: 发送Agent ID
            sender_name: 发送Agent名称
            target_agent_id: 目标Agent ID
            record_id: 分配记录ID
            success_score: 成功评分 (0.0-1.0)
            feedback_text: 反馈文本
        """
        super().__init__(
            message_type=MessageType.FEEDBACK,
            sender_id=sender_id,
            sender_name=sender_name,
            receiver_id=target_agent_id,
            payload={
                "record_id": record_id,
                "success_score": success_score,
                "feedback_text": feedback_text,
            },
        )
