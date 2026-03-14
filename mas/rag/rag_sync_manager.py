"""
RAG同步管理器 - 实现跨Agent通信的广播和收集机制

功能：
- 广播机制：将消息发送给多个Agent
- 收集机制：收集Agent的响应
- 消息队列：异步消息处理
- 同步机制：保证数据一致性
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict
from datetime import datetime

from .agent_message import AgentMessage, MessageType, MessageStatus

logger = logging.getLogger(__name__)


class RAGSyncManager:
    """跨Agent同步管理器"""

    def __init__(self, agent_id: int, agent_name: str):
        """
        初始化同步管理器

        Args:
            agent_id: 本Agent的ID
            agent_name: 本Agent的名称
        """
        self.agent_id = agent_id
        self.agent_name = agent_name

        # 消息队列
        self.outgoing_queue: asyncio.Queue = asyncio.Queue()
        """待发送消息队列"""

        self.incoming_queue: asyncio.Queue = asyncio.Queue()
        """接收消息队列"""

        # 消息追踪
        self.pending_messages: Dict[str, AgentMessage] = {}
        """待响应的消息 {message_id: message}"""

        self.message_responses: Dict[str, List[AgentMessage]] = defaultdict(list)
        """消息响应 {message_id: [responses]}"""

        self.message_callbacks: Dict[str, Callable] = {}
        """消息回调 {message_id: callback}"""

        # Agent目录
        self.agent_directory: Dict[int, Dict[str, Any]] = {}
        """Agent信息目录 {agent_id: info}"""

        # 统计信息
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "broadcasts_sent": 0,
            "responses_collected": 0,
            "errors": 0,
        }

        logger.info(f"✓ RAGSyncManager initialized for Agent {agent_id} ({agent_name})")

    # ────────────────────────────────────────────────────────
    # Agent目录管理
    # ────────────────────────────────────────────────────────

    async def register_agent(
        self, agent_id: int, agent_name: str, task_types: List[str], success_rate: float
    ) -> None:
        """
        注册Agent到目录

        Args:
            agent_id: Agent ID
            agent_name: Agent名称
            task_types: 支持的任务类型
            success_rate: 成功率
        """
        try:
            self.agent_directory[agent_id] = {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "task_types": task_types,
                "success_rate": success_rate,
                "is_online": True,
                "last_seen": datetime.now().isoformat(),
            }
            logger.info(f"✓ Registered agent {agent_id} ({agent_name})")

        except Exception as e:
            logger.error(f"✗ Failed to register agent: {e}")
            self.stats["errors"] += 1

    async def get_agent_info(self, agent_id: int) -> Optional[Dict[str, Any]]:
        """获取Agent信息"""
        return self.agent_directory.get(agent_id)

    async def list_agents_for_task(self, task_type: str) -> List[int]:
        """
        获取支持特定任务类型的Agent列表

        Args:
            task_type: 任务类型

        Returns:
            Agent ID列表
        """
        agents = []
        for agent_id, info in self.agent_directory.items():
            if agent_id != self.agent_id and task_type in info.get("task_types", []):
                agents.append(agent_id)
        return agents

    # ────────────────────────────────────────────────────────
    # 广播机制
    # ────────────────────────────────────────────────────────

    async def broadcast_task_request(
        self, task_request: Dict[str, Any], target_agents: Optional[List[int]] = None
    ) -> str:
        """
        广播任务请求给其他Agent

        Args:
            task_request: 任务请求内容
            target_agents: 目标Agent列表（None表示广播给所有）

        Returns:
            消息ID
        """
        try:
            # 创建广播消息
            message = AgentMessage(
                message_type=MessageType.TASK_BROADCAST,
                sender_id=self.agent_id,
                sender_name=self.agent_name,
                payload=task_request,
            )

            # 设置接收者
            if target_agents:
                message.receiver_ids = target_agents
            else:
                # 广播给所有其他Agent
                message.receiver_ids = [
                    aid for aid in self.agent_directory.keys() if aid != self.agent_id
                ]

            logger.info(
                f"[Agent {self.agent_id}] Broadcasting task to {len(message.receiver_ids)} agents"
            )

            # 添加到待响应列表
            self.pending_messages[message.message_id] = message

            # 添加到发送队列
            await self.outgoing_queue.put(message)

            self.stats["broadcasts_sent"] += 1

            return message.message_id

        except Exception as e:
            logger.error(f"✗ Broadcast failed: {e}")
            self.stats["errors"] += 1
            raise

    async def broadcast_to_all(self, message: AgentMessage) -> str:
        """
        广播消息给所有其他Agent

        Args:
            message: 消息对象

        Returns:
            消息ID
        """
        try:
            message.sender_id = self.agent_id
            message.sender_name = self.agent_name
            message.receiver_ids = [
                aid for aid in self.agent_directory.keys() if aid != self.agent_id
            ]

            self.pending_messages[message.message_id] = message
            await self.outgoing_queue.put(message)

            logger.info(
                f"[Agent {self.agent_id}] Broadcasted {message.message_type.value} "
                f"to {len(message.receiver_ids)} agents"
            )

            self.stats["broadcasts_sent"] += 1

            return message.message_id

        except Exception as e:
            logger.error(f"✗ Broadcast failed: {e}")
            self.stats["errors"] += 1
            raise

    # ────────────────────────────────────────────────────────
    # 收集机制
    # ────────────────────────────────────────────────────────

    async def collect_responses(
        self, message_id: str, timeout: float = 5.0, min_responses: int = 1
    ) -> List[AgentMessage]:
        """
        收集特定消息的响应

        Args:
            message_id: 消息ID
            timeout: 超时时间（秒）
            min_responses: 最少响应数

        Returns:
            响应消息列表
        """
        try:
            logger.info(
                f"[Agent {self.agent_id}] Collecting responses for {message_id}, "
                f"timeout={timeout}s, min_responses={min_responses}"
            )

            start_time = time.time()
            collected_responses = []

            while time.time() - start_time < timeout:
                # 检查已收集的响应
                collected_responses = self.message_responses.get(message_id, [])

                if len(collected_responses) >= min_responses:
                    break

                # 等待新响应
                await asyncio.sleep(0.1)

            # 清理已回收的消息
            if message_id in self.pending_messages:
                del self.pending_messages[message_id]

            elapsed = time.time() - start_time
            logger.info(
                f"[Agent {self.agent_id}] ✓ Collected {len(collected_responses)} "
                f"responses in {elapsed:.2f}s"
            )

            self.stats["responses_collected"] += len(collected_responses)

            return collected_responses

        except Exception as e:
            logger.error(f"✗ Failed to collect responses: {e}")
            self.stats["errors"] += 1
            return []

    async def wait_for_response(
        self, message_id: str, timeout: float = 10.0
    ) -> Optional[AgentMessage]:
        """
        等待单个响应

        Args:
            message_id: 消息ID
            timeout: 超时时间（秒）

        Returns:
            响应消息，或None如果超时
        """
        try:
            start_time = time.time()

            while time.time() - start_time < timeout:
                responses = self.message_responses.get(message_id, [])
                if responses:
                    response = responses[0]
                    del responses[0]
                    return response

                await asyncio.sleep(0.1)

            logger.warning(f"Response timeout for message {message_id}")
            return None

        except Exception as e:
            logger.error(f"✗ Failed to wait for response: {e}")
            return None

    # ────────────────────────────────────────────────────────
    # 消息处理
    # ────────────────────────────────────────────────────────

    async def process_incoming_message(self, message: AgentMessage) -> None:
        """
        处理接收到的消息

        Args:
            message: 消息对象
        """
        try:
            logger.info(
                f"[Agent {self.agent_id}] Received {message.message_type.value} "
                f"from Agent {message.sender_id}"
            )

            # 根据消息类型处理
            if message.message_type == MessageType.TASK_BROADCAST:
                await self.handle_task_broadcast(message)

            elif message.message_type == MessageType.TASK_RESPONSE:
                await self.handle_task_response(message)

            elif message.message_type == MessageType.FEEDBACK:
                await self.handle_feedback(message)

            self.stats["messages_received"] += 1

        except Exception as e:
            logger.error(f"✗ Failed to process message: {e}")
            self.stats["errors"] += 1

    async def handle_task_broadcast(self, message: AgentMessage) -> None:
        """处理任务广播"""
        logger.info(f"[Agent {self.agent_id}] Handling task broadcast from Agent {message.sender_id}")
        # 这里会在实际应用中被覆盖
        pass

    async def handle_task_response(self, message: AgentMessage) -> None:
        """处理任务响应"""
        if message.reply_to:
            if message.reply_to not in self.message_responses:
                self.message_responses[message.reply_to] = []
            self.message_responses[message.reply_to].append(message)

            logger.info(f"[Agent {self.agent_id}] Recorded response for message {message.reply_to}")

    async def handle_feedback(self, message: AgentMessage) -> None:
        """处理反馈消息"""
        logger.info(
            f"[Agent {self.agent_id}] Received feedback from Agent {message.sender_id}: "
            f"score={message.payload.get('success_score')}"
        )
        # 这里会在实际应用中被覆盖
        pass

    # ────────────────────────────────────────────────────────
    # 统计和监控
    # ────────────────────────────────────────────────────────

    async def get_stats(self) -> Dict[str, Any]:
        """获取同步管理器统计信息"""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "registered_agents": len(self.agent_directory),
            "pending_messages": len(self.pending_messages),
            "stats": self.stats,
            "timestamp": datetime.now().isoformat(),
        }

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        queue_sizes = {
            "outgoing": self.outgoing_queue.qsize(),
            "incoming": self.incoming_queue.qsize(),
        }

        return {
            "agent_id": self.agent_id,
            "status": "healthy",
            "queue_sizes": queue_sizes,
            "pending_messages": len(self.pending_messages),
            "agents_online": sum(
                1 for info in self.agent_directory.values() if info.get("is_online")
            ),
        }
