"""P2P 共识协议 — 消息定义与序列化"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Optional


class MsgType(str, Enum):
    # Agent → Registry
    REGISTER = "register"
    DISCOVER = "discover"
    # Registry → Agent
    PEERS = "peers"
    TASK = "task"
    # Agent ↔ Agent (P2P)
    AEIC_RECORD = "aeic_record"
    THETA_UPDATE = "theta_update"
    CONSENSUS_REACHED = "consensus_reached"


@dataclass
class Message:
    type: MsgType
    agent_id: str
    task_id: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def serialize(self) -> str:
        d = asdict(self)
        d["type"] = self.type.value
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def deserialize(cls, raw: str) -> "Message":
        d = json.loads(raw)
        d["type"] = MsgType(d["type"])
        return cls(**d)


def make_register(agent_id: str, model: str, port: int, **capabilities) -> Message:
    return Message(
        type=MsgType.REGISTER,
        agent_id=agent_id,
        payload={"model": model, "port": port, **capabilities},
    )


def make_task(task_id: str, question: str, context: Dict = None) -> Message:
    return Message(
        type=MsgType.TASK,
        agent_id="registry",
        task_id=task_id,
        payload={"question": question, "context": context or {}},
    )


def make_aeic_record(agent_id: str, task_id: str, record: Dict) -> Message:
    return Message(
        type=MsgType.AEIC_RECORD,
        agent_id=agent_id,
        task_id=task_id,
        payload={"record": record},
    )


def make_theta_update(agent_id: str, task_id: str, round_num: int, theta: float) -> Message:
    return Message(
        type=MsgType.THETA_UPDATE,
        agent_id=agent_id,
        task_id=task_id,
        payload={"round": round_num, "theta": theta},
    )


def make_consensus(agent_id: str, task_id: str, result: Dict) -> Message:
    return Message(
        type=MsgType.CONSENSUS_REACHED,
        agent_id=agent_id,
        task_id=task_id,
        payload={"result": result},
    )
