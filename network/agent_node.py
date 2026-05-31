"""Agent P2P Node — 去中心化共识参与者"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Set

import aiohttp
from aiohttp import web, WSMsgType
import numpy as np

from .protocol import (
    Message, MsgType,
    make_register, make_aeic_record, make_theta_update, make_consensus,
)

logger = logging.getLogger("agent")


class AgentNode:
    """P2P Agent 节点 — 独立生成推理并参与分布式共识"""

    def __init__(
        self,
        agent_id: str,
        model: str = "simulated",
        port: int = 9001,
        registry_url: str = "http://localhost:9000",
        eta: float = 0.05,
        gamma: float = 0.1,
        convergence_threshold: float = 1e-4,
        max_rounds: int = 20,
    ):
        self.agent_id = agent_id
        self.model = model
        self.port = port
        self.registry_url = registry_url
        self.eta = eta
        self.gamma = gamma
        self.convergence_threshold = convergence_threshold
        self.max_rounds = max_rounds

        self.peers: List[Dict] = []
        self.peer_ws: Dict[str, aiohttp.ClientWebSocketResponse] = {}

        # Per-task state
        self.current_task: Optional[Dict] = None
        self.my_record: Optional[Dict] = None
        self.peer_records: Dict[str, Dict] = {}  # agent_id → AEIC record
        self.theta: float = np.random.uniform(0.3, 0.7)
        self.peer_thetas: Dict[str, float] = {}

        # Stats
        self.messages_sent = 0
        self.messages_received = 0
        self.consensus_result: Optional[Dict] = None
        self._peer_sessions: Dict[str, aiohttp.ClientSession] = {}

        # Events for coordination
        self._records_ready = asyncio.Event()
        self._consensus_done = asyncio.Event()
        self._app = web.Application()
        self._app.router.add_get("/ws", self._handle_peer_ws)

    # ── Lifecycle ──────────────────────────────────────

    async def start(self):
        """启动本地 WebSocket server 并注册到 Registry"""
        runner = web.AppRunner(self._app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()
        logger.info(f"[{self.agent_id}] Listening on port {self.port}")

        await self._register()
        await self._discover_peers()
        logger.info(f"[{self.agent_id}] Registered, {len(self.peers)} peers discovered")
        return runner

    async def _register(self):
        async with aiohttp.ClientSession() as session:
            payload = {
                "agent_id": self.agent_id,
                "host": "localhost",
                "port": self.port,
                "model": self.model,
            }
            async with session.post(f"{self.registry_url}/register", json=payload) as resp:
                data = await resp.json()
                logger.debug(f"[{self.agent_id}] Register response: {data}")

    async def _discover_peers(self):
        async with aiohttp.ClientSession() as session:
            url = f"{self.registry_url}/discover?exclude={self.agent_id}"
            async with session.get(url) as resp:
                data = await resp.json()
                self.peers = data["peers"]

    # ── P2P WebSocket Server ───────────────────────────

    async def _handle_peer_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                self.messages_received += 1
                message = Message.deserialize(msg.data)
                await self._dispatch(message)
            elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                break
        return ws

    async def _dispatch(self, message: Message):
        if message.type == MsgType.AEIC_RECORD:
            await self._on_peer_record(message)
        elif message.type == MsgType.THETA_UPDATE:
            await self._on_theta_update(message)
        elif message.type == MsgType.CONSENSUS_REACHED:
            self.consensus_result = message.payload.get("result")
            self._consensus_done.set()

    # ── P2P Communication ──────────────────────────────

    async def _connect_to_peer(self, peer: Dict) -> Optional[aiohttp.ClientWebSocketResponse]:
        peer_id = peer["agent_id"]
        if peer_id in self.peer_ws:
            return self.peer_ws[peer_id]
        try:
            session = aiohttp.ClientSession()
            ws = await session.ws_connect(f"ws://localhost:{peer['port']}/ws")
            self.peer_ws[peer_id] = ws
            self._peer_sessions[peer_id] = session
            return ws
        except Exception as e:
            logger.warning(f"[{self.agent_id}] Cannot connect to {peer_id}: {e}")
            await session.close()
            return None

    async def _broadcast(self, message: Message):
        payload = message.serialize()
        for peer in self.peers:
            ws = await self._connect_to_peer(peer)
            if ws:
                try:
                    await ws.send_str(payload)
                    self.messages_sent += 1
                except Exception:
                    self.peer_ws.pop(peer["agent_id"], None)

    # ── Task Handling ──────────────────────────────────

    async def on_task(self, task: Dict):
        """接收任务，生成 AEIC 推理记录并广播"""
        self.current_task = task
        self.peer_records.clear()
        self.peer_thetas.clear()
        self._records_ready.clear()
        self._consensus_done.clear()

        self.my_record = self._generate_aeic(task)
        logger.info(f"[{self.agent_id}] Generated AEIC record for task {task.get('task_id', '?')}")

        msg = make_aeic_record(self.agent_id, task.get("task_id", ""), self.my_record)
        await self._broadcast(msg)

    def _generate_aeic(self, task: Dict) -> Dict:
        """生成 AEIC 推理（当前为模拟，可替换为 LLM API 调用）"""
        question = task.get("question", "")
        context = task.get("context", {})

        # Simulated multi-agent diversity
        import hashlib
        seed = int(hashlib.md5(f"{self.agent_id}{question}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        a = f"Analyzing '{question}' under {self.model} reasoning framework, assuming standard logical axioms apply."
        e_items = [
            f"Evidence-{i+1}: contextual data point supporting analysis ({self.agent_id})"
            for i in range(rng.integers(2, 5))
        ]
        i_text = f"Through {'deductive' if rng.random() > 0.5 else 'inductive'} reasoning over {len(e_items)} evidence points, the logical chain suggests a {'positive' if rng.random() > 0.3 else 'nuanced'} conclusion."
        c = f"Conclusion: Based on systematic analysis, the answer to '{question[:50]}...' is determined with {'high' if rng.random() > 0.4 else 'moderate'} confidence."

        return {"A": a, "E": e_items, "I": i_text, "C": c}

    # ── Consensus Protocol ─────────────────────────────

    async def _on_peer_record(self, message: Message):
        """收到 peer 的 AEIC record"""
        peer_id = message.agent_id
        record = message.payload.get("record", {})
        self.peer_records[peer_id] = record
        logger.debug(f"[{self.agent_id}] Got record from {peer_id} ({len(self.peer_records)}/{len(self.peers)})")

        if len(self.peer_records) >= len(self.peers):
            self._records_ready.set()

    async def wait_for_records(self, timeout: float = 30.0):
        """等待所有 peer 的 AEIC records 到齐"""
        try:
            await asyncio.wait_for(self._records_ready.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"[{self.agent_id}] Timeout waiting for records, got {len(self.peer_records)}/{len(self.peers)}")

    async def run_consensus(self) -> Dict:
        """执行分布式 Stackelberg 共识"""
        from core.consensus import ConsensusEngine

        all_records = [self.my_record] + list(self.peer_records.values())
        engine = ConsensusEngine(similarity_method="embedding")
        evaluation = engine.evaluate(all_records)

        history = []
        for round_num in range(1, self.max_rounds + 1):
            # 广播当前 θ
            msg = make_theta_update(self.agent_id, self.current_task.get("task_id", ""), round_num, float(self.theta))
            await self._broadcast(msg)

            # 等待 peer θ 到达
            await asyncio.sleep(0.15)

            # 本地 Stackelberg 更新
            old_theta = self.theta
            peer_pressure = sum(
                (self.theta - pt) for pt in self.peer_thetas.values()
            ) / max(len(self.peer_thetas), 1)

            gradient = evaluation["utility"] / 100.0 * (1.0 - self.theta)
            self.theta = self.theta + self.eta * gradient - self.gamma * peer_pressure
            self.theta = float(np.clip(self.theta, 0.05, 0.95))

            energy = self._local_energy()
            change = abs(self.theta - old_theta)

            history.append({
                "round": round_num,
                "theta": self.theta,
                "energy": energy,
                "change": change,
            })

            logger.info(f"[{self.agent_id}] Round {round_num}: θ={self.theta:.4f}, E={energy:.6f}, Δ={change:.6f}")

            if energy < self.convergence_threshold:
                logger.info(f"[{self.agent_id}] Converged at round {round_num}")
                break

        result = {
            "agent_id": self.agent_id,
            "final_theta": self.theta,
            "rounds": len(history),
            "final_energy": history[-1]["energy"] if history else 0,
            "evaluation": evaluation,
            "history": history,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
        }

        # 广播共识结果
        msg = make_consensus(self.agent_id, self.current_task.get("task_id", ""), {
            "theta": self.theta,
            "similarity": evaluation["avg_similarity"],
            "decision": evaluation["decision"],
        })
        await self._broadcast(msg)
        self.consensus_result = result
        self._consensus_done.set()
        return result

    async def _on_theta_update(self, message: Message):
        """接收 peer 的 θ 更新"""
        peer_id = message.agent_id
        theta = message.payload.get("theta", 0.5)
        self.peer_thetas[peer_id] = theta

    def _local_energy(self) -> float:
        """计算本地共识能量 E_i = (1/2) Σ_j (θ_i - θ_j)²"""
        if not self.peer_thetas:
            return float('inf')  # no data → not converged
        diffs = [(self.theta - pt) ** 2 for pt in self.peer_thetas.values()]
        return sum(diffs) / (2.0 * len(diffs))

    # ── Cleanup ────────────────────────────────────────

    async def close(self):
        for ws in self.peer_ws.values():
            await ws.close()
        for session in self._peer_sessions.values():
            await session.close()
        self.peer_ws.clear()
        self._peer_sessions.clear()
