"""Registry Server — 中心化服务发现 + 任务广播"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Dict, Set

from aiohttp import web, WSMsgType

from .protocol import Message, MsgType, make_task

logger = logging.getLogger("registry")


class RegistryServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 9000):
        self.host = host
        self.port = port
        self.agents: Dict[str, Dict] = {}  # agent_id → {host, port, model, capabilities}
        self.ws_clients: Set[web.WebSocketResponse] = set()
        self.app = web.Application()
        self._setup_routes()

    def _setup_routes(self):
        self.app.router.add_post("/register", self._handle_register)
        self.app.router.add_get("/discover", self._handle_discover)
        self.app.router.add_post("/task", self._handle_task)
        self.app.router.add_get("/ws", self._handle_ws)

    async def _handle_register(self, request: web.Request) -> web.Response:
        data = await request.json()
        agent_id = data["agent_id"]
        self.agents[agent_id] = {
            "host": data.get("host", request.remote),
            "port": data["port"],
            "model": data.get("model", "unknown"),
            "capabilities": data.get("capabilities", {}),
        }
        logger.info(f"[Registry] Agent registered: {agent_id} @ port {data['port']}")
        await self._broadcast_ws({"event": "agent_joined", "agent_id": agent_id})
        return web.json_response({"status": "ok", "peers": len(self.agents)})

    async def _handle_discover(self, request: web.Request) -> web.Response:
        exclude = request.query.get("exclude", "")
        peers = [
            {"agent_id": aid, **info}
            for aid, info in self.agents.items()
            if aid != exclude
        ]
        return web.json_response({"peers": peers})

    async def _handle_task(self, request: web.Request) -> web.Response:
        data = await request.json()
        task_id = data["task_id"]
        question = data["question"]
        context = data.get("context", {})
        logger.info(f"[Registry] Task published: {task_id}")

        msg = make_task(task_id, question, context)
        await self._broadcast_ws({"event": "task", "data": json.loads(msg.serialize())})
        return web.json_response({"status": "ok", "task_id": task_id, "agents": len(self.agents)})

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.ws_clients.add(ws)
        logger.info(f"[Registry] WebSocket client connected ({len(self.ws_clients)} total)")

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    pass  # clients don't send commands via ws
                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            self.ws_clients.discard(ws)
        return ws

    async def _broadcast_ws(self, data: dict):
        payload = json.dumps(data, ensure_ascii=False)
        dead = set()
        for ws in self.ws_clients:
            try:
                await ws.send_str(payload)
            except (ConnectionResetError, RuntimeError):
                dead.add(ws)
        self.ws_clients -= dead

    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info(f"[Registry] Started on {self.host}:{self.port}")
        return runner


def run_registry(host: str = "0.0.0.0", port: int = 9000):
    """Standalone entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    server = RegistryServer(host, port)

    async def _main():
        await server.start()
        await asyncio.Event().wait()

    asyncio.run(_main())


if __name__ == "__main__":
    run_registry()
