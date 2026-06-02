"""
Agent Node — 独立HTTP服务，每个agent拥有自己的LLM和persona

启动后监听 port 800{id+1}，暴露:
  POST /aeic      — 独立AEIC推理
  POST /aeic_ctx  — 带context的AEIC推理 (sequential/star/debate)
  GET  /status    — 健康检查
"""
from __future__ import annotations

import json
import sys
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.llm import LLMClient, AGENT_PERSONAS, LLM_BACKENDS


class AgentHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass

    def _read_body(self) -> Dict:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        return json.loads(body) if body else {}

    def _respond(self, code: int, data: Dict):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/status":
            node: AgentNode = self.server.node
            self._respond(200, {
                "agent_id": node.agent_id,
                "backend": node.backend,
                "persona": node.persona_name,
                "port": node.port,
                "status": "ready",
            })
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        node: AgentNode = self.server.node
        body = self._read_body()

        if self.path == "/aeic":
            question = body.get("question", "")
            result = node.generate_aeic(question)
            self._respond(200, {"agent_id": node.agent_id, "aeic": result})

        elif self.path == "/aeic_ctx":
            question = body.get("question", "")
            context = body.get("context", "")
            result = node.generate_aeic_with_context(question, context)
            self._respond(200, {"agent_id": node.agent_id, "aeic": result})

        else:
            self._respond(404, {"error": "not found"})


class AgentNode:
    """独立Agent节点 — 持有自己的LLM和persona，通过HTTP对外服务。"""

    def __init__(self, agent_id: int, port: int, backend: str = "deepseek",
                 persona_id: int = None):
        self.agent_id = agent_id
        self.port = port
        self.backend = backend
        self.persona_id = persona_id if persona_id is not None else (agent_id % len(AGENT_PERSONAS))
        self.persona_name = AGENT_PERSONAS[self.persona_id]["name"]
        self.llm = LLMClient(backend=backend)
        self._server = None
        self._thread = None

    def generate_aeic(self, question: str) -> Dict:
        return self.llm.generate_aeic(question, persona_id=self.persona_id)

    def generate_aeic_with_context(self, question: str, context: str) -> Dict:
        return self.llm.generate_aeic_with_context(question, context, persona_id=self.persona_id)

    def start(self):
        self._server = HTTPServer(("0.0.0.0", self.port), AgentHandler)
        self._server.node = self
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server = None

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"

    def __repr__(self):
        return f"AgentNode(id={self.agent_id}, port={self.port}, backend={self.backend}, persona={self.persona_name})"


def make_agent_nodes(n_agents: int, base_port: int = 8001) -> list:
    """创建N个agent节点，交替分配LLM后端。"""
    backends = [k for k, v in LLM_BACKENDS.items() if v["api_key"]]
    if not backends:
        backends = ["deepseek"]

    nodes = []
    for i in range(n_agents):
        backend = backends[i % len(backends)]
        node = AgentNode(
            agent_id=i,
            port=base_port + i,
            backend=backend,
            persona_id=i % len(AGENT_PERSONAS),
        )
        nodes.append(node)
    return nodes


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--backend", default="deepseek")
    args = parser.parse_args()

    node = AgentNode(args.id, args.port, args.backend)
    node.start()
    print(f"Agent-{args.id} running on port {args.port} ({args.backend})")
    try:
        node._thread.join()
    except KeyboardInterrupt:
        node.stop()
