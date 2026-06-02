"""
Registry — 注册中心 + 任务协调

Port 8000, 提供:
  POST /register  — agent注册
  GET  /agents    — 发现所有agent
  POST /task      — 提交推理任务 (触发共识协议)
  GET  /result/<task_id> — 获取任务结果
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.similarity import SimilarityCalculator
from core.p2p.protocol import ConsensusProtocol


class RegistryHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass

    def _read_body(self) -> Dict:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        return json.loads(body) if body else {}

    def _respond(self, code: int, data):
        payload = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        registry: Registry = self.server.registry

        if self.path == "/agents":
            self._respond(200, {"agents": registry.agents, "count": len(registry.agents)})
        elif self.path.startswith("/result/"):
            task_id = self.path.split("/result/")[1]
            result = registry.results.get(task_id)
            if result is None:
                self._respond(404, {"error": "task not found or still running"})
            else:
                self._respond(200, result)
        elif self.path == "/status":
            self._respond(200, {
                "status": "running",
                "agents_registered": len(registry.agents),
                "tasks_completed": len(registry.results),
            })
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        registry: Registry = self.server.registry
        body = self._read_body()

        if self.path == "/register":
            agent_info = {
                "agent_id": body["agent_id"],
                "url": body["url"],
                "backend": body.get("backend", "unknown"),
                "persona": body.get("persona", "unknown"),
            }
            registry.agents.append(agent_info)
            self._respond(200, {"status": "registered", "agent_id": body["agent_id"]})

        elif self.path == "/task":
            question = body.get("question", "")
            mode = body.get("mode", "parallel")
            use_stackelberg = body.get("use_stackelberg", True)
            task_id = str(uuid.uuid4())[:8]

            # Run protocol in background thread
            def _run():
                protocol = ConsensusProtocol(registry.agents, registry.sim_calc)
                result = protocol.run(question, mode=mode, use_stackelberg=use_stackelberg)
                result["task_id"] = task_id
                result["question"] = question
                result["mode"] = mode
                result["timestamp"] = time.time()
                # Convert numpy for JSON
                for k in ["trust", "ic", "ec", "ev", "agreement", "theta"]:
                    if k in result and hasattr(result[k], "tolist"):
                        result[k] = result[k].tolist()
                    elif k in result and isinstance(result[k], list):
                        result[k] = [float(x) if hasattr(x, 'item') else x for x in result[k]]
                registry.results[task_id] = result

            t = threading.Thread(target=_run, daemon=True)
            t.start()

            if body.get("sync", False):
                t.join(timeout=120)
                result = registry.results.get(task_id, {"status": "timeout"})
                self._respond(200, result)
            else:
                self._respond(202, {"task_id": task_id, "status": "accepted"})

        else:
            self._respond(404, {"error": "not found"})


class Registry:
    """注册中心 — 维护agent列表，协调共识任务。"""

    def __init__(self, port: int = 8000, sim_method: str = "embedding"):
        self.port = port
        self.agents: List[Dict] = []
        self.results: Dict[str, Dict] = {}
        self.sim_calc = SimilarityCalculator(method=sim_method)
        self._server = None
        self._thread = None

    def start(self):
        self._server = HTTPServer(("0.0.0.0", self.port), RegistryHandler)
        self._server.registry = self
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server = None

    def wait_for_agents(self, n_expected: int, timeout: float = 30.0):
        """等待所有agent注册完成。"""
        t0 = time.time()
        while len(self.agents) < n_expected:
            if time.time() - t0 > timeout:
                raise TimeoutError(
                    f"Only {len(self.agents)}/{n_expected} agents registered in {timeout}s")
            time.sleep(0.1)

    def submit_task(self, question: str, mode: str = "parallel",
                    use_stackelberg: bool = True) -> Dict:
        """直接提交任务并等待结果（进程内调用）。"""
        protocol = ConsensusProtocol(self.agents, self.sim_calc)
        result = protocol.run(question, mode=mode, use_stackelberg=use_stackelberg)
        return result

    def __repr__(self):
        return f"Registry(port={self.port}, agents={len(self.agents)})"


if __name__ == "__main__":
    reg = Registry()
    reg.start()
    print(f"Registry running on port {reg.port}")
    try:
        reg._thread.join()
    except KeyboardInterrupt:
        reg.stop()
