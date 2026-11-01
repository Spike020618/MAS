"""Dependency-free HTTP API for environments that only provide a DeepSeek key."""
from __future__ import annotations

import argparse
import cgi
import json
import mimetypes
import socket
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from dataclasses import fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

from .domain import SearchRequest
from .workflow import ConsensusWorkflow


INDEX_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>博弈驱动影视推荐</title>
<style>
:root{color-scheme:dark;font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0b1020;color:#e8ecf4}
body{max-width:1080px;margin:0 auto;padding:32px 20px;background:radial-gradient(circle at 80% 0,#26345f55,transparent 45%)}
h1{margin:0 0 8px;font-size:30px} .sub{color:#9eabc7;margin-bottom:24px}
form{display:grid;grid-template-columns:1fr 130px 130px auto;gap:10px;background:#121a31;padding:16px;border:1px solid #29375e;border-radius:14px}
input,button{border:1px solid #34456f;border-radius:9px;padding:11px 12px;background:#0d1428;color:#eef2ff;font-size:14px}button{background:#5c7cfa;border:0;font-weight:650;cursor:pointer}button:disabled{opacity:.6;cursor:wait}
#status{min-height:22px;margin:18px 0;color:#aebce0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}
.card{background:#121a31;border:1px solid #29375e;border-radius:12px;padding:15px}.card h3{margin:0 0 8px;font-size:17px}.meta{font-size:12px;color:#aebce0}.why{margin-top:10px;line-height:1.5;color:#d9e0f5}
details{margin-top:20px;background:#10182d;border:1px solid #29375e;border-radius:10px;padding:12px}pre{white-space:pre-wrap;overflow:auto;color:#b9c7e7;font-size:12px}
@media(max-width:760px){form{grid-template-columns:1fr 1fr}form input:first-child{grid-column:1/-1}}
</style></head><body>
<h1>博弈驱动的去中心化语义共识推荐</h1><div class="sub">DeepSeek 驱动的多智能体检索、约束校验与证据共识演示</div>
<form id="f"><input id="q" placeholder="例如：推荐几部温暖的科幻电影" required><input id="k" type="number" min="1" max="20" value="5" title="返回数量"><input id="g" placeholder="类型（可选）"><button id="b">开始推荐</button></form>
<div id="status">服务已启动，输入需求开始检索。</div><section id="results" class="grid"></section><details><summary>查看共识轨迹与审计信息</summary><pre id="trace">尚未执行查询</pre></details>
<script>
const f=document.querySelector('#f'),b=document.querySelector('#b'),s=document.querySelector('#status'),out=document.querySelector('#results'),trace=document.querySelector('#trace');
f.addEventListener('submit',async e=>{e.preventDefault();b.disabled=true;s.textContent='正在检索、博弈分配与共识校验…';out.innerHTML='';
 try{const payload={query:q.value,top_k:Number(k.value),preferred_genres:g.value?g.value.split(',').map(x=>x.trim()).filter(Boolean):[]};const r=await fetch('/v1/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const d=await r.json();if(!r.ok)throw Error(d.detail||d.error||'请求失败');
 const items=d.decisions||[]; out.innerHTML=items.map((x,i)=>{const c=x.candidate||{};return `<article class="card"><h3>${i+1}. ${c.title||'未命名'}</h3><div class="meta">${c.year||''}　${(c.genres||[]).join(' · ')}</div><div class="why">共识得分 ${(x.score||0).toFixed(3)}　·　证据共识 ${(x.evidence_consensus||0).toFixed(3)}</div></article>`}).join('')||'<div class="card">没有找到满足条件的候选。</div>';
 trace.textContent=JSON.stringify(d.trace||{},null,2);s.textContent=`完成：返回 ${items.length} 条候选。`;
 }catch(err){s.textContent='请求失败：'+err.message}finally{b.disabled=false}});
</script></body></html>"""

_DASHBOARD = Path(__file__).with_name("web") / "index.html"
if _DASHBOARD.is_file():
    INDEX_HTML = _DASHBOARD.read_text(encoding="utf-8")


class ConsensusHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, workflow=None):
        self.workflow = workflow or ConsensusWorkflow.from_environment()
        self.trace_dir = Path(__file__).resolve().parents[2] / "experiments/results/system_traces"
        self.frontend_dist = Path(__file__).resolve().parents[1] / "frontend/frontend/dist"
        self.templates_dir = Path(__file__).resolve().parents[1] / "frontend/templates"
        self.latest_result = None
        self.latest_sequence = 0
        super().__init__(address, ConsensusHandler)

    def archive(self, response) -> Path:
        """Persist every successful decision package for later audit."""
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        path = self.trace_dir / f"{response.request_id}.json"
        path.write_text(json.dumps(response.to_dict(), ensure_ascii=False, sort_keys=True),
                        encoding="utf-8")
        return path

    def publish(self, payload: Dict[str, Any]) -> None:
        self.latest_sequence += 1
        self.latest_result = payload


class ConsensusHandler(BaseHTTPRequestHandler):
    server: ConsensusHTTPServer

    def _json(self, status: int, payload: Dict[str, Any]):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _browser_origin(self) -> str:
        host = (self.headers.get("Host") or "").split(":", 1)[0]
        if host in {"", "localhost", "127.0.0.1", "0.0.0.0"}:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                probe.connect(("8.8.8.8", 80))
                host = probe.getsockname()[0]
            except OSError:
                host = "127.0.0.1"
            finally:
                probe.close()
        return f"http://{host}:{self.server.server_address[1]}"

    def _file(self, path: Path, cache: bool = False):
        if not path.is_file():
            self._json(404, {"error": "not_found"})
            return
        body = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", mime + ("; charset=utf-8" if mime.startswith("text/") else ""))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _mobile_payload(self, message: str, source: str) -> Dict[str, Any]:
        known_genres = ("动作", "科幻", "剧情", "喜剧", "爱情", "动画", "冒险", "悬疑", "历史", "灾难")
        preferred = [genre for genre in known_genres if genre in message]
        response = self.server.workflow.run(SearchRequest(query=message, preferred_genres=preferred))
        self.server.archive(response)
        medias = []
        for index, decision in enumerate(response.decisions):
            candidate = decision.candidate
            medias.append({
                "id": candidate.movie_id, "name": candidate.title, "year": candidate.year,
                "genres": ", ".join(candidate.genres),
                "rating": round(float(candidate.metadata.get("rating", decision.score * 10)), 1),
                "description": "；".join(e.claim for e in decision.accepted_evidence) or "已通过语义共识校验",
                "views": "-", "duration": "-",
                "img": f"https://picsum.photos/640/360?random={index + 1}", "url": "/agent/media/1",
            })
        raw = response.to_dict()
        trace = raw["trace"]
        trace["decisions"] = raw["decisions"]
        return {"status": "success", "type": "media_search", "source": source,
                "query": message, "medias": [m["name"] for m in medias],
                "medias_info": medias, "chat": f"已为你找到 {len(medias)} 部相关影片。",
                "trace": trace}

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if urlparse(self.path).path == '/workspace':
            self._file(Path(__file__).with_name('web') / 'workspace.html')
            return
        if urlparse(self.path).path == '/v1/research/status':
            from .research_view import inventory
            self._json(200, inventory())
            return
        if urlparse(self.path).path == '/v1/research/case':
            from .research_view import sealed_case
            params = parse_qs(urlparse(self.path).query)
            try:
                stage = (params.get('stage') or ['main'])[0]
                raw_index = (params.get('index') or ['0'])[0]
                if not raw_index.isdecimal(): raise ValueError('invalid case index')
                self._json(200, sealed_case(stage, int(raw_index)))
            except (ValueError, IndexError) as exc:
                self._json(400, {'error': str(exc)})
            except RuntimeError as exc:
                self._json(409, {'error': str(exc)})
            return
        if self.path in ("/", "/index.html"):
            built = self.server.frontend_dist / "index.html"
            if built.is_file():
                self._file(built)
            else:
                body = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
        elif self.path.startswith("/assets/"):
            relative = Path(urlparse(self.path).path).relative_to("/")
            asset = (self.server.frontend_dist / relative).resolve()
            root = self.server.frontend_dist.resolve()
            if root not in asset.parents:
                self._json(403, {"error": "forbidden"})
            else:
                self._file(asset, cache=True)
        elif self.path == "/health":
            self._json(200, {"status": "ok", "service": "game-consensus-recsys"})
        elif self.path == "/agent/server_ip":
            origin = self._browser_origin()
            self._json(200, {"status": "success", "ip": urlparse(origin).hostname,
                             "host": urlparse(origin).hostname,
                             "port": self.server.server_address[1], "origin": origin})
        elif self.path == "/agent/uploade_page":
            self._file(self.server.templates_dir / "uploade.html")
        elif self.path == "/agent/voice_page":
            self._file(self.server.templates_dir / "voice.html")
        elif self.path.startswith("/agent/latest_result"):
            after = int((parse_qs(urlparse(self.path).query).get("after") or [0])[0])
            result = self.server.latest_result if self.server.latest_sequence > after else None
            self._json(200, {"sequence": self.server.latest_sequence, "result": result})
        elif self.path.startswith("/agent/voice_page_search"):
            params = parse_qs(urlparse(self.path).query)
            message = (params.get("message") or [""])[0].strip()
            if not message:
                self._json(422, {"status": "error", "chat": "请输入搜索关键词"})
                return
            try:
                payload = self._mobile_payload(message, "voice_page_search")
                self.server.publish(payload)
                self._json(200, payload)
            except Exception as error:
                self._json(503, {"status": "error", "chat": f"推荐服务暂时不可用：{type(error).__name__}"})
        elif self.path.startswith("/agent/media_search"):
            # Compatibility endpoint used by the existing Youku-style frontend.
            params = parse_qs(urlparse(self.path).query)
            message = (params.get("message") or [""])[0].strip()
            if not message:
                self._json(422, {"status": "error", "chat": "请输入搜索关键词"})
                return
            try:
                # Treat explicit Chinese genre words as hard user preferences.
                known_genres = ("动作", "科幻", "剧情", "喜剧", "爱情", "动画", "冒险", "悬疑", "历史", "灾难")
                preferred = [genre for genre in known_genres if genre in message]
                response = self.server.workflow.run(SearchRequest(query=message, preferred_genres=preferred))
                medias = []
                for index, decision in enumerate(response.decisions):
                    candidate = decision.candidate
                    medias.append({
                        "id": candidate.movie_id,
                        "name": candidate.title,
                        "year": candidate.year,
                        "genres": ", ".join(candidate.genres),
                        "rating": round(float(candidate.metadata.get("rating", decision.score * 10)), 1),
                        "description": "；".join(e.claim for e in decision.accepted_evidence) or "已通过语义共识校验",
                        "views": "-", "duration": "-",
                        "img": f"https://picsum.photos/640/360?random={index + 1}",
                        "url": "/agent/media/1",
                    })
                self.server.archive(response)
                compact_trace = {
                    "mechanism": response.trace.mechanism,
                    "topology": response.trace.topology,
                    "events": [{"stage": event.stage, "event": event.event, "payload": event.payload} for event in response.trace.events],
                    "plan": response.trace.plan.__dict__ if response.trace.plan else None,
                    "reputations_before": response.trace.reputations_before,
                    "reputations_after": response.trace.reputations_after,
                    "reports": [{"agent_id": report.agent_id, "role": report.role, "action": report.action, "candidate_id": report.candidate_id, "assumption": report.assumption, "evidence": [item.__dict__ for item in report.evidence], "inference": report.inference, "verdict": report.verdict.value, "utility": report.utility, "confidence": report.confidence, "token_cost": report.token_cost, "latency_ms": report.latency_ms} for report in response.trace.reports],
                    "decisions": [{"candidate_id": decision.candidate.movie_id, "title": decision.candidate.title, "score": decision.score, "decision_consensus": decision.decision_consensus, "evidence_consensus": decision.evidence_consensus, "residual_dissent": decision.residual_dissent, "accepted_evidence": [item.__dict__ for item in decision.accepted_evidence], "contributing_agents": decision.contributing_agents, "score_components": decision.score_components, "agent_coverage": decision.agent_coverage} for decision in response.decisions],
                }
                payload = {"status": "success", "type": "media_search", "source": "media_search", "query": message, "medias": [m["name"] for m in medias], "medias_info": medias, "chat": f"已为你找到 {len(medias)} 部相关影片。", "trace": compact_trace}
                self.server.publish(payload)
                self._json(200, payload)
            except Exception as error:
                self._json(503, {"status": "error", "chat": f"推荐服务暂时不可用：{type(error).__name__}"})
        else:
            self._json(404, {"error": "not_found"})

    def do_POST(self):
        if self.path == "/agent/upload":
            try:
                content_type = self.headers.get("Content-Type", "")
                if not content_type.startswith("multipart/form-data"):
                    raise ValueError("multipart/form-data required")
                form = cgi.FieldStorage(fp=self.rfile, headers=self.headers,
                                        environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type})
                message = str(form.getfirst("text", "")).strip()
                image = form["image"] if "image" in form else None
                if not message:
                    filename = getattr(image, "filename", "") if image is not None else ""
                    self._json(200, {"status": "success", "message": "图片已接收，请补充文字描述以进行影视检索。", "medias_info": [], "filename": filename})
                    return
                payload = self._mobile_payload(message, "mobile_upload")
                self.server.publish(payload)
                self._json(200, payload)
            except (ValueError, TypeError) as error:
                self._json(422, {"status": "error", "message": str(error)})
            except Exception as error:
                self._json(503, {"status": "error", "message": f"服务暂时不可用：{type(error).__name__}"})
            return
        if self.path != "/v1/search":
            self._json(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise ValueError("request body must be between 1 byte and 1 MB")
            raw = json.loads(self.rfile.read(length).decode("utf-8"))
            allowed = {item.name for item in fields(SearchRequest)}
            unknown = sorted(set(raw) - allowed)
            if unknown:
                raise ValueError(f"unknown fields: {', '.join(unknown)}")
            response = self.server.workflow.run(SearchRequest(**raw))
            self.server.archive(response)
            self._json(200, response.to_dict())
        except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as error:
            self._json(422, {"error": "invalid_request", "detail": str(error)})
        except Exception as error:
            self._json(503, {"error": "service_unavailable", "detail": type(error).__name__})

    def log_message(self, fmt, *args):
        print(f"[http] {self.address_string()} {fmt % args}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ConsensusHTTPServer((args.host, args.port))
    print(f"Game Consensus API listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
