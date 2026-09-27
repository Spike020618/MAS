"""Dependency-free HTTP API for environments that only provide a DeepSeek key."""
from __future__ import annotations

import argparse
import json
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


class ConsensusHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, workflow=None):
        self.workflow = workflow or ConsensusWorkflow.from_environment()
        super().__init__(address, ConsensusHandler)


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

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/health":
            self._json(200, {"status": "ok", "service": "game-consensus-recsys"})
        elif self.path == "/agent/server_ip":
            self._json(200, {"status": "success", "host": self.server.server_address[0], "port": self.server.server_address[1]})
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
                compact_trace = {
                    "mechanism": response.trace.mechanism,
                    "topology": response.trace.topology,
                    "events": [{"stage": event.stage, "event": event.event, "payload": event.payload} for event in response.trace.events],
                    "plan": response.trace.plan.__dict__ if response.trace.plan else None,
                    "reputations_before": response.trace.reputations_before,
                    "reputations_after": response.trace.reputations_after,
                    "reports": [{"agent_id": report.agent_id, "role": report.role, "action": report.action, "candidate_id": report.candidate_id, "assumption": report.assumption, "evidence": [item.__dict__ for item in report.evidence], "inference": report.inference, "verdict": report.verdict.value, "utility": report.utility, "confidence": report.confidence, "token_cost": report.token_cost, "latency_ms": report.latency_ms} for report in response.trace.reports],
                    "decisions": [{"candidate_id": decision.candidate.movie_id, "title": decision.candidate.title, "score": decision.score, "decision_consensus": decision.decision_consensus, "evidence_consensus": decision.evidence_consensus, "residual_dissent": decision.residual_dissent, "accepted_evidence": [item.__dict__ for item in decision.accepted_evidence], "contributing_agents": decision.contributing_agents} for decision in response.decisions],
                }
                self._json(200, {"status": "success", "query": message, "medias": [m["name"] for m in medias], "medias_info": medias, "chat": f"已为你找到 {len(medias)} 部相关影片。", "trace": compact_trace})
            except Exception as error:
                self._json(503, {"status": "error", "chat": f"推荐服务暂时不可用：{type(error).__name__}"})
        else:
            self._json(404, {"error": "not_found"})

    def do_POST(self):
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
