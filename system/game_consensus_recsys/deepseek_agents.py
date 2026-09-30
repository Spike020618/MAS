"""Logical heterogeneous agents backed by one shared DeepSeek API client."""
from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterable, List

from .deepseek import DeepSeekClient, DeepSeekError
from .domain import CandidateReport, Evidence, MovieCandidate, SearchRequest, Verdict


ROLE_INSTRUCTIONS = {
    "preference": "评估候选与用户显式偏好、当前意图的匹配。不得虚构用户历史。",
    "diversity": "评估候选对列表多样性、长尾覆盖和同质化风险的贡献。",
    "critic": "主动寻找候选与查询之间的冲突、夸大推断和无证据推荐理由。",
}


class DeepSeekReviewAgent:
    def __init__(self, client: DeepSeekClient, role: str, agent_id: str):
        if role not in ROLE_INSTRUCTIONS:
            raise ValueError(f"unsupported DeepSeek role: {role}")
        self.client = client
        self.role = role
        self.agent_id = agent_id

    def review(self, request: SearchRequest, candidates: Iterable[MovieCandidate], *,
               action: str = "aeic_effort", max_tokens: int | None = None) -> List[CandidateReport]:
        candidate_list = list(candidates)
        if not candidate_list:
            return []
        action_instruction = {
            "low_effort": "采用低成本路径：每个候选只给最短假设、一个输入内证据和直接结论，不做追加核验。",
            "aeic_effort": "采用完整 AEIC 路径：分别给出假设、输入内证据、推断和结论。",
            "verify": "采用核验路径：先完成 AEIC，再主动寻找一个反例或约束冲突；后续还会执行独立审计调用。",
            "defer_imitate": "当前没有可见同伴报告，不得假装模仿；依据现有输入做最小判断，证据不足时弃权。",
        }.get(action, "采用完整 AEIC 路径。")
        system_prompt = f"""你是影视推荐多智能体系统中的 {self.role} Agent。
{ROLE_INSTRUCTIONS[self.role]}
执行动作：{action}。{action_instruction}
只允许使用用户请求和候选 JSON 中提供的信息。不得补充外部事实。
必须输出 JSON 对象，格式为 {{"reports":[...]}}。每个候选恰好一条报告，字段为：
candidate_id, assumption, evidence_claims(字符串数组), inference,
verdict(support/oppose/abstain), utility(0到1), confidence(0到1)。
即使反对候选，也要给出基于输入的明确证据。"""
        user_payload = {
            "request": {
                "query": request.query,
                "preferred_genres": request.preferred_genres,
                "required_genres": request.required_genres,
                "excluded_genres": request.excluded_genres,
                "min_year": request.min_year,
                "max_year": request.max_year,
                "top_k": request.top_k,
            },
            "candidates": [
                {
                    "movie_id": item.movie_id,
                    "title": item.title,
                    "year": item.year,
                    "genres": item.genres,
                    "retrieval_score": item.retrieval_score,
                    "metadata": item.metadata,
                }
                for item in candidate_list
            ],
        }
        started = time.perf_counter()
        result = self.client.complete_json(
            system_prompt, json.dumps(user_payload, ensure_ascii=False),
            temperature={"preference": 0.15, "diversity": 0.35, "critic": 0.1}[self.role],
            max_tokens=max_tokens,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        usage = result.get("_usage") or {}
        token_total = int(usage.get("total_tokens") or 0)
        raw_reports = result.get("reports")
        if not isinstance(raw_reports, list):
            raise DeepSeekError("DeepSeek response requires a reports array")
        by_id: Dict[str, MovieCandidate] = {item.movie_id: item for item in candidate_list}
        parsed = []
        seen = set()
        for raw in raw_reports:
            if not isinstance(raw, dict):
                continue
            candidate_id = str(raw.get("candidate_id", ""))
            if candidate_id not in by_id or candidate_id in seen:
                continue
            seen.add(candidate_id)
            claims = raw.get("evidence_claims") or []
            if not isinstance(claims, list):
                claims = [str(claims)]
            parsed.append(CandidateReport(
                agent_id=self.agent_id,
                role=self.role,
                candidate_id=candidate_id,
                assumption=str(raw.get("assumption", "仅依据当前请求和候选信息")),
                evidence=[Evidence(
                    claim=str(claim), source=f"deepseek:{self.role}",
                    reference=f"request_candidate:{candidate_id}", verified=False, strength=0.55,
                ) for claim in claims if str(claim).strip()],
                inference=str(raw.get("inference", "模型未提供推断")),
                verdict=Verdict(str(raw.get("verdict", "abstain"))),
                utility=max(0.0, min(1.0, float(raw.get("utility", 0.5)))),
                confidence=max(0.0, min(1.0, float(raw.get("confidence", 0.5)))),
                action=action,
                token_cost=token_total,
                latency_ms=latency_ms,
            ))
        # Fail closed: missing candidates become abstentions instead of disappearing.
        for candidate in candidate_list:
            if candidate.movie_id not in seen:
                parsed.append(CandidateReport(
                    self.agent_id, self.role, candidate.movie_id,
                    "模型未返回该候选", [], "缺少结构化报告，因此弃权。",
                    Verdict.ABSTAIN, 0.5, 0.0, action=action,
                    token_cost=token_total, latency_ms=latency_ms,
                ))
        return parsed
