"""
LLM 调用模块 — DeepSeek API (OpenAI-compatible)

使用方式:
  from core.llm import LLMClient, AGENT_PERSONAS
  client = LLMClient()
  aeic = client.generate_aeic("问题", persona_id=0)
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Dict, List, Optional

import requests


def _load_env():
    """Load .env file if python-dotenv is not available."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())


_load_env()

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY", "")
ZHIPU_BASE_URL = os.environ.get("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
ZHIPU_MODEL = os.environ.get("ZHIPU_MODEL", "glm-4-flash")

ALI_API_KEY = os.environ.get("ALI_API_KEY", "")
ALI_BASE_URL = os.environ.get("ALI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
ALI_EMBED_MODEL = os.environ.get("ALI_EMBED_MODEL", "text-embedding-v3")
ALI_CHAT_MODEL = os.environ.get("ALI_CHAT_MODEL", "qwen-plus")

# 多LLM后端配置
LLM_BACKENDS = {
    "deepseek": {"api_key": DEEPSEEK_API_KEY, "base_url": DEEPSEEK_BASE_URL, "model": DEEPSEEK_MODEL},
    "deepseek_reasoner": {"api_key": DEEPSEEK_API_KEY, "base_url": DEEPSEEK_BASE_URL, "model": "deepseek-reasoner"},
    "zhipu_flash": {"api_key": ZHIPU_API_KEY, "base_url": ZHIPU_BASE_URL, "model": ZHIPU_MODEL},
    "ali_qwen": {"api_key": ALI_API_KEY, "base_url": ALI_BASE_URL, "model": ALI_CHAT_MODEL},
}

# Last-resort addresses observed from authoritative/public DNS during successful
# project runs. They are used only when system DNS, UDP DNS, and DoH all fail;
# the HTTPS URL remains the hostname, so TLS still verifies the intended host.
KNOWN_FALLBACK_IPS = {
    "api.deepseek.com": ["36.147.63.115", "111.12.215.79", "39.156.8.67"],
    "open.bigmodel.cn": ["60.205.172.105"],
    "dashscope.aliyuncs.com": ["47.74.214.77"],
}


def install_dns_fallback(hostname: str, dns_server: str = "223.5.5.5") -> bool:
    """系统 DNS 解析失败时，给 socket.getaddrinfo 打补丁（公共 DNS）。

    解析顺序：1) dig @公共DNS（UDP）；2) AliDNS DoH（HTTPS，UDP 被断时用）。
    补丁只影响 TCP 连接目标，TLS SNI 与证书校验仍用原域名，安全语义不变。
    """
    import socket
    import subprocess
    try:
        socket.getaddrinfo(hostname, 443)
        return False  # 系统 DNS 正常，无需补丁
    except socket.gaierror:
        pass
    import re as _re
    _ipv4 = _re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
    ips: list = []
    try:
        out = subprocess.run(
            ["dig", "+short", f"@{dns_server}", hostname],
            capture_output=True, text=True, timeout=10).stdout
        ips = [ln.strip() for ln in out.splitlines()
               if _ipv4.match(ln.strip())]
    except Exception:
        ips = []
    if not ips:
        # DoH 回退：UDP 53 被断时走 HTTPS（223.5.5.5 是 IP，免 DNS）
        try:
            resp = requests.get(
                f"https://{dns_server}/resolve",
                params={"name": hostname, "type": "A"}, timeout=10,
                verify=False)
            data = resp.json()
            ips = [a["data"] for a in data.get("Answer", [])
                   if a.get("type") == 1 and _ipv4.match(a.get("data", ""))]
        except Exception:
            ips = []
    if not ips:
        ips = list(KNOWN_FALLBACK_IPS.get(hostname, []))
    if not ips:
        return False
    # Rotate across processes/calls so a stale edge address does not become a
    # single point of failure.
    ip = ips[int(time.time()) % len(ips)]
    orig = socket.getaddrinfo

    def patched(host, *args, **kwargs):
        if host == hostname:
            host = ip
        return orig(host, *args, **kwargs)

    socket.getaddrinfo = patched
    print(f"  [dns_fallback] {hostname} -> {ip}")
    return True


# ── Agent Personas ───────────────────────────────────────────
AGENT_PERSONAS = [
    {
        "id": 0,
        "name": "Analyst",
        "name_zh": "分析师",
        "system": (
            "你是一个严谨的分析师。你注重逻辑推理和证据链的完整性。"
            "请仔细分析每一个前提条件，确保推理步骤无遗漏。"
            "只在证据充分时才给出肯定结论。"
        ),
        "temperature": 0.3,
    },
    {
        "id": 1,
        "name": "Skeptic",
        "name_zh": "怀疑者",
        "system": (
            "你是一个怀疑论者。你倾向于质疑常见观点，寻找反例和边界情况。"
            "如果证据不充分，你会明确指出不确定性。"
            "不要轻信表面信息，深入思考是否有反面证据。"
        ),
        "temperature": 0.5,
    },
    {
        "id": 2,
        "name": "Creative",
        "name_zh": "创意者",
        "system": (
            "你是一个创造性思考者。你善于从不同角度看问题，找到隐含的联系。"
            "你会考虑非传统的解释和可能性，但最终结论仍基于证据。"
            "尝试从跨学科的角度分析问题。"
        ),
        "temperature": 0.5,
    },
    {
        "id": 3,
        "name": "Expert",
        "name_zh": "专家",
        "system": (
            "你是一个领域专家。你依赖深厚的专业知识来分析问题。"
            "注重领域特定的细节和术语准确性。"
            "用专业知识来验证常见说法是否正确。"
        ),
        "temperature": 0.4,
    },
    {
        "id": 4,
        "name": "Pragmatist",
        "name_zh": "实用者",
        "system": (
            "你是一个实用主义者。你关注问题的实际意义和常识判断。"
            "不会过度复杂化简单问题，但也不会忽略关键细节。"
            "用最简洁的推理路径得出结论。"
        ),
        "temperature": 0.4,
    },
    {
        "id": 5,
        "name": "Devil's Advocate",
        "name_zh": "反方",
        "system": (
            "你是一个严格的审查者。你会主动寻找推理中的漏洞和反面证据。"
            "你必须同时列出支持和反对的论据，然后基于证据强度做出判断。"
            "你的价值在于发现别人忽略的反面证据，但最终结论必须遵循证据。"
        ),
        "temperature": 0.4,
    },
    {
        "id": 6,
        "name": "Synthesizer",
        "name_zh": "综合者",
        "system": (
            "你是一个综合分析者。你善于整合多方面信息，权衡不同观点。"
            "你会明确指出证据的强度和局限性。"
            "给出平衡、全面的结论，并标明置信度。"
        ),
        "temperature": 0.5,
    },
]

# ── AEIC Prompts ─────────────────────────────────────────────
AEIC_SYSTEM_BASE = """你是一个多智能体共识系统中的推理节点。对于给定的是非判断题，你必须按AEIC四层结构进行严格推理：

A (Assumption): 明确问题的核心前提和关键判断点（1-2句话）
E (Evidence): 列出2-4条具体的事实证据。每条必须是可验证的具体事实或数据，不能笼统概括
I (Inference): 基于证据逐步推理。先考虑支持"Yes"的论据，再考虑支持"No"的论据，然后权衡得出结论（2-3句话）
C (Conclusion): 必须严格以"Yes"或"No"开头（不允许用其他词开头），然后给出一句核心理由

重要原则：结论必须基于证据推理而非直觉或流行说法。如果证据与常见看法矛盾，以证据为准。

{persona_instruction}

请严格输出JSON格式（不要加markdown代码块）：
{{"A": "假设文本", "E": ["证据1", "证据2", ...], "I": "推理文本", "C": "Yes/No: 理由"}}"""

AEIC_USER_TEMPLATE = "请对以下问题进行AEIC推理：\n\n{question}"


class LLMClient:
    """LLM client — OpenAI-compatible API (DeepSeek, ZhiPu, etc.)."""

    def __init__(self, api_key: str = None, base_url: str = None,
                 model: str = None, max_tokens: int = 800,
                 backend: str = "deepseek"):
        if backend in LLM_BACKENDS and api_key is None:
            cfg = LLM_BACKENDS[backend]
            api_key = cfg["api_key"]
            base_url = base_url or cfg["base_url"]
            model = model or cfg["model"]

        self.api_key = api_key or DEEPSEEK_API_KEY
        self.base_url = (base_url or DEEPSEEK_BASE_URL).rstrip("/")
        self.model = model or DEEPSEEK_MODEL
        self.backend = backend
        self.max_tokens = max_tokens
        self._session = None
        self._call_count = 0
        self.last_usage = {}
        self.last_latency_s = None
        self.last_status_code = None

        if not self.api_key:
            raise ValueError(
                f"API key required for backend '{backend}'. "
                "Set the corresponding env var or pass api_key parameter."
            )

    @property
    def session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            })
        return self._session

    def chat(self, messages: List[Dict[str, str]],
             temperature: float = 0.7, **kwargs) -> str:
        """Send a chat completion request."""
        if self.base_url.rstrip("/").endswith(("/v1", "/v2", "/v3", "/v4")):
            url = f"{self.base_url}/chat/completions"
        else:
            url = f"{self.base_url}/v1/chat/completions"
        model = kwargs.get("model", self.model)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if kwargs.get("response_format"):
            payload["response_format"] = kwargs["response_format"]
        # deepseek-reasoner 不支持 temperature 等采样参数
        reasoner = "reasoner" in str(model).lower()
        if reasoner:
            payload.pop("temperature", None)
        req_timeout = 180 if reasoner else 90

        for attempt in range(5):
            try:
                t0 = time.time()
                resp = self.session.post(url, json=payload, timeout=req_timeout)
                self.last_latency_s = time.time() - t0
                self.last_status_code = resp.status_code
                resp.raise_for_status()
                data = resp.json()
                self._call_count += 1
                self.last_usage = data.get("usage") or {}
                msg = data["choices"][0]["message"]
                content = msg.get("content") or msg.get("reasoning_content") or ""
                return content
            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError) as e:
                if attempt < 4:
                    # DNS can degrade after a process starts. Re-evaluate the
                    # fallback on every connection failure instead of only at
                    # experiment startup.
                    try:
                        from urllib.parse import urlparse
                        host = urlparse(self.base_url).hostname
                        if host:
                            install_dns_fallback(host)
                    except Exception:
                        pass
                    time.sleep(2 ** attempt)
                    continue
                raise
            except requests.exceptions.HTTPError as e:
                if resp.status_code == 429:
                    time.sleep(3 * (attempt + 1))
                    continue
                # Provider validation messages are essential for diagnosing
                # backend-compatibility failures. Bound and normalize the body;
                # request headers (including credentials) are never included.
                detail = " ".join((resp.text or "").split())[:500]
                raise requests.exceptions.HTTPError(
                    f"HTTP {resp.status_code} from {self.backend}: {detail}",
                    response=resp,
                ) from e
        raise RuntimeError(f"API call failed after 5 attempts (last status: 429)")

    def generate_aeic(self, question: str, persona_id: int = 0,
                      evidence_hints: List[str] = None,
                      temperature_override: float = None) -> Dict:
        """Generate AEIC reasoning for a question using specified persona.

        Args:
            question: The yes/no question to reason about
            persona_id: Index into AGENT_PERSONAS (0-6)
            evidence_hints: Optional background context

        Returns:
            {"A": str, "E": [str, ...], "I": str, "C": str}
        """
        persona = AGENT_PERSONAS[persona_id % len(AGENT_PERSONAS)]
        system_prompt = AEIC_SYSTEM_BASE.format(
            persona_instruction=f"你的分析风格：{persona['system']}"
        )
        user_msg = AEIC_USER_TEMPLATE.format(question=question)
        if evidence_hints:
            user_msg += "\n\n参考背景：\n" + "\n".join(f"- {e}" for e in evidence_hints)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        temp = temperature_override if temperature_override is not None else persona["temperature"]
        raw = self.chat(messages, temperature=temp)
        return self._parse_aeic_response(raw, question, persona_id)

    def generate_aeic_with_context(self, question: str, context: str,
                                    persona_id: int = 0,
                                    temperature_override: float = None) -> Dict:
        """Generate AEIC reasoning with upstream context injected.

        Used by sequential/star/debate modes where agents receive
        prior conclusions or other agents' outputs.

        Args:
            question: The yes/no question
            context: Formatted context string (e.g., prior agent's conclusion)
            persona_id: Index into AGENT_PERSONAS
        """
        persona = AGENT_PERSONAS[persona_id % len(AGENT_PERSONAS)]
        system_prompt = AEIC_SYSTEM_BASE.format(
            persona_instruction=f"你的分析风格：{persona['system']}"
        )
        user_msg = f"{context}\n\n问题：{question}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        temp = temperature_override if temperature_override is not None else persona["temperature"]
        raw = self.chat(messages, temperature=temp)
        return self._parse_aeic_response(raw, question, persona_id)

    def _parse_aeic_response(self, raw: str, question: str,
                             persona_id: int) -> Dict:
        """Parse LLM response into AEIC dict, with fallback."""
        if raw is None:
            return {
                "A": f"Analysis of: {question[:80]}",
                "E": ["API call failed"],
                "I": "Unable to generate inference due to API error",
                "C": "No: insufficient data",
            }
        text = raw.strip()

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            result = json.loads(text)
            if isinstance(result.get("E"), str):
                result["E"] = [result["E"]]
            return {
                "A": str(result.get("A", "")),
                "E": list(result.get("E", [])),
                "I": str(result.get("I", "")),
                "C": str(result.get("C", "")),
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            import re
            # Try balanced brace matching
            for m in re.finditer(r'\{', text):
                start = m.start()
                depth = 0
                for k in range(start, len(text)):
                    if text[k] == '{':
                        depth += 1
                    elif text[k] == '}':
                        depth -= 1
                    if depth == 0:
                        candidate = text[start:k+1]
                        try:
                            result = json.loads(candidate)
                            if "C" in result:
                                if isinstance(result.get("E"), str):
                                    result["E"] = [result["E"]]
                                return {
                                    "A": str(result.get("A", "")),
                                    "E": list(result.get("E", [])) if isinstance(result.get("E"), list) else [str(result.get("E", ""))],
                                    "I": str(result.get("I", "")),
                                    "C": str(result.get("C", "")),
                                }
                        except (json.JSONDecodeError, TypeError):
                            pass
                        break

            # Final fallback: try to extract stance from raw text
            raw_lower = raw.lower().strip()
            if raw_lower.startswith("yes") or "answer is yes" in raw_lower:
                fallback_c = "Yes: " + raw[:100]
            elif raw_lower.startswith("no") or "answer is no" in raw_lower:
                fallback_c = "No: " + raw[:100]
            else:
                fallback_c = raw[-200:] if len(raw) > 200 else raw
            return {
                "A": f"Analysis of: {question[:80]}",
                "E": [raw[:150]],
                "I": raw[:300],
                "C": fallback_c,
            }

    @property
    def call_count(self) -> int:
        return self._call_count

    def close(self):
        if self._session:
            self._session.close()
            self._session = None

    def __del__(self):
        self.close()

    def __repr__(self):
        return f"LLMClient(backend={self.backend!r}, model={self.model!r})"


class MultiLLMPool:
    """多LLM后端池 — 每个agent随机分配不同的LLM，增加推理多样性。"""

    def __init__(self, backends: List[str] = None,
                 agent_map: Dict[int, str] = None,
                 temperature_map: Dict[int, float] = None,
                 seed: int = 42):
        if agent_map is not None:
            all_backends = sorted(set(agent_map.values()))
            self.backends = all_backends
            self.clients = {b: LLMClient(backend=b) for b in all_backends}
            self.agent_map = agent_map
        else:
            if backends is None:
                backends = [k for k, v in LLM_BACKENDS.items() if v["api_key"]]
            self.backends = backends
            self.clients = {b: LLMClient(backend=b) for b in backends}
            self.agent_map = None
        self.temperature_map = temperature_map or {}
        self._rng = random.Random(seed)
        self._call_count = 0

    def get_client(self, agent_id: int = None) -> LLMClient:
        """为指定agent分配LLM后端。"""
        if self.agent_map is not None and agent_id is not None:
            backend = self.agent_map.get(agent_id, self.backends[0])
            return self.clients[backend]
        if agent_id is not None:
            idx = agent_id % len(self.backends)
        else:
            idx = self._rng.randint(0, len(self.backends) - 1)
        return self.clients[self.backends[idx]]

    def get_backend_name(self, agent_id: int) -> str:
        """返回指定agent的后端名称。"""
        if self.agent_map is not None:
            return self.agent_map.get(agent_id, self.backends[0])
        return self.backends[agent_id % len(self.backends)]

    def generate_aeic(self, question: str, persona_id: int = 0,
                      agent_id: int = 0,
                      evidence_hints: List[str] = None) -> Dict:
        """使用分配给该agent的LLM生成AEIC。"""
        client = self.get_client(agent_id)
        temp = self.temperature_map.get(agent_id)
        result = client.generate_aeic(question, persona_id=persona_id,
                                      evidence_hints=evidence_hints,
                                      temperature_override=temp)
        self._call_count += 1
        return result

    def generate_aeic_with_context(self, question: str, context: str,
                                    persona_id: int = 0,
                                    agent_id: int = 0) -> Dict:
        """使用分配给该agent的LLM生成带context的AEIC。"""
        client = self.get_client(agent_id)
        temp = self.temperature_map.get(agent_id)
        result = client.generate_aeic_with_context(
            question, context, persona_id, temperature_override=temp)
        self._call_count += 1
        return result

    @property
    def call_count(self) -> int:
        return sum(c.call_count for c in self.clients.values())

    def close(self):
        for c in self.clients.values():
            c.close()

    def __repr__(self):
        return f"MultiLLMPool(backends={self.backends})"
