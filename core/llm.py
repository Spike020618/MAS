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

# 多LLM后端配置 — 3种后端轮换，模拟异构agent网络
# Agent 0,3,6 → deepseek (deepseek-chat)
# Agent 1,4,7 → zhipu_strong (glm-5.1)
# Agent 2,5,8 → zhipu_flash (glm-4-flash, 免费/弱模型)
LLM_BACKENDS = {
    "deepseek": {"api_key": DEEPSEEK_API_KEY, "base_url": DEEPSEEK_BASE_URL, "model": DEEPSEEK_MODEL},
    "zhipu_strong": {"api_key": ZHIPU_API_KEY, "base_url": ZHIPU_BASE_URL, "model": "glm-5.1"},
    "zhipu_flash": {"api_key": ZHIPU_API_KEY, "base_url": ZHIPU_BASE_URL, "model": "glm-4-flash"},
}


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
        "temperature": 0.8,
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
        "temperature": 0.6,
    },
    {
        "id": 5,
        "name": "Devil's Advocate",
        "name_zh": "反方",
        "system": (
            "你扮演反方立场的角色。即使你认为某个答案可能正确，"
            "你也会尝试找到支持相反结论的论据。"
            "你的目标是测试推理的鲁棒性，但最终还是要给出你认为最可能正确的答案。"
        ),
        "temperature": 0.7,
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
AEIC_SYSTEM_BASE = """你是一个多智能体共识系统中的推理节点。对于给定的是非判断题，请按AEIC四层结构进行严格推理：

A (Assumption): 明确问题的前提假设和隐含条件（1-2句话）
E (Evidence): 列出2-4条关键事实证据（必须是具体的，不能笼统）
I (Inference): 基于证据的逻辑推理过程（2-3句话，逻辑链清晰）
C (Conclusion): 必须以"Yes"或"No"开头，然后给出简要理由

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
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        for attempt in range(3):
            try:
                resp = self.session.post(url, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                self._call_count += 1
                msg = data["choices"][0]["message"]
                content = msg.get("content") or msg.get("reasoning_content") or ""
                return content
            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError) as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except requests.exceptions.HTTPError as e:
                if resp.status_code == 429:
                    time.sleep(5)
                    continue
                raise

    def generate_aeic(self, question: str, persona_id: int = 0,
                      evidence_hints: List[str] = None) -> Dict:
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

        raw = self.chat(messages, temperature=persona["temperature"])
        return self._parse_aeic_response(raw, question, persona_id)

    def generate_aeic_with_context(self, question: str, context: str,
                                    persona_id: int = 0) -> Dict:
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

        raw = self.chat(messages, temperature=persona["temperature"])
        return self._parse_aeic_response(raw, question, persona_id)

    def _parse_aeic_response(self, raw: str, question: str,
                             persona_id: int) -> Dict:
        """Parse LLM response into AEIC dict, with fallback."""
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
            # Try to find JSON in the text
            import re
            json_match = re.search(r'\{[^{}]*"A"[^{}]*\}', text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    return {
                        "A": str(result.get("A", "")),
                        "E": list(result.get("E", [])) if isinstance(result.get("E"), list) else [str(result.get("E", ""))],
                        "I": str(result.get("I", "")),
                        "C": str(result.get("C", "")),
                    }
                except json.JSONDecodeError:
                    pass

            # Final fallback
            return {
                "A": f"Analysis of: {question[:80]}",
                "E": [raw[:150]],
                "I": raw[:300],
                "C": raw[-200:] if len(raw) > 200 else raw,
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

    def __init__(self, backends: List[str] = None, seed: int = 42):
        if backends is None:
            backends = [k for k, v in LLM_BACKENDS.items() if v["api_key"]]
        self.backends = backends
        self.clients = {b: LLMClient(backend=b) for b in backends}
        self._rng = random.Random(seed)
        self._call_count = 0

    def get_client(self, agent_id: int = None) -> LLMClient:
        """为指定agent分配LLM后端（确定性映射）。"""
        if agent_id is not None:
            idx = agent_id % len(self.backends)
        else:
            idx = self._rng.randint(0, len(self.backends) - 1)
        return self.clients[self.backends[idx]]

    def generate_aeic(self, question: str, persona_id: int = 0,
                      agent_id: int = 0) -> Dict:
        """使用分配给该agent的LLM生成AEIC。"""
        client = self.get_client(agent_id)
        result = client.generate_aeic(question, persona_id=persona_id)
        self._call_count += 1
        return result

    def generate_aeic_with_context(self, question: str, context: str,
                                    persona_id: int = 0,
                                    agent_id: int = 0) -> Dict:
        """使用分配给该agent的LLM生成带context的AEIC。"""
        client = self.get_client(agent_id)
        result = client.generate_aeic_with_context(question, context, persona_id)
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
