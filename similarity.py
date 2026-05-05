"""
共享的文本相似度计算模块
=====================================================================

提供统一的 SimilarityCalculator 类，支持 5 种相似度计算方法：
  1. char_jaccard    - 字符级 Jaccard 相似度
  2. word_tfidf      - 词级 TF-IDF 相似度
  3. bm25            - BM25 相似度（推荐）
  4. sentence_bert   - Sentence BERT 嵌入
  5. llm_judge       - LLM 语义判断（最高精度）

被 mas/consensus 和 experiments/baselines 共同使用
"""

import math
import warnings
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

# ── 可选依赖 ──────────────────────────────────────────
try:
    import jieba
    jieba.setLogLevel(60)
    _JIEBA_OK = True
except ImportError:
    _JIEBA_OK = False
    warnings.warn("jieba 未安装，word_tfidf/bm25 将退回字符级分词")

try:
    from rank_bm25 import BM25Okapi
    _BM25_OK = True
except ImportError:
    _BM25_OK = False
    warnings.warn("rank_bm25 未安装，bm25 将 fallback 到 word_tfidf")


# ─────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────

_STOPWORDS = {
    "的", "了", "在", "是", "都", "很", "和", "与", "或", "不", "为",
    "有", "也", "而", "到", "以", "就", "但", "从", "被", "把",
}


def _tokenize(text: str) -> List[str]:
    """分词：优先使用 jieba，否则退回字符级"""
    text = str(text).strip()
    if not text:
        return []
    if _JIEBA_OK:
        return [t for t in jieba.cut(text) if t.strip() and t not in _STOPWORDS]
    return [c for c in text if c.strip() and c not in _STOPWORDS]


def _cosine(v1: np.ndarray, v2: np.ndarray) -> float:
    """计算两向量的余弦相似度"""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.clip(np.dot(v1, v2) / (n1 * n2), 0.0, 1.0))


# ─────────────────────────────────────────────────────
# 相似度计算器
# ─────────────────────────────────────────────────────

class SimilarityCalculator:
    """文本相似度计算，5 种方法，签名统一为 (text1, text2) -> float"""

    @staticmethod
    def char_jaccard(t1: str, t2: str) -> float:
        """字符级 Jaccard 相似度（最快）"""
        s1, s2 = set(str(t1)), set(str(t2))
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)

    @staticmethod
    def word_tfidf(t1: str, t2: str, corpus: Optional[List[str]] = None) -> float:
        """词级 TF-IDF 相似度"""
        toks1, toks2 = _tokenize(t1), _tokenize(t2)
        if not toks1 or not toks2:
            return 0.0
        vocab = sorted(set(toks1) | set(toks2))
        if corpus:
            N = len(corpus)
            df = defaultdict(int)
            for doc in corpus:
                for w in set(_tokenize(doc)):
                    df[w] += 1
            idf = {w: math.log((N + 1) / (df.get(w, 0) + 1)) + 1.0 for w in vocab}
        else:
            idf = {w: 1.0 for w in vocab}
        tf1, tf2 = Counter(toks1), Counter(toks2)
        l1, l2 = max(len(toks1), 1), max(len(toks2), 1)
        v1 = np.array([tf1.get(w, 0) / l1 * idf[w] for w in vocab])
        v2 = np.array([tf2.get(w, 0) / l2 * idf[w] for w in vocab])
        return _cosine(v1, v2)

    @staticmethod
    def bm25_similarity(t1: str, t2: str, k1: float = 1.5, b: float = 0.75) -> float:
        """BM25 相似度（推荐方法）"""
        if not _BM25_OK:
            return SimilarityCalculator.word_tfidf(t1, t2)
        toks1, toks2 = _tokenize(t1), _tokenize(t2)
        if not toks1 or not toks2:
            return 0.0
        bm25 = BM25Okapi([toks2], k1=k1, b=b)
        raw = float(bm25.get_scores(toks1)[0])
        upper = len(toks1) * (k1 + 1)
        return float(np.clip(raw / upper, 0.0, 1.0)) if upper > 0 else 0.0

    @staticmethod
    def sentence_bert(t1: str, t2: str, engine: Optional[Any] = None) -> float:
        """Sentence BERT 嵌入相似度"""
        if engine is None:
            return SimilarityCalculator.bm25_similarity(t1, t2)
        try:
            s = engine.bert_similarity(str(t1), str(t2))
            return s if s > 0.0 else SimilarityCalculator.bm25_similarity(t1, t2)
        except Exception:
            return SimilarityCalculator.bm25_similarity(t1, t2)

    @staticmethod
    def llm_judge(
        t1: str,
        t2: str,
        api_key: str = None,
        api_url: str = None,
        model: str = None,
        timeout: int = 15,
    ) -> float:
        """LLM 语义判断相似度（最高精度）"""
        if not api_key:
            return SimilarityCalculator.bm25_similarity(t1, t2)

        prompt = (
            "你是一位专业的语义相似度评审专家。\n"
            "请评估以下两段文本在**语义含义**上的相似程度，\n"
            "输出一个 0.00（完全不同）到 1.00（完全相同）之间的浮点数，\n"
            "**仅输出数字，不要任何解释**。\n\n"
            f"文本1：{str(t1)[:300]}\n"
            f"文本2：{str(t2)[:300]}\n\n相似度分数："
        )
        try:
            from openai import OpenAI

            base = api_url.split("/chat/completions")[0].split("/v1")[0] + "/v1" if api_url else None
            client = OpenAI(api_key=api_key, base_url=base)
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                max_tokens=10, temperature=0.0
            )
            return float(np.clip(float(resp.choices[0].message.content.strip()), 0.0, 1.0))
        except ImportError:
            pass
        except Exception as e:
            warnings.warn(f"LLM-Judge openai SDK 失败 ({e})，尝试 requests")
        try:
            import requests as _req

            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
                "temperature": 0.0,
            }
            resp = _req.post(api_url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            return float(
                np.clip(float(resp.json()["choices"][0]["message"]["content"].strip()), 0.0, 1.0)
            )
        except Exception as e:
            warnings.warn(f"LLM-Judge 全部失败 ({e})，fallback BM25")
            return SimilarityCalculator.bm25_similarity(t1, t2)
