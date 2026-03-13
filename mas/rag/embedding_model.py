"""
向量化模块 - 将文本转换为向量表示

支持多个模型：
- sentence-transformers (推荐)
- OpenAI Embedding API
- 本地简单哈希方案 (fallback)
"""

import logging
import hashlib
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """文本向量化模型"""

    def __init__(self, model_name: str = "local_hash", api_key: Optional[str] = None):
        """
        初始化向量化模型

        Args:
            model_name: 模型名称
                - "local_hash": 本地哈希（快速，无依赖）
                - "sentence-transformers": 句向量模型（精准）
                - "openai": OpenAI API（最精准）
            api_key: OpenAI API密钥（仅用于openai模型）
        """
        self.model_name = model_name
        self.dimension = 1536  # 标准向量维度
        self.model = None

        if model_name == "local_hash":
            logger.info("✓ Using local hash embedding (fast, no dependencies)")
        elif model_name == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                self.dimension = self.model.get_sentence_embedding_dimension()
                logger.info(f"✓ Loaded SentenceTransformer (dimension: {self.dimension})")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed, fallback to local_hash"
                )
                self.model_name = "local_hash"
        elif model_name == "openai":
            if not api_key:
                raise ValueError("api_key is required for OpenAI model")
            self.api_key = api_key
            logger.info("✓ Using OpenAI Embedding API")
        else:
            raise ValueError(f"Unknown model: {model_name}")

    async def embed(self, text: str) -> List[float]:
        """
        将文本向量化

        Args:
            text: 输入文本

        Returns:
            向量 (float列表)
        """
        if not text or not isinstance(text, str):
            return [0.0] * self.dimension

        if self.model_name == "local_hash":
            return self._embed_hash(text)
        elif self.model_name == "sentence-transformers":
            return self._embed_sentence_transformers(text)
        elif self.model_name == "openai":
            return await self._embed_openai(text)
        else:
            return self._embed_hash(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量向量化

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        if not texts:
            return []

        if self.model_name == "sentence-transformers" and self.model:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return [emb.tolist() for emb in embeddings]
        else:
            # 逐个处理
            return [await self.embed(text) for text in texts]

    def _embed_hash(self, text: str) -> List[float]:
        """基于哈希的快速向量化"""
        # 使用多个哈希算法生成向量
        text_bytes = text.encode("utf-8")
        embeddings = []

        for i in range(self.dimension):
            seed = f"{text_bytes}{i}".encode()
            hash_value = int(hashlib.sha256(seed).hexdigest(), 16)
            # 归一化到 [-1, 1]
            normalized = (hash_value % 10000) / 10000.0 * 2 - 1
            embeddings.append(normalized)

        return embeddings

    def _embed_sentence_transformers(self, text: str) -> List[float]:
        """使用SentenceTransformer向量化"""
        if not self.model:
            return self._embed_hash(text)

        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"SentenceTransformer embedding failed: {e}")
            return self._embed_hash(text)

    async def _embed_openai(self, text: str) -> List[float]:
        """使用OpenAI API向量化"""
        try:
            import openai

            openai.api_key = self.api_key

            response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
            return response["data"][0]["embedding"]

        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            return self._embed_hash(text)

    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.dimension
