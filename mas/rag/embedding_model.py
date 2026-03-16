"""
向量化模块 - 使用阿里云 DashScope Embedding API

支持多个模型：
- DashScope (阿里云，推荐用于生产)
- SentenceTransformers (本地，无需API)
- 本地简单哈希方案 (fallback)
"""

import logging
import hashlib
import asyncio
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """文本向量化模型"""

    def __init__(
        self,
        model_name: str = "dashscope",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        初始化向量化模型

        Args:
            model_name: 模型名称
                - "dashscope": 阿里云 DashScope (推荐)
                - "sentence-transformers": 本地句向量模型
                - "local_hash": 本地哈希 (fallback)
            api_key: API 密钥 (用于 DashScope)
            base_url: API 基础URL (用于 DashScope)
            model: 模型名称 (用于 DashScope)
        """
        self.model_name = model_name
        # DashScope text-embedding-v4 返回 1024 维向量
        self.dimension = 1024
        self.model = None
        self.api_key = api_key
        self.base_url = base_url
        self.api_model = model

        if model_name == "dashscope":
            if not api_key or not base_url:
                logger.warning(
                    "DashScope 配置不完整，fallback 到 SentenceTransformers"
                )
                self.model_name = "sentence-transformers"
                self._init_sentence_transformers()
            else:
                logger.info("✓ Using DashScope Embedding API (Production)")
                # 确保维度是 1024（DashScope text-embedding-v4）
                self.dimension = 1024

        elif model_name == "sentence-transformers":
            self._init_sentence_transformers()

        elif model_name == "local_hash":
            logger.info("✓ Using local hash embedding (fast, no dependencies)")
            self.dimension = 1024

        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _init_sentence_transformers(self):
        """初始化 SentenceTransformers"""
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(
                f"✓ Loaded SentenceTransformer (dimension: {self.dimension})"
            )
        except ImportError:
            logger.warning(
                "sentence-transformers not installed, fallback to local_hash"
            )
            self.model_name = "local_hash"
            self.dimension = 1024

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

        if self.model_name == "dashscope":
            return await self._embed_dashscope(text)
        elif self.model_name == "sentence-transformers":
            return self._embed_sentence_transformers(text)
        elif self.model_name == "local_hash":
            return self._embed_hash(text)
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

        if self.model_name == "dashscope":
            return await self._embed_batch_dashscope(texts)
        elif self.model_name == "sentence-transformers" and self.model:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return [emb.tolist() for emb in embeddings]
        else:
            # 逐个处理
            return [await self.embed(text) for text in texts]

    async def _embed_dashscope(self, text: str) -> List[float]:
        """使用 DashScope API 向量化"""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"input": text, "model": self.api_model},
                    timeout=30.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    embedding = data["data"][0]["embedding"]
                    # 验证维度
                    if len(embedding) != self.dimension:
                        logger.warning(f"Expected dimension {self.dimension}, got {len(embedding)}")
                    return embedding
                else:
                    logger.error(
                        f"DashScope API 错误: {response.status_code} {response.text}"
                    )
                    return self._embed_hash(text)

        except Exception as e:
            logger.error(f"DashScope embedding failed: {e}")
            return self._embed_hash(text)

    async def _embed_batch_dashscope(self, texts: List[str]) -> List[List[float]]:
        """使用 DashScope API 批量向量化"""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"input": texts, "model": self.api_model},
                    timeout=30.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    embeddings = [item["embedding"] for item in data["data"]]
                    return embeddings
                else:
                    logger.error(
                        f"DashScope API 错误: {response.status_code} {response.text}"
                    )
                    return [self._embed_hash(text) for text in texts]

        except Exception as e:
            logger.error(f"DashScope batch embedding failed: {e}")
            return [self._embed_hash(text) for text in texts]

    def _embed_hash(self, text: str) -> List[float]:
        """基于哈希的快速向量化 (fallback)"""
        text_bytes = text.encode("utf-8")
        embeddings = []

        for i in range(self.dimension):
            seed = f"{text_bytes}{i}".encode()
            hash_value = int(hashlib.sha256(seed).hexdigest(), 16)
            normalized = (hash_value % 10000) / 10000.0 * 2 - 1
            embeddings.append(normalized)

        return embeddings

    def _embed_sentence_transformers(self, text: str) -> List[float]:
        """使用 SentenceTransformer 向量化"""
        if not self.model:
            return self._embed_hash(text)

        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"SentenceTransformer embedding failed: {e}")
            return self._embed_hash(text)

    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.dimension
