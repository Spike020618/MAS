"""
RAG 数据库 - 支持 Milvus 向量存储 + DashScope 嵌入

提供统一的接口用于：
- 使用 Milvus 存储和搜索任务向量
- 使用 DashScope API 生成语义向量
- Agent 配置管理
- 权重管理
"""

import logging
import json
import os
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime

from .embedding_model import EmbeddingModel
from .milvus_db import MilvusDatabase
from .config import RAGConfig

logger = logging.getLogger(__name__)


class RAGDatabase:
    """RAG 数据库 - Milvus + DashScope 版本（生产级）"""

    def __init__(
        self,
        storage_path: str = "./rag_storage",
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """
        初始化 RAG 数据库

        Args:
            storage_path: 本地存储路径（用于元数据和备份）
            milvus_host: Milvus 服务器地址
            milvus_port: Milvus 服务器端口
            embedding_api_key: DashScope API 密钥
            embedding_base_url: DashScope API 地址
            embedding_model: DashScope 模型名称
        """
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        os.makedirs(os.path.join(storage_path, "metadata"), exist_ok=True)

        # 初始化向量化模型
        logger.info("初始化向量化模型...")
        self.embedding = EmbeddingModel(
            model_name="dashscope",
            api_key=embedding_api_key or RAGConfig.DASHSCOPE_API_KEY,
            base_url=embedding_base_url or RAGConfig.DASHSCOPE_BASE_URL,
            model=embedding_model or RAGConfig.DASHSCOPE_MODEL,
        )

        # 初始化 Milvus 数据库
        logger.info("初始化 Milvus 向量数据库...")
        self.milvus = MilvusDatabase(
            host=milvus_host,
            port=milvus_port,
            db_name=RAGConfig.MILVUS_DB_NAME,
            collection_name=RAGConfig.MILVUS_COLLECTION_NAME,
            dimension=self.embedding.get_dimension(),
        )

        # 内存缓存
        self.agents_registry = {}  # agent_id -> agent_info
        self.weights = {
            "w_A": 0.25,  # Ability
            "w_E": 0.25,  # Evidence
            "w_I": 0.25,  # Inference
            "w_C": 0.25,  # Conclusion
        }

        # 统计信息
        self.stats = {
            "total_tasks": 0,
            "total_agents": 0,
            "embeddings_generated": 0,
        }

        logger.info("✓ RAG 数据库初始化完成")

    async def initialize(self) -> None:
        """初始化数据库连接"""
        try:
            logger.info("连接到 Milvus 向量数据库...")
            await self.milvus.connect()

            # 加载本地缓存数据
            await self._load_cache()

            logger.info("✓ 数据库初始化成功")

        except Exception as e:
            logger.error(f"✗ 数据库初始化失败: {e}")
            raise

    async def register_agent(
        self,
        agent_id: int,
        name: str,
        task_types: List[str],
        success_rate: float,
    ) -> None:
        """
        注册 Agent

        Args:
            agent_id: Agent ID
            name: Agent 名称
            task_types: 支持的任务类型列表
            success_rate: 成功率
        """
        self.agents_registry[agent_id] = {
            "id": agent_id,
            "name": name,
            "task_types": task_types,
            "success_rate": success_rate,
            "created_at": datetime.now().isoformat(),
        }

        self.stats["total_agents"] = len(self.agents_registry)
        logger.info(f"✓ 注册 Agent: {name} (ID: {agent_id})")

    async def add_task(
        self,
        task_id: str,
        task_type: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        添加任务

        Args:
            task_id: 任务ID
            task_type: 任务类型
            description: 任务描述
            metadata: 元数据
        """
        try:
            # 生成向量
            logger.info(f"生成任务向量: {task_id}")
            embedding = await self.embedding.embed(description)

            # 存储到 Milvus
            await self.milvus.insert(
                task_id=task_id,
                task_type=task_type,
                description=description,
                embedding=embedding,
                metadata=metadata or {},
            )

            self.stats["embeddings_generated"] += 1
            self.stats["total_tasks"] += 1

            # 保存到本地元数据
            await self._save_task_metadata(task_id, task_type, description, metadata)

        except Exception as e:
            logger.error(f"✗ 添加任务失败: {e}")
            raise

    async def search_tasks(
        self,
        query_text: str,
        task_type: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        搜索相似任务

        Args:
            query_text: 查询文本
            task_type: 任务类型过滤 (可选)
            top_k: 返回最相似的K个结果

        Returns:
            相似任务列表
        """
        try:
            # 生成查询向量
            query_embedding = await self.embedding.embed(query_text)

            # 在 Milvus 中搜索
            results = await self.milvus.search(
                query_embedding=query_embedding,
                top_k=top_k,
                task_type=task_type,
            )

            logger.info(f"✓ 搜索完成，找到 {len(results)} 个相似任务")
            return results

        except Exception as e:
            logger.error(f"✗ 搜索失败: {e}")
            return []

    async def list_agents(self) -> Dict[int, Dict[str, Any]]:
        """
        获取所有 Agent

        Returns:
            Agent 字典
        """
        return self.agents_registry

    async def get_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        return self.weights

    async def set_weights(self, weights: Dict[str, float]) -> None:
        """设置权重"""
        if set(weights.keys()) != {"w_A", "w_E", "w_I", "w_C"}:
            raise ValueError("权重必须包含 w_A, w_E, w_I, w_C")

        self.weights = weights
        await self._save_weights()
        logger.info(f"✓ 权重已更新: {weights}")

    async def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        milvus_stats = await self.milvus.get_stats()

        return {
            "embedding_model": self.embedding.model_name,
            "embedding_dimension": self.embedding.get_dimension(),
            "total_agents": self.stats["total_agents"],
            "total_tasks": self.stats["total_tasks"],
            "embeddings_generated": self.stats["embeddings_generated"],
            "milvus": milvus_stats,
            "weights": self.weights,
        }

    async def _save_task_metadata(
        self,
        task_id: str,
        task_type: str,
        description: str,
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """保存任务元数据到本地文件"""
        try:
            metadata_file = os.path.join(
                self.storage_path, "metadata", f"{task_id}.json"
            )

            data = {
                "task_id": task_id,
                "task_type": task_type,
                "description": description,
                "metadata": metadata or {},
                "created_at": datetime.now().isoformat(),
            }

            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save task metadata: {e}")

    async def _save_weights(self) -> None:
        """保存权重到文件"""
        try:
            weights_file = os.path.join(self.storage_path, "weights.json")
            with open(weights_file, "w", encoding="utf-8") as f:
                json.dump(self.weights, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save weights: {e}")

    async def _load_cache(self) -> None:
        """加载本地缓存"""
        try:
            # 加载权重
            weights_file = os.path.join(self.storage_path, "weights.json")
            if os.path.exists(weights_file):
                with open(weights_file, "r", encoding="utf-8") as f:
                    self.weights = json.load(f)
                logger.info("✓ 权重已加载")

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    async def save(self) -> None:
        """保存数据库状态"""
        try:
            await self._save_weights()
            logger.info("✓ 数据库已保存")
        except Exception as e:
            logger.error(f"✗ 保存失败: {e}")

    async def close(self) -> None:
        """关闭数据库连接"""
        try:
            await self.save()
            await self.milvus.close()
            logger.info("✓ 数据库已关闭")
        except Exception as e:
            logger.error(f"✗ 关闭失败: {e}")
