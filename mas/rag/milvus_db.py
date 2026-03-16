"""
Milvus 向量数据库模块

使用 Milvus 作为向量存储后端
支持高性能向量搜索和管理
"""

import logging
from typing import Dict, List, Optional, Any
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)

logger = logging.getLogger(__name__)


class MilvusDatabase:
    """Milvus 向量数据库"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        db_name: str = "rag_system",
        collection_name: str = "tasks",
        dimension: int = 1024,
    ):
        """
        初始化 Milvus 数据库

        Args:
            host: Milvus 服务器地址
            port: Milvus 服务器端口
            db_name: 数据库名称
            collection_name: 集合名称
            dimension: 向量维度
        """
        self.host = host
        self.port = port
        self.db_name = db_name
        self.collection_name = collection_name
        self.dimension = dimension
        self.collection = None
        self.connected = False

    async def connect(self):
        """连接到 Milvus"""
        try:
            # 先不指定 db_name，连接到默认数据库
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
            )

            logger.info(f"✓ Connected to Milvus at {self.host}:{self.port}")
            self.connected = True

            # 创建或使用数据库
            try:
                if self.db_name != "default":
                    if not utility.has_database(self.db_name):
                        logger.info(f"Creating database: {self.db_name}")
                        utility.create_database(db_name=self.db_name)
                    # 使用指定数据库
                    connections.get_connection("default").use_database(self.db_name)
                    logger.info(f"✓ Using database: {self.db_name}")
            except Exception as e:
                logger.warning(f"Database operation failed, using default: {e}")

            # 获取或创建集合
            await self._get_or_create_collection()

        except Exception as e:
            logger.error(f"✗ Failed to connect to Milvus: {e}")
            raise

    async def _get_or_create_collection(self):
        """获取或创建集合"""
        try:
            # 检查集合是否存在
            if utility.has_collection(self.collection_name):
                self.collection = Collection(name=self.collection_name)
                logger.info(f"✓ Loaded existing collection: {self.collection_name}")
            else:
                # 创建新集合
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                    FieldSchema(name="task_id", dtype=DataType.VARCHAR, max_length=256),
                    FieldSchema(name="task_type", dtype=DataType.VARCHAR, max_length=256),
                    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=2048),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
                    FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2048),
                ]

                schema = CollectionSchema(
                    fields=fields,
                    description="RAG System Task Collection",
                    enable_dynamic_field=False,
                )

                self.collection = Collection(
                    name=self.collection_name,
                    schema=schema,
                )

                logger.info(f"✓ Created new collection: {self.collection_name}")

                # 创建索引以加快搜索
                index_params = {
                    "metric_type": "L2",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024},
                }

                self.collection.create_index(
                    field_name="embedding",
                    index_params=index_params,
                )

                logger.info("✓ Created index on embedding field")

        except Exception as e:
            logger.error(f"✗ Failed to create/load collection: {e}")
            raise

    async def insert(
        self,
        task_id: str,
        task_type: str,
        description: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        插入任务到 Milvus

        Args:
            task_id: 任务ID
            task_type: 任务类型
            description: 任务描述
            embedding: 向量
            metadata: 元数据

        Returns:
            插入的ID
        """
        try:
            import json

            # 生成唯一ID
            import time

            doc_id = int(time.time() * 1000000) % 10000000000

            data = [
                [doc_id],  # id
                [task_id],  # task_id
                [task_type],  # task_type
                [description],  # description
                [embedding],  # embedding
                [json.dumps(metadata or {})],  # metadata
            ]

            result = self.collection.insert(data)

            # 刷新以确保数据可搜索
            self.collection.flush()

            logger.info(f"✓ Inserted task {task_id} to Milvus")
            return doc_id

        except Exception as e:
            logger.error(f"✗ Failed to insert task: {e}")
            raise

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        task_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        搜索相似任务

        Args:
            query_embedding: 查询向量
            top_k: 返回最相似的K个结果
            task_type: 任务类型过滤 (可选)

        Returns:
            相似任务列表
        """
        try:
            # 确保集合已加载
            if not self.collection.is_empty:
                self.collection.load()
            
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["task_id", "task_type", "description", "metadata"],
            )

            output = []
            for hit in results[0]:
                doc = {
                    "task_id": hit.entity.get("task_id"),
                    "task_type": hit.entity.get("task_type"),
                    "description": hit.entity.get("description"),
                    "distance": hit.distance,
                    "similarity": 1 / (1 + hit.distance),  # 将距离转换为相似度
                }

                # 如果指定了task_type，则过滤
                if task_type is None or doc["task_type"] == task_type:
                    output.append(doc)

            logger.info(f"✓ Found {len(output)} similar tasks")
            return output

        except Exception as e:
            logger.error(f"✗ Failed to search: {e}")
            return []

    async def get_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            if not self.collection:
                return {}

            num_entities = self.collection.num_entities

            return {
                "collection_name": self.collection_name,
                "num_entities": num_entities,
                "dimension": self.dimension,
                "status": "healthy" if self.connected else "disconnected",
            }

        except Exception as e:
            logger.error(f"✗ Failed to get stats: {e}")
            return {}

    async def close(self):
        """关闭连接"""
        try:
            if self.connected:
                connections.disconnect(alias="default")
                logger.info("✓ Disconnected from Milvus")
                self.connected = False
        except Exception as e:
            logger.error(f"✗ Failed to close connection: {e}")
