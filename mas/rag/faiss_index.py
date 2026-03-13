"""
FAISS 向量索引包装器

管理本地向量索引的创建、保存、加载和搜索
"""

import logging
import os
import pickle
from typing import List, Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class FAISSIndex:
    """FAISS向量索引包装器"""

    def __init__(self, index_name: str, dimension: int = 1536):
        """
        初始化FAISS索引

        Args:
            index_name: 索引名称 (task, solution, record)
            dimension: 向量维度
        """
        self.index_name = index_name
        self.dimension = dimension
        self.index = None
        self.id_mapping = {}  # ID到向量的映射
        self.vector_mapping = {}  # 向量到完整对象的映射

        self._init_index()

    def _init_index(self):
        """初始化FAISS索引"""
        try:
            import faiss

            # 创建简单的平面索引（L2距离）
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info(f"✓ Initialized FAISS index for {self.index_name}")

        except ImportError:
            logger.warning(
                "faiss-cpu not installed, using fallback similarity search"
            )
            self.index = None

    async def add_vector(
        self, vector_id: str, vector: List[float], metadata: Dict = None
    ) -> None:
        """
        添加向量到索引

        Args:
            vector_id: 向量ID
            vector: 向量数据
            metadata: 关联的元数据
        """
        try:
            if len(vector) != self.dimension:
                raise ValueError(
                    f"Vector dimension {len(vector)} doesn't match expected {self.dimension}"
                )

            vector_np = np.array([vector], dtype=np.float32)

            if self.index:
                # 使用FAISS
                self.index.add(vector_np)
                self.id_mapping[len(self.id_mapping)] = vector_id
            else:
                # Fallback: 内存存储
                self.id_mapping[vector_id] = vector

            self.vector_mapping[vector_id] = metadata or {}
            logger.debug(f"Added vector: {vector_id}")

        except Exception as e:
            logger.error(f"Failed to add vector {vector_id}: {e}")
            raise

    async def search(
        self, query_vector: List[float], top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        搜索相似向量

        Args:
            query_vector: 查询向量
            top_k: 返回最相似的K个

        Returns:
            [(vector_id, distance), ...] 列表，按距离排序
        """
        try:
            if len(query_vector) != self.dimension:
                raise ValueError(
                    f"Query dimension {len(query_vector)} doesn't match {self.dimension}"
                )

            query_np = np.array([query_vector], dtype=np.float32)

            if self.index:
                # 使用FAISS搜索
                distances, indices = self.index.search(query_np, min(top_k, self.index.ntotal))

                results = []
                for idx, distance in zip(indices[0], distances[0]):
                    if idx in self.id_mapping:
                        vector_id = self.id_mapping[idx]
                        results.append((vector_id, float(distance)))

                return results

            else:
                # Fallback: 暴力搜索
                return await self._fallback_search(query_vector, top_k)

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def _fallback_search(
        self, query_vector: List[float], top_k: int
    ) -> List[Tuple[str, float]]:
        """Fallback搜索（当FAISS不可用时）"""
        query_np = np.array(query_vector, dtype=np.float32)
        results = []

        for vector_id, stored_vector in self.id_mapping.items():
            if isinstance(stored_vector, list):
                stored_np = np.array(stored_vector, dtype=np.float32)
                # 计算L2距离
                distance = float(np.linalg.norm(query_np - stored_np))
                results.append((vector_id, distance))

        # 按距离排序
        results.sort(key=lambda x: x[1])
        return results[:top_k]

    async def get_metadata(self, vector_id: str) -> Optional[Dict]:
        """获取向量关联的元数据"""
        return self.vector_mapping.get(vector_id)

    async def save(self, save_path: str) -> None:
        """
        保存索引到磁盘

        Args:
            save_path: 保存路径
        """
        try:
            os.makedirs(save_path, exist_ok=True)

            if self.index:
                import faiss

                faiss.write_index(
                    self.index, os.path.join(save_path, f"{self.index_name}.faiss")
                )

            # 保存映射和元数据
            with open(
                os.path.join(save_path, f"{self.index_name}_mapping.pkl"), "wb"
            ) as f:
                pickle.dump(self.id_mapping, f)

            with open(
                os.path.join(save_path, f"{self.index_name}_metadata.pkl"), "wb"
            ) as f:
                pickle.dump(self.vector_mapping, f)

            logger.info(f"✓ Saved FAISS index: {self.index_name}")

        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise

    async def load(self, load_path: str) -> None:
        """
        从磁盘加载索引

        Args:
            load_path: 加载路径
        """
        try:
            if self.index:
                import faiss

                index_file = os.path.join(load_path, f"{self.index_name}.faiss")
                if os.path.exists(index_file):
                    self.index = faiss.read_index(index_file)

            # 加载映射和元数据
            mapping_file = os.path.join(load_path, f"{self.index_name}_mapping.pkl")
            if os.path.exists(mapping_file):
                with open(mapping_file, "rb") as f:
                    self.id_mapping = pickle.load(f)

            metadata_file = os.path.join(load_path, f"{self.index_name}_metadata.pkl")
            if os.path.exists(metadata_file):
                with open(metadata_file, "rb") as f:
                    self.vector_mapping = pickle.load(f)

            logger.info(f"✓ Loaded FAISS index: {self.index_name}")

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise

    async def get_stats(self) -> Dict:
        """获取索引统计信息"""
        total = self.index.ntotal if self.index else len(self.id_mapping)
        return {
            "index_name": self.index_name,
            "total_vectors": total,
            "dimension": self.dimension,
        }
