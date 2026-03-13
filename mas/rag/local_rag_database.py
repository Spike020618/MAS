"""
本地 RAG 数据库 - 基于FAISS和JSON存储

提供统一的接口用于：
- 任务模板存储和检索
- Agent方案存储和检索
- 成功记录存储
- 权重管理
"""

import logging
import json
import os
import asyncio
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import time

from .embedding_model import EmbeddingModel
from .faiss_index import FAISSIndex

logger = logging.getLogger(__name__)


class LocalRAGDatabase:
    """本地RAG数据库 - FAISS索引 + JSON元数据存储"""

    def __init__(
        self,
        storage_path: str = "./rag_storage",
        embedding_model: str = "local_hash",
        embedding_dimension: int = 1536,
    ):
        """
        初始化本地RAG数据库

        Args:
            storage_path: 存储路径（存放JSON文件和FAISS索引）
            embedding_model: 向量化模型 ("local_hash", "sentence-transformers", "openai")
            embedding_dimension: 向量维度
        """
        self.storage_path = storage_path
        self.embedding_dimension = embedding_dimension

        # 创建存储目录结构
        os.makedirs(storage_path, exist_ok=True)
        os.makedirs(os.path.join(storage_path, "metadata"), exist_ok=True)
        os.makedirs(os.path.join(storage_path, "indexes"), exist_ok=True)

        # 初始化向量化模型
        self.embedding = EmbeddingModel(model_name=embedding_model)

        # 初始化FAISS索引
        self.task_index = FAISSIndex("tasks", dimension=embedding_dimension)
        self.solution_index = FAISSIndex("solutions", dimension=embedding_dimension)
        self.record_index = FAISSIndex("records", dimension=embedding_dimension)

        # 内存缓存
        self.tasks_cache = {}  # task_id -> task_data
        self.solutions_cache = {}  # solution_id -> solution_data
        self.records_cache = {}  # record_id -> record_data
        self.agents_registry = {}  # agent_id -> agent_info
        self.weights = {
            "w_A": 0.2,
            "w_E": 0.3,
            "w_I": 0.2,
            "w_C": 0.3,
        }  # AEIC权重

        logger.info(f"✓ LocalRAGDatabase initialized at {storage_path}")

    async def initialize(self) -> None:
        """初始化数据库（加载已保存的索引和元数据）"""
        try:
            logger.info("Initializing RAG database...")

            # 尝试加载已保存的索引
            index_path = os.path.join(self.storage_path, "indexes")
            await self.task_index.load(index_path)
            await self.solution_index.load(index_path)
            await self.record_index.load(index_path)

            # 加载缓存数据
            await self._load_cache()

            # 加载权重
            await self._load_weights()

            logger.info("✓ RAG database initialized")

        except Exception as e:
            logger.warning(f"Failed to load existing data: {e}")
            logger.info("Starting with empty database")

    async def _load_cache(self) -> None:
        """加载缓存的元数据"""
        metadata_path = os.path.join(self.storage_path, "metadata")

        # 加载任务
        tasks_file = os.path.join(metadata_path, "tasks.json")
        if os.path.exists(tasks_file):
            with open(tasks_file, "r", encoding="utf-8") as f:
                self.tasks_cache = json.load(f)

        # 加载方案
        solutions_file = os.path.join(metadata_path, "solutions.json")
        if os.path.exists(solutions_file):
            with open(solutions_file, "r", encoding="utf-8") as f:
                self.solutions_cache = json.load(f)

        # 加载记录
        records_file = os.path.join(metadata_path, "records.json")
        if os.path.exists(records_file):
            with open(records_file, "r", encoding="utf-8") as f:
                self.records_cache = json.load(f)

        # 加载Agent注册表
        agents_file = os.path.join(metadata_path, "agents.json")
        if os.path.exists(agents_file):
            with open(agents_file, "r", encoding="utf-8") as f:
                self.agents_registry = json.load(f)

        logger.info("✓ Loaded cache data")

    async def _save_cache(self) -> None:
        """保存缓存的元数据到磁盘"""
        metadata_path = os.path.join(self.storage_path, "metadata")

        # 保存任务
        with open(os.path.join(metadata_path, "tasks.json"), "w", encoding="utf-8") as f:
            json.dump(self.tasks_cache, f, indent=2, ensure_ascii=False)

        # 保存方案
        with open(
            os.path.join(metadata_path, "solutions.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(self.solutions_cache, f, indent=2, ensure_ascii=False)

        # 保存记录
        with open(os.path.join(metadata_path, "records.json"), "w", encoding="utf-8") as f:
            json.dump(self.records_cache, f, indent=2, ensure_ascii=False)

        # 保存Agent注册表
        with open(
            os.path.join(metadata_path, "agents.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(self.agents_registry, f, indent=2, ensure_ascii=False)

    async def _load_weights(self) -> None:
        """加载权重"""
        metadata_path = os.path.join(self.storage_path, "metadata")
        weights_file = os.path.join(metadata_path, "weights.json")

        if os.path.exists(weights_file):
            with open(weights_file, "r", encoding="utf-8") as f:
                self.weights = json.load(f)
            logger.info(f"Loaded weights: {self.weights}")

    async def _save_weights(self) -> None:
        """保存权重"""
        metadata_path = os.path.join(self.storage_path, "metadata")
        weights_file = os.path.join(metadata_path, "weights.json")

        with open(weights_file, "w", encoding="utf-8") as f:
            json.dump(self.weights, f, indent=2)

    # ────────────────────────────────────────────────────────
    # 任务模板操作
    # ────────────────────────────────────────────────────────

    async def add_task(
        self,
        task_id: str,
        task_type: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        添加任务模板

        Args:
            task_id: 任务ID
            task_type: 任务类型 (review, planning, development)
            description: 任务描述
            metadata: 额外元数据

        Returns:
            task_id
        """
        try:
            # 向量化描述
            embedding = await self.embedding.embed(description)

            # 保存到缓存
            task_data = {
                "task_id": task_id,
                "task_type": task_type,
                "description": description,
                "metadata": metadata or {},
                "created_at": datetime.now().isoformat(),
            }
            self.tasks_cache[task_id] = task_data

            # 添加到FAISS索引
            await self.task_index.add_vector(task_id, embedding, task_data)

            logger.info(f"✓ Added task: {task_id}")
            return task_id

        except Exception as e:
            logger.error(f"✗ Failed to add task: {e}")
            raise

    async def search_tasks(
        self,
        query: str,
        task_type: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        搜索相似的任务

        Args:
            query: 搜索查询
            task_type: 可选的任务类型过滤
            top_k: 返回最相似的K个

        Returns:
            相似任务列表
        """
        try:
            # 向量化查询
            query_embedding = await self.embedding.embed(query)

            # FAISS搜索
            results = await self.task_index.search(query_embedding, top_k=top_k * 2)

            # 过滤和排序
            filtered_results = []
            for task_id, distance in results:
                if task_id in self.tasks_cache:
                    task = self.tasks_cache[task_id]

                    # 类型过滤
                    if task_type and task.get("task_type") != task_type:
                        continue

                    filtered_results.append(
                        {
                            **task,
                            "similarity_distance": distance,
                        }
                    )

            logger.info(f"Found {len(filtered_results)} similar tasks")
            return filtered_results[:top_k]

        except Exception as e:
            logger.error(f"✗ Task search failed: {e}")
            return []

    # ────────────────────────────────────────────────────────
    # Agent方案操作
    # ────────────────────────────────────────────────────────

    async def add_solution(
        self,
        solution_id: str,
        agent_id: int,
        task_type: str,
        solution_text: str,
        success_rate: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        添加Agent方案

        Args:
            solution_id: 方案ID
            agent_id: Agent ID
            task_type: 任务类型
            solution_text: 方案描述
            success_rate: 历史成功率 (0.0-1.0)
            metadata: 额外元数据

        Returns:
            solution_id
        """
        try:
            # 向量化方案文本
            embedding = await self.embedding.embed(solution_text)

            # 保存到缓存
            solution_data = {
                "solution_id": solution_id,
                "agent_id": agent_id,
                "task_type": task_type,
                "solution_text": solution_text,
                "success_rate": success_rate,
                "metadata": metadata or {},
                "created_at": datetime.now().isoformat(),
            }
            self.solutions_cache[solution_id] = solution_data

            # 添加到FAISS索引
            await self.solution_index.add_vector(solution_id, embedding, solution_data)

            logger.info(f"✓ Added solution: {solution_id} (Agent {agent_id})")
            return solution_id

        except Exception as e:
            logger.error(f"✗ Failed to add solution: {e}")
            raise

    async def search_solutions(
        self,
        query: str,
        task_type: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        搜索最佳的Agent方案

        Args:
            query: 查询文本
            task_type: 任务类型
            top_k: 返回最相似的K个

        Returns:
            最佳方案列表
        """
        try:
            # 向量化查询
            query_embedding = await self.embedding.embed(query)

            # FAISS搜索
            results = await self.solution_index.search(query_embedding, top_k=top_k * 2)

            # 过滤和排序
            filtered_results = []
            for solution_id, distance in results:
                if solution_id in self.solutions_cache:
                    solution = self.solutions_cache[solution_id]

                    # 类型过滤
                    if task_type and solution.get("task_type") != task_type:
                        continue

                    filtered_results.append(
                        {
                            **solution,
                            "similarity_distance": distance,
                        }
                    )

            # 按成功率排序
            filtered_results.sort(
                key=lambda x: x.get("success_rate", 0), reverse=True
            )

            logger.info(f"Found {len(filtered_results)} solutions")
            return filtered_results[:top_k]

        except Exception as e:
            logger.error(f"✗ Solution search failed: {e}")
            return []

    # ────────────────────────────────────────────────────────
    # 成功记录操作
    # ────────────────────────────────────────────────────────

    async def record_success(
        self,
        record_id: str,
        task_id: str,
        agent_ids: List[int],
        feedback: str,
        success_score: float = 0.8,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        记录成功的任务分配

        Args:
            record_id: 记录ID
            task_id: 任务ID
            agent_ids: 分配的Agent IDs
            feedback: 反馈文本
            success_score: 成功评分 (0.0-1.0)
            metadata: 额外元数据

        Returns:
            record_id
        """
        try:
            # 向量化反馈
            embedding = await self.embedding.embed(feedback)

            # 保存到缓存
            record_data = {
                "record_id": record_id,
                "task_id": task_id,
                "agent_ids": agent_ids,
                "feedback": feedback,
                "success_score": success_score,
                "metadata": metadata or {},
                "created_at": datetime.now().isoformat(),
            }
            self.records_cache[record_id] = record_data

            # 添加到FAISS索引
            await self.record_index.add_vector(record_id, embedding, record_data)

            logger.info(f"✓ Recorded success: {record_id} (agents={agent_ids})")
            return record_id

        except Exception as e:
            logger.error(f"✗ Failed to record success: {e}")
            raise

    # ────────────────────────────────────────────────────────
    # Agent管理
    # ────────────────────────────────────────────────────────

    async def register_agent(
        self,
        agent_id: int,
        name: str,
        task_types: List[str],
        success_rate: float = 0.5,
    ) -> None:
        """
        注册Agent

        Args:
            agent_id: Agent ID
            name: Agent名称
            task_types: 支持的任务类型列表
            success_rate: 历史成功率
        """
        try:
            agent_info = {
                "agent_id": agent_id,
                "name": name,
                "task_types": task_types,
                "success_rate": success_rate,
                "is_online": True,
                "registered_at": datetime.now().isoformat(),
            }
            self.agents_registry[str(agent_id)] = agent_info
            logger.info(f"✓ Registered agent: {agent_id} ({name})")

        except Exception as e:
            logger.error(f"✗ Failed to register agent: {e}")
            raise

    async def get_agent(self, agent_id: int) -> Optional[Dict[str, Any]]:
        """获取Agent信息"""
        return self.agents_registry.get(str(agent_id))

    async def list_agents(self) -> Dict[int, Dict[str, Any]]:
        """获取所有已注册的Agent"""
        return {
            int(agent_id): info for agent_id, info in self.agents_registry.items()
        }

    # ────────────────────────────────────────────────────────
    # 权重管理
    # ────────────────────────────────────────────────────────

    async def get_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        return self.weights.copy()

    async def set_weights(self, weights: Dict[str, float]) -> None:
        """
        设置权重

        Args:
            weights: 权重字典 {"w_A": 0.2, "w_E": 0.3, "w_I": 0.2, "w_C": 0.3}
        """
        try:
            # 验证权重和
            total = sum(weights.values())
            if abs(total - 1.0) > 0.01:
                logger.warning(
                    f"Weight sum is {total}, normalizing to 1.0"
                )
                total = sum(weights.values())
                weights = {k: v / total for k, v in weights.items()}

            self.weights = weights
            await self._save_weights()
            logger.info(f"✓ Updated weights: {weights}")

        except Exception as e:
            logger.error(f"✗ Failed to set weights: {e}")
            raise

    # ────────────────────────────────────────────────────────
    # 统计和监控
    # ────────────────────────────────────────────────────────

    async def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        try:
            task_stats = await self.task_index.get_stats()
            solution_stats = await self.solution_index.get_stats()
            record_stats = await self.record_index.get_stats()

            stats = {
                "tasks": task_stats["total_vectors"],
                "solutions": solution_stats["total_vectors"],
                "records": record_stats["total_vectors"],
                "agents": len(self.agents_registry),
                "weights": self.weights,
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(f"RAG Stats: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    # ────────────────────────────────────────────────────────
    # 持久化
    # ────────────────────────────────────────────────────────

    async def save(self) -> None:
        """保存所有数据到磁盘"""
        try:
            logger.info("Saving RAG database...")

            # 保存缓存
            await self._save_cache()

            # 保存FAISS索引
            index_path = os.path.join(self.storage_path, "indexes")
            await self.task_index.save(index_path)
            await self.solution_index.save(index_path)
            await self.record_index.save(index_path)

            # 保存权重
            await self._save_weights()

            logger.info("✓ RAG database saved")

        except Exception as e:
            logger.error(f"✗ Failed to save database: {e}")
            raise

    async def close(self) -> None:
        """关闭数据库（保存并清理）"""
        try:
            await self.save()
            logger.info("✓ RAG database closed")
        except Exception as e:
            logger.error(f"✗ Failed to close database: {e}")
