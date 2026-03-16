"""
配置管理模块 - RAG系统的配置文件

支持多个数据源：
- Milvus 向量数据库
- 阿里云 DashScope Embedding API
- 本地 FAISS (fallback)
"""

import os
from typing import Optional, Dict, Any


class RAGConfig:
    """RAG系统配置"""

    # ════════════════════════════════════════════════════════════════
    # Milvus 向量数据库配置
    # ════════════════════════════════════════════════════════════════

    # Milvus 连接地址
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = int(os.getenv("MILVUS_PORT", 19530))

    # Milvus 集合配置
    MILVUS_DB_NAME = "rag_system"
    MILVUS_COLLECTION_NAME = "tasks"
    MILVUS_COLLECTION_DIM = 1024  # 嵌入维度 (DashScope text-embedding-v4 返回 1024 维向量)

    # ════════════════════════════════════════════════════════════════
    # 阿里云 DashScope Embedding API 配置
    # ════════════════════════════════════════════════════════════════

    # API 密钥和地址
    DASHSCOPE_API_KEY = os.getenv(
        "DASHSCOPE_API_KEY", "sk-f771855105fe43b28584a0f4d68fb5e9"
    )
    DASHSCOPE_BASE_URL = os.getenv(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    DASHSCOPE_MODEL = os.getenv("DASHSCOPE_MODEL", "text-embedding-v4")

    # ════════════════════════════════════════════════════════════════
    # RAG 系统配置
    # ════════════════════════════════════════════════════════════════

    # 向量数据库类型: "milvus" 或 "faiss"
    VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "milvus")

    # 向量模型类型: "dashscope" 或 "sentence-transformers"
    EMBEDDING_MODEL_TYPE = os.getenv("EMBEDDING_MODEL_TYPE", "dashscope")

    # 存储路径 (用于 FAISS fallback)
    STORAGE_PATH = os.getenv("STORAGE_PATH", "./rag_storage")

    # ════════════════════════════════════════════════════════════════
    # 日志配置
    # ════════════════════════════════════════════════════════════════

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def get_milvus_connection_args(cls) -> Dict[str, Any]:
        """获取 Milvus 连接参数"""
        return {
            "host": cls.MILVUS_HOST,
            "port": cls.MILVUS_PORT,
        }

    @classmethod
    def get_dashscope_config(cls) -> Dict[str, str]:
        """获取 DashScope 配置"""
        return {
            "api_key": cls.DASHSCOPE_API_KEY,
            "base_url": cls.DASHSCOPE_BASE_URL,
            "model": cls.DASHSCOPE_MODEL,
        }

    @classmethod
    def validate(cls) -> bool:
        """验证配置"""
        errors = []

        # 检查 Milvus 连接
        if cls.VECTOR_DB_TYPE == "milvus":
            if not cls.MILVUS_HOST:
                errors.append("MILVUS_HOST 未配置")
            if cls.MILVUS_PORT <= 0:
                errors.append("MILVUS_PORT 无效")

        # 检查 DashScope 配置
        if cls.EMBEDDING_MODEL_TYPE == "dashscope":
            if not cls.DASHSCOPE_API_KEY:
                errors.append("DASHSCOPE_API_KEY 未配置")
            if not cls.DASHSCOPE_BASE_URL:
                errors.append("DASHSCOPE_BASE_URL 未配置")
            if not cls.DASHSCOPE_MODEL:
                errors.append("DASHSCOPE_MODEL 未配置")

        if errors:
            print("❌ 配置错误:")
            for error in errors:
                print(f"  - {error}")
            return False

        print("✅ 配置验证通过")
        return True
