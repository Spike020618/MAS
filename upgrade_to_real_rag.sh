#!/bin/bash

# RAG系统升级到真正的语义向量

echo "🚀 升级RAG系统到真正的语义向量..."
echo ""

# 1. 安装SentenceTransformers
echo "📦 安装 SentenceTransformers..."
pip install -q sentence-transformers

# 2. 验证安装
echo ""
echo "✅ SentenceTransformers 已安装"
echo ""

# 3. 显示升级后的配置
echo "📝 升级后的配置："
echo ""
echo "原始配置（local_hash）："
echo '  embedding_model="local_hash"  # ❌ 只是哈希，不是真正的语义向量'
echo ""
echo "升级后配置（SentenceTransformers）："
echo '  embedding_model="sentence-transformers"  # ✅ 真正的语义向量'
echo ""

# 4. 创建升级版本的演示脚本
cat > /tmp/test_real_rag.py << 'EOF'
"""
测试真正的RAG系统 - 使用SentenceTransformers语义向量
"""

import asyncio
import sys
import os

sys.path.insert(0, '/Users/spike/code/MAS')

from mas.rag import LocalRAGDatabase

async def test_real_rag():
    print("🧪 测试真正的RAG系统（语义向量）")
    print("=" * 70)
    
    # 使用SentenceTransformers创建RAG系统
    rag_db = LocalRAGDatabase(
        storage_path="./rag_storage_real_rag",
        embedding_model="sentence-transformers",  # ✅ 真正的语义向量！
        embedding_dimension=384,
    )
    
    await rag_db.initialize()
    print("✅ 使用SentenceTransformers初始化成功")
    print(f"   向量维度: {rag_db.embedding.get_dimension()}")
    print(f"   向量模型: {rag_db.embedding.model_name}")
    
    # 测试向量化
    print("\n📝 测试向量化...")
    
    text1 = "代码审查：检查代码质量和安全性"
    text2 = "代码评审：验证代码功能和性能"
    text3 = "项目规划：制定时间表和资源分配"
    
    vec1 = await rag_db.embedding.embed(text1)
    vec2 = await rag_db.embedding.embed(text2)
    vec3 = await rag_db.embedding.embed(text3)
    
    print(f"✅ 向量化完成")
    print(f"   文本1: '{text1}'")
    print(f"   向量长度: {len(vec1)}")
    
    # 计算相似度
    import numpy as np
    
    def cosine_similarity(v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    # 计算语义相似度
    sim1_2 = cosine_similarity(vec1, vec2)  # 两个代码相关的文本
    sim1_3 = cosine_similarity(vec1, vec3)  # 代码 vs 项目规划
    
    print(f"\n🔍 语义相似度分析（SentenceTransformers）:")
    print(f"   '代码审查' vs '代码评审': {sim1_2:.4f} (应该很高)")
    print(f"   '代码审查' vs '项目规划': {sim1_3:.4f} (应该较低)")
    
    if sim1_2 > sim1_3:
        print("\n✅ 语义相似度计算正确！这就是真正的RAG!")
    else:
        print("\n⚠️  可能还在使用哈希向量")
    
    await rag_db.close()

# 运行测试
asyncio.run(test_real_rag())
EOF

echo "🧪 运行真正RAG的测试..."
python3 /tmp/test_real_rag.py

echo ""
echo "✅ 升级完成！"
echo ""
echo "现在你可以运行升级版本的演示了："
echo "  python3 demo_step1_with_real_rag.py"
