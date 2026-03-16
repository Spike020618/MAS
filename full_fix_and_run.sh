#!/bin/bash

# 完整的修复和重启脚本

set -e

cd /Users/spike/code/MAS

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║            🔧 RAG 系统完整修复 - 解决维度不匹配问题                       ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 第1步：停止并删除所有容器和卷
echo "第1步：清除旧的 Docker 容器和数据..."
docker-compose -f docker-compose-milvus.yml down -v 2>/dev/null || true
sleep 2

# 第2步：清除系统资源
echo "第2步：清除未使用的 Docker 资源..."
docker system prune -f --volumes 2>/dev/null || true
sleep 1

# 第3步：重新启动容器
echo "第3步：启动 Milvus 容器（这需要 1-2 分钟）..."
docker-compose -f docker-compose-milvus.yml up -d
echo "等待容器完全启动..."
sleep 30

# 第4步：验证 Milvus 就绪
echo "第4步：验证 Milvus 连接..."
for i in {1..20}; do
    if curl -s http://localhost:9091/healthz > /dev/null 2>&1; then
        echo "✅ Milvus 已就绪"
        break
    fi
    if [ $i -eq 20 ]; then
        echo "❌ Milvus 启动失败"
        exit 1
    fi
    echo "  等待中... ($i/20)"
    sleep 3
done

# 第5步：验证 Python 连接
echo ""
echo "第5步：验证 Python 连接..."
python3 << 'EOF'
import sys
sys.path.insert(0, '/Users/spike/code/MAS')

try:
    from pymilvus import connections
    connections.connect(host="localhost", port=19530)
    print("✅ Python 连接到 Milvus 成功")
    connections.disconnect(alias="default")
except Exception as e:
    print(f"❌ 连接失败: {e}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║               ✅ 系统准备就绪！现在运行演示脚本                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 第6步：运行演示
echo "第6步：运行 RAG 演示脚本..."
echo ""

export DASHSCOPE_API_KEY="sk-f771855105fe43b28584a0f4d68fb5e9"
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_MODEL="text-embedding-v4"

python3 mas/rag/demo_rag_milvus.py

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                   ✅ 演示完成！                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "下一步：打开 Attu 可视化界面"
echo "  http://localhost:8000"
