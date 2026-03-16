#!/bin/bash

# 终极清理脚本 - 完全清除所有旧数据

set -e

cd /Users/spike/code/MAS

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║              🔥 终极清理 - 完全删除所有旧数据                             ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 第1步：停止所有容器
echo "第1步：停止所有 Docker 容器..."
docker-compose -f docker-compose-milvus.yml stop 2>/dev/null || true
sleep 2

# 第2步：删除所有容器
echo "第2步：删除所有容器..."
docker-compose -f docker-compose-milvus.yml down 2>/dev/null || true
sleep 2

# 第3步：删除所有卷（非常重要！）
echo "第3步：删除所有 Docker 卷..."
docker-compose -f docker-compose-milvus.yml down -v 2>/dev/null || true
sleep 2

# 第4步：删除本地卷目录
echo "第4步：删除本地卷目录..."
rm -rf ./volumes 2>/dev/null || true
mkdir -p ./volumes
sleep 1

# 第5步：清除 Docker 系统
echo "第5步：清除未使用的 Docker 资源..."
docker system prune -af --volumes 2>/dev/null || true
sleep 2

# 第6步：重新启动容器
echo "第6步：启动全新的 Milvus 容器..."
docker-compose -f docker-compose-milvus.yml up -d
echo "等待容器启动（这需要 2-3 分钟）..."
sleep 45

# 第7步：验证 Milvus 就绪
echo "第7步：验证 Milvus 连接..."
for i in {1..30}; do
    if curl -s http://localhost:9091/healthz > /dev/null 2>&1; then
        echo "✅ Milvus 已就绪"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Milvus 启动失败"
        echo "请查看日志："
        echo "  docker logs milvus-standalone"
        exit 1
    fi
    echo "  等待中... ($i/30)"
    sleep 3
done

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                  ✅ 清理完成！现在所有数据都是新的                        ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 第8步：验证 Python 连接
echo "第8步：验证 Python 连接..."
python3 << 'EOF'
import sys
sys.path.insert(0, '/Users/spike/code/MAS')

try:
    from pymilvus import connections
    connections.connect(host="localhost", port=19530)
    print("✅ Python 连接到全新 Milvus 成功")
    connections.disconnect(alias="default")
except Exception as e:
    print(f"❌ 连接失败: {e}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""
echo "现在运行演示脚本..."
echo ""

# 第9步：运行演示
export DASHSCOPE_API_KEY="sk-f771855105fe43b28584a0f4d68fb5e9"
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_MODEL="text-embedding-v4"

python3 mas/rag/demo_rag_milvus.py

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                   ✅ 演示完成！                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "🎉 恭喜！RAG 系统已成功运行！"
echo ""
echo "下一步：打开 Attu 可视化界面查看向量数据"
echo "  http://localhost:8000"
