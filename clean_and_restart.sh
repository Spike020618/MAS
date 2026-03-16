#!/bin/bash

# 最终解决方案：彻底清除所有数据并重新启动

set -e

cd /Users/spike/code/MAS

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║              🔥 最终清理 - 删除所有持久化数据                             ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 第1步：停止容器
echo "第1步：停止所有容器..."
docker-compose -f docker-compose-milvus.yml stop 2>/dev/null || true
sleep 2

# 第2步：删除容器和卷
echo "第2步：删除容器和卷..."
docker-compose -f docker-compose-milvus.yml down -v 2>/dev/null || true
sleep 2

# 第3步：手动删除所有卷目录
echo "第3步：删除所有本地卷数据..."
rm -rf ./volumes/etcd 2>/dev/null || true
rm -rf ./volumes/minio 2>/dev/null || true
rm -rf ./volumes/milvus 2>/dev/null || true
mkdir -p ./volumes
sleep 1

# 第4步：删除本地 RAG 存储
echo "第4步：删除本地 RAG 存储缓存..."
rm -rf ./rag_storage_milvus 2>/dev/null || true
sleep 1

# 第5步：清除 Docker 系统资源
echo "第5步：清除 Docker 系统..."
docker system prune -af --volumes 2>/dev/null || true
sleep 2

# 第6步：重新启动所有容器
echo "第6步：启动全新 Milvus 容器（这需要 2-3 分钟）..."
docker-compose -f docker-compose-milvus.yml up -d
echo "等待容器启动..."
sleep 50

# 第7步：验证容器就绪
echo "第7步：检查容器状态..."
docker-compose -f docker-compose-milvus.yml ps

echo ""
echo "等待 Milvus 完全就绪..."
for i in {1..60}; do
    if curl -s http://localhost:9091/healthz > /dev/null 2>&1; then
        echo "✅ Milvus HTTP 端口已就绪"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "❌ 超时"
        exit 1
    fi
    echo -n "."
    sleep 2
done

# 额外等待时间确保 Proxy 启动
sleep 30

# 第8步：验证 Python 连接
echo ""
echo "第8步：验证 Python 连接..."
python3 << 'EOF'
import sys
sys.path.insert(0, '/Users/spike/code/MAS')

from pymilvus import connections

print("尝试连接 Milvus...")
for attempt in range(20):
    try:
        connections.connect(host="localhost", port=19530)
        print("✅ 连接成功")
        connections.disconnect(alias="default")
        break
    except Exception as e:
        print(f"  尝试 {attempt + 1}... ", end="", flush=True)
        import time
        time.sleep(2)
else:
    print("\n❌ 连接失败")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║              ✅ 所有数据已清除，系统已重新启动！                           ║"
echo "║                     现在运行演示脚本...                                   ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 第9步：运行演示
export DASHSCOPE_API_KEY="sk-f771855105fe43b28584a0f4d68fb5e9"
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_MODEL="text-embedding-v4"

python3 mas/rag/demo_rag_milvus.py

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                  ✅ 演示成功运行！                                        ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "🎉 恭喜！生产级 RAG 系统已成功启动！"
echo ""
echo "📊 下一步操作："
echo "   1. 打开 Attu: http://localhost:8000"
echo "   2. 查看向量数据"
echo "   3. 在浏览器中可视化管理"
