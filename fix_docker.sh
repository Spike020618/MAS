#!/bin/bash

# 修复 Docker Milvus 启动问题的脚本

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║          🔧 修复 Docker Milvus 启动问题                                   ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

cd /Users/spike/code/MAS

echo "1️⃣  停止所有容器..."
docker-compose -f docker-compose-milvus.yml down -v 2>/dev/null || true

echo ""
echo "2️⃣  清除旧的网络..."
docker network rm milvus 2>/dev/null || true

echo ""
echo "3️⃣  清除未使用的资源..."
docker system prune -f --volumes 2>/dev/null || true

echo ""
echo "4️⃣  启动容器..."
docker-compose -f docker-compose-milvus.yml up -d

echo ""
echo "5️⃣  等待 Milvus 就绪（最多2分钟）..."
for i in {1..60}; do
    if curl -s http://localhost:9091/healthz > /dev/null 2>&1; then
        echo "✅ Milvus 已就绪！"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "❌ Milvus 启动超时"
        echo ""
        echo "📋 调试信息："
        echo "  检查容器日志："
        echo "    docker logs milvus-standalone"
        echo ""
        echo "  检查容器状态："
        echo "    docker-compose -f docker-compose-milvus.yml ps"
        exit 1
    fi
    printf "."
    sleep 2
done

echo ""
echo "6️⃣  验证所有容器..."
docker-compose -f docker-compose-milvus.yml ps

echo ""
echo "7️⃣  检查连接..."
python3 << 'EOF'
import sys
import time
sys.path.insert(0, '/Users/spike/code/MAS')

try:
    from pymilvus import connections
    connections.connect(
        alias="default",
        host="localhost",
        port=19530,
    )
    print("✅ 成功连接到 Milvus")
    connections.disconnect(alias="default")
except Exception as e:
    print(f"❌ 无法连接到 Milvus: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║                    ✅ 修复完成！                                           ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "🔗 重要地址："
    echo "  Milvus gRPC：    localhost:19530"
    echo "  Attu 管理界面：  http://localhost:8000"
    echo "  MinIO 控制台：   http://localhost:9001"
    echo ""
    echo "🧪 现在可以运行演示："
    echo "  python3 mas/rag/demo_rag_milvus.py"
    echo ""
else
    exit 1
fi
