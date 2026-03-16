#!/bin/bash

# RAG 系统安装脚本 - Milvus + DashScope 版本

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║     🚀 RAG 系统安装 - Milvus + DashScope 版本                             ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 检查 Python
echo "📋 检查 Python 环境..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未找到"
    exit 1
fi
echo "✅ Python3 已安装: $(python3 --version)"

# 检查 pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 未找到"
    exit 1
fi
echo "✅ pip3 已安装"

# 检查 Docker
echo ""
echo "📋 检查 Docker..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未找到"
    echo "   请访问: https://www.docker.com/products/docker-desktop"
    exit 1
fi
echo "✅ Docker 已安装: $(docker --version)"

# 检查 Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose 未找到"
    exit 1
fi
echo "✅ Docker Compose 已安装"

# 安装 Python 依赖
echo ""
echo "📦 安装 Python 依赖..."

dependencies=(
    "pymilvus>=2.3.0"
    "httpx>=0.24.0"
    "numpy>=1.21.0"
)

for dep in "${dependencies[@]}"; do
    echo "  安装 $dep..."
    pip3 install -q "$dep"
    if [ $? -eq 0 ]; then
        echo "    ✅ $dep 已安装"
    else
        echo "    ❌ 安装失败: $dep"
        exit 1
    fi
done

# 启动 Milvus
echo ""
echo "🐳 启动 Milvus Docker 容器..."
echo "  这可能需要几分钟..."

cd /Users/spike/code/MAS

# 检查是否已经运行
if docker-compose -f docker-compose-milvus.yml ps | grep -q "milvus-standalone"; then
    echo "✅ Milvus 已在运行"
else
    echo "  启动容器..."
    docker-compose -f docker-compose-milvus.yml up -d
    
    # 等待 Milvus 启动
    echo "  等待 Milvus 就绪..."
    for i in {1..30}; do
        if curl -s http://localhost:9091/healthz > /dev/null; then
            echo "✅ Milvus 已启动"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "❌ Milvus 启动超时"
            exit 1
        fi
        echo -n "."
        sleep 2
    done
fi

# 验证连接
echo ""
echo "🔍 验证 Milvus 连接..."

python3 << 'EOF'
import sys
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

if [ $? -ne 0 ]; then
    exit 1
fi

# 验证 DashScope 配置
echo ""
echo "🔑 验证 DashScope 配置..."

python3 << 'EOF'
import sys
sys.path.insert(0, '/Users/spike/code/MAS')

from mas.rag.config import RAGConfig

if RAGConfig.validate():
    print("✅ DashScope 配置有效")
else:
    print("⚠️  DashScope 配置有问题")
    sys.exit(1)
EOF

# 显示访问地址
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                    ✅ 安装完成！                                           ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 重要地址："
echo "  🔗 Milvus gRPC:   localhost:19530"
echo "  🔗 Attu 管理界面: http://localhost:8000"
echo "  🔗 MinIO 控制台:  http://localhost:9001"
echo ""
echo "🧪 运行演示："
echo "  python3 mas/rag/demo_rag_milvus.py"
echo ""
echo "📚 查看文档："
echo "  cat RAG_MILVUS_DASHSCOPE_GUIDE.md"
echo ""
echo "🚀 快速开始："
echo "  1. 配置 DashScope API 密钥"
echo "  2. 运行演示脚本"
echo "  3. 在 Attu 中查看向量数据"
echo ""
