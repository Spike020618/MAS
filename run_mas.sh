#!/bin/bash

# 运行 MAS 项目 - 生产级 RAG 系统
# Run MAS Project - Production RAG System

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║        🚀 MAS 项目 - 生产级 RAG 系统 (Milvus + DashScope)                ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

show_help() {
    echo "使用方法:"
    echo ""
    echo "  ./run_mas.sh [命令]"
    echo ""
    echo "命令:"
    echo "  (无命令)       启动容器并运行演示"
    echo "  demo           运行演示脚本"
    echo "  start          启动容器"
    echo "  stop           停止容器"
    echo "  clean          清理容器和数据"
    echo "  help           显示此帮助"
    echo ""
    echo "例子:"
    echo "  ./run_mas.sh"
    echo "  ./run_mas.sh demo"
    echo "  ./run_mas.sh start"
    echo ""
}

start_containers() {
    echo "🐳 启动 Milvus 容器..."
    cd "$PROJECT_ROOT"
    docker-compose -f docker-compose-milvus.yml up -d
    
    echo "等待容器启动（约 1-2 分钟）..."
    sleep 60
    
    echo "✅ 容器已启动"
    echo ""
}

stop_containers() {
    echo "停止容器..."
    cd "$PROJECT_ROOT"
    docker-compose -f docker-compose-milvus.yml stop
    
    echo "✅ 容器已停止"
    echo ""
}

clean_containers() {
    echo "清理容器和数据..."
    cd "$PROJECT_ROOT"
    
    docker-compose -f docker-compose-milvus.yml down -v 2>/dev/null || true
    rm -rf ./volumes/* 2>/dev/null || true
    
    echo "✅ 清理完成"
    echo ""
}

run_demo() {
    echo "检查容器..."
    if ! curl -s http://localhost:9091/healthz > /dev/null 2>&1; then
        echo "⚠️  容器未就绪，启动容器..."
        start_containers
    fi
    
    echo "📊 运行 RAG 演示..."
    echo ""
    
    export DASHSCOPE_API_KEY="sk-f771855105fe43b28584a0f4d68fb5e9"
    export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
    export DASHSCOPE_MODEL="text-embedding-v4"
    
    cd "$PROJECT_ROOT"
    python3 mas/rag/demo_rag_milvus.py
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║                    ✅ 演示完成！                                          ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📊 访问管理界面:"
    echo "  • Attu (向量管理): http://localhost:8000"
    echo "  • MinIO (存储管理): http://localhost:9001"
    echo ""
}

main() {
    local command=$1
    
    case "$command" in
        ""|demo)
            start_containers
            run_demo
            ;;
        start)
            start_containers
            ;;
        stop)
            stop_containers
            ;;
        clean)
            clean_containers
            ;;
        help)
            show_help
            ;;
        *)
            echo "❌ 未知命令: $command"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

main "$@"
