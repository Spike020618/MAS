#!/bin/bash

# 生产级 RAG 系统 - 快速启动脚本
# Quick Start Script for Production RAG System

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║            🚀 RAG 系统演示 - Milvus + DashScope (生产级)                 ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 显示帮助
show_help() {
    echo "使用方法:"
    echo ""
    echo "  ./quick_start.sh [命令]"
    echo ""
    echo "命令:"
    echo "  (无命令)    直接运行演示"
    echo "  run         运行演示脚本"
    echo "  help        显示此帮助"
    echo ""
    echo "前置条件:"
    echo "  • Docker 容器必须已启动"
    echo "  • 运行: docker-compose -f docker-compose-milvus.yml up -d"
    echo ""
    echo "例子:"
    echo "  ./quick_start.sh"
    echo "  ./quick_start.sh run"
    echo ""
}

# 检查容器
check_container() {
    echo "检查 Milvus 容器..."
    
    if ! curl -s http://localhost:9091/healthz > /dev/null 2>&1; then
        echo ""
        echo "❌ Milvus 容器未运行或未就绪"
        echo ""
        echo "请先启动容器:"
        echo "  docker-compose -f docker-compose-milvus.yml up -d"
        echo ""
        echo "等待容器启动（约 1-2 分钟）:"
        echo "  sleep 60"
        echo ""
        exit 1
    fi
    
    echo "✅ Milvus 已就绪"
    echo ""
}

# 运行演示
run_demo() {
    echo "📊 运行 RAG 演示脚本..."
    echo ""
    
    # 设置环境变量
    export DASHSCOPE_API_KEY="sk-f771855105fe43b28584a0f4d68fb5e9"
    export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
    export DASHSCOPE_MODEL="text-embedding-v4"
    
    cd "$PROJECT_ROOT"
    
    # 运行演示
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

# 主函数
main() {
    local command=$1
    
    case "$command" in
        run|"")
            check_container
            run_demo
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
