#!/bin/bash

# ════════════════════════════════════════════════════════════════════════════
# RAG系统快速启动脚本
# Quick Start Script for RAG System
# ════════════════════════════════════════════════════════════════════════════

set -e

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                      🚀 RAG系统快速启动                                  ║"
echo "║                    RAG System Quick Start                                ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 显示使用帮助
show_help() {
    echo "使用方法 (Usage):"
    echo ""
    echo "  ./quick_start.sh [命令] [选项]"
    echo ""
    echo "命令 (Commands):"
    echo ""
    echo "  demo          运行演示脚本 (Run demo)"
    echo "    - demo 1    运行第1步演示: 基础存储"
    echo "    - demo 2    运行第2步演示: 工作流"
    echo "    - demo 3    运行第3步演示: 通信"
    echo "    - demo 4    运行第4步演示: 权重学习"
    echo "    - demo 5    运行第5步演示: 对比实验 (推荐首选!)"
    echo ""
    echo "  example       运行集成示例 (Run integration examples)"
    echo "    - example 1 简单任务分配 (Simple task allocation)"
    echo "    - example 2 任务+自动学习 (Task + auto learning)"
    echo "    - example 3 完整对比实验 (Full comparison)"
    echo "    - example 4 多Agent协调 (Multi-agent coordination)"
    echo ""
    echo "  docs          查看文档 (View documentation)"
    echo "    - docs 指南列表 (List all guides)"
    echo "    - docs quick      查看快速参考 (Quick reference)"
    echo "    - docs usage      查看使用指南 (Usage guide)"
    echo "    - docs final      查看最终完整指南 (Final guide)"
    echo ""
    echo "  check         检查安装 (Check installation)"
    echo ""
    echo "  help          显示此帮助信息 (Show this help)"
    echo ""
    echo "示例 (Examples):"
    echo ""
    echo "  # 运行完整对比实验演示"
    echo "  ./quick_start.sh demo 5"
    echo ""
    echo "  # 运行集成示例(任务+学习)"
    echo "  ./quick_start.sh example 2"
    echo ""
    echo "  # 查看快速参考"
    echo "  ./quick_start.sh docs quick"
    echo ""
}

# 检查Python环境
check_environment() {
    echo "📋 检查环境..."
    
    if ! command -v python3 &> /dev/null; then
        echo "❌ 错误: 未找到 python3"
        exit 1
    fi
    
    echo "✅ Python3 已安装"
    
    if [ ! -d "$PROJECT_ROOT/mas/rag" ]; then
        echo "❌ 错误: 未找到RAG模块目录"
        exit 1
    fi
    
    echo "✅ RAG模块目录存在"
    echo ""
}

# 检查安装
check_installation() {
    echo "🔍 检查RAG系统安装..."
    echo ""
    
    cd "$PROJECT_ROOT"
    
    python3 << 'EOF'
import sys
sys.path.insert(0, '/Users/spike/code/MAS')

try:
    from mas.rag import (
        LocalRAGDatabase,
        RAGWorkflow,
        WeightLearningIntegration,
        ExperimentRunner,
        ResultsAnalyzer,
    )
    print("✅ RAG系统所有模块导入成功!")
    print("")
    print("可用的模块:")
    print("  • LocalRAGDatabase         - 向量存储")
    print("  • RAGWorkflow             - 工作流")
    print("  • WeightLearningIntegration - 权重学习")
    print("  • ExperimentRunner        - 实验运行")
    print("  • ResultsAnalyzer         - 结果分析")
    print("")
    print("✅ 安装检查通过!")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)
EOF
}

# 运行演示
run_demo() {
    local demo_num=$1
    
    if [ -z "$demo_num" ]; then
        echo "❌ 错误: 请指定演示步骤 (1-5)"
        echo ""
        echo "用法: ./quick_start.sh demo [1|2|3|4|5]"
        exit 1
    fi
    
    echo "▶️  运行第${demo_num}步演示..."
    echo ""
    
    cd "$PROJECT_ROOT"
    python3 -m mas.rag.demo_step${demo_num}
}

# 运行集成示例
run_example() {
    local example_num=$1
    
    if [ -z "$example_num" ]; then
        echo "❌ 错误: 请指定示例编号 (1-4)"
        echo ""
        echo "用法: ./quick_start.sh example [1|2|3|4]"
        exit 1
    fi
    
    echo "▶️  运行示例${example_num}..."
    echo ""
    
    cd "$PROJECT_ROOT"
    python3 rag_integration_example.py --example $example_num
}

# 显示文档
show_docs() {
    local doc_type=$1
    
    if [ -z "$doc_type" ] || [ "$doc_type" = "指南列表" ]; then
        echo "📚 可用文档列表:"
        echo ""
        echo "快速参考:"
        echo "  • QUICK_REFERENCE.md      - 30秒快速开始"
        echo "  • HOW_TO_USE_FINAL.md     - 最终完整指南"
        echo ""
        echo "使用指南:"
        echo "  • HOW_TO_USE_RAG_SYSTEM.md - 详细使用指南"
        echo "  • RAG_PROJECT_OVERVIEW.md  - 项目架构总览"
        echo ""
        echo "步骤指南:"
        echo "  • RAG_STEP1_GUIDE.md      - 第1步: 基础存储"
        echo "  • RAG_STEP2_GUIDE.md      - 第2步: 工作流"
        echo "  • RAG_STEP3_GUIDE.md      - 第3步: 通信"
        echo "  • RAG_STEP4_GUIDE.md      - 第4步: 学习"
        echo "  • RAG_STEP5_GUIDE.md      - 第5步: 对比实验"
        echo ""
        echo "完成总结:"
        echo "  • STEP1_COMPLETION.md     - 第1步完成总结"
        echo "  • STEP2_COMPLETION.md     - 第2步完成总结"
        echo "  • STEP3_COMPLETION.md     - 第3步完成总结"
        echo "  • STEP4_COMPLETION.md     - 第4步完成总结"
        echo "  • STEP5_COMPLETION.md     - 第5步完成总结"
        echo ""
        echo "用法: ./quick_start.sh docs [quick|usage|final|overview|step1-5|completion]"
        return
    fi
    
    local doc_file=""
    
    case "$doc_type" in
        quick)
            doc_file="QUICK_REFERENCE.md"
            ;;
        usage)
            doc_file="HOW_TO_USE_RAG_SYSTEM.md"
            ;;
        final)
            doc_file="HOW_TO_USE_FINAL.md"
            ;;
        overview)
            doc_file="RAG_PROJECT_OVERVIEW.md"
            ;;
        step1|step2|step3|step4|step5)
            doc_file="RAG_${doc_type^^}_GUIDE.md"
            ;;
        completion1|completion2|completion3|completion4|completion5)
            step_num=${doc_type: -1}
            doc_file="STEP${step_num}_COMPLETION.md"
            ;;
        *)
            echo "❌ 错误: 未知的文档类型"
            show_docs
            exit 1
            ;;
    esac
    
    if [ ! -f "$PROJECT_ROOT/$doc_file" ]; then
        echo "❌ 错误: 文件不存在: $doc_file"
        exit 1
    fi
    
    echo "📖 显示文档: $doc_file"
    echo ""
    cat "$PROJECT_ROOT/$doc_file" | less
}

# 主函数
main() {
    local command=$1
    local option=$2
    
    case "$command" in
        demo)
            check_environment
            run_demo "$option"
            ;;
        example)
            check_environment
            run_example "$option"
            ;;
        docs)
            show_docs "$option"
            ;;
        check)
            check_environment
            check_installation
            ;;
        help)
            show_help
            ;;
        "")
            echo "❌ 错误: 请指定命令"
            echo ""
            show_help
            exit 1
            ;;
        *)
            echo "❌ 错误: 未知的命令: $command"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"
