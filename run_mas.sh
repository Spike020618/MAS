#!/bin/bash
# =============================================================================
# MAS 项目一键启动脚本
# Game-Driven Decentralized Semantic Mechanism
#
# 用法: ./run_mas.sh <命令> [选项]
#   ./run_mas.sh help    查看所有命令
# =============================================================================

PROJECT_DIR="/Users/spike/code/MAS"
CONDA_ENV="research"
START_PY="$PROJECT_DIR/start.py"
EVAL_PY="$PROJECT_DIR/mas/eval/run_eval.py"
GEN_PY="$PROJECT_DIR/mas/data/generator.py"
REGISTRY_PY="$PROJECT_DIR/mas/registry_center.py"
AGENT_PY="$PROJECT_DIR/mas/agent_node.py"

# ── 颜色 ──────────────────────────────────────────────
BOLD='\033[1m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
DIM='\033[2m'
NC='\033[0m'

# ── 工具函数 ───────────────────────────────────────────

info()    { echo -e "${CYAN}▶ $*${NC}"; }
success() { echo -e "${GREEN}✓ $*${NC}"; }
warn()    { echo -e "${YELLOW}⚠ $*${NC}"; }
error()   { echo -e "${RED}✗ $*${NC}"; exit 1; }
dim()     { echo -e "${DIM}  $*${NC}"; }
header()  { echo -e "\n${BOLD}${GREEN}$*${NC}\n"; }

check_conda() {
    command -v conda &>/dev/null || error "conda 未安装或不在 PATH 中"
}

activate_env() {
    info "激活 conda 环境: $CONDA_ENV"
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV" || error "无法激活 conda 环境 '$CONDA_ENV'"
    cd "$PROJECT_DIR"
    success "环境就绪  (Python: $(python --version 2>&1))"
    echo ""
}

# ── 命令实现 ───────────────────────────────────────────

cmd_exp() {
    # ./run_mas.sh exp [1|2|3|all] [透传参数]
    local which="${1:-all}"
    shift 2>/dev/null   # 剩余参数透传给 start.py

    case "$which" in
        1|exp1) info "Exp-1: 收敛性验证";       python "$START_PY" --exp 1 "$@" ;;
        2|exp2) info "Exp-2: 语义方法对比";     python "$START_PY" --exp 2 "$@" ;;
        3|exp3) info "Exp-3: 基线对比";         python "$START_PY" --exp 3 "$@" ;;
        all)    info "全部实验";                 python "$START_PY" --exp all "$@" ;;
        *)
            warn "未知实验 '$which'，支持: 1 2 3 all"
            echo "  示例: ./run_mas.sh exp 1 --rounds 20"
            return 1
            ;;
    esac
}

cmd_plot() {
    info "运行全部实验并生成论文图表"
    python "$START_PY" --exp all --plot "$@"
    success "图表保存至 results/figures/"
}

cmd_regen() {
    info "强制重新生成数据集（调用 DeepSeek）"
    python "$START_PY" --regen "$@"
}

cmd_data() {
    # ./run_mas.sh data [--nodes N] [--n N] [--preview] [--export]
    info "数据集操作"
    python "$GEN_PY" "$@"
}

cmd_eval() {
    # ./run_mas.sh eval [mcq|qa|all] [透传参数]
    local task="${1:-all}"
    shift 2>/dev/null

    # 检查 evalscope 是否已安装
    python -c "import evalscope" 2>/dev/null || {
        warn "evalscope 未安装，正在安装..."
        pip install evalscope -q
    }

    case "$task" in
        mcq) info "evalscope MCQ 共识分类评测";         python "$EVAL_PY" --task mcq "$@" ;;
        qa)  info "evalscope QA 共识分析评测（judge）"; python "$EVAL_PY" --task qa  "$@" ;;
        all) info "evalscope 全量评测（MCQ + QA）";     python "$EVAL_PY" --task all "$@" ;;
        *)
            warn "未知评测任务 '$task'，支持: mcq qa all"
            echo "  示例: ./run_mas.sh eval mcq --limit 5"
            return 1
            ;;
    esac
}

cmd_nodes() {
    # ./run_mas.sh nodes [N]  — 启动 N 个节点进程（默认3，含注册中心）
    local n="${1:-3}"

    header "🌐 启动分布式节点网络（${n} 个节点 + 注册中心）"

    # 检查端口是否被占用
    for port in 9000 $(seq 8001 $((8000 + n))); do
        lsof -ti:"$port" &>/dev/null && {
            warn "端口 $port 已被占用，请先运行: ./run_mas.sh stop"
            return 1
        }
    done

    # 启动注册中心
    info "启动注册中心 (port 9000)"
    python "$REGISTRY_PY" --port 9000 &
    REGISTRY_PID=$!
    sleep 1

    # 启动节点
    ROLES=("solver" "reviewer" "solver" "reviewer" "solver")
    for i in $(seq 1 "$n"); do
        local port=$((8000 + i))
        local role="${ROLES[$((i-1))]}"
        info "启动 node_$((i-1)) (port $port, role=$role)"
        python "$AGENT_PY" \
            --port "$port" \
            --model deepseek \
            --role "$role" \
            --registry "http://127.0.0.1:9000" &
        sleep 0.5
    done

    success "网络启动完成，共 ${n} 个节点"
    echo ""
    dim "注册中心: http://127.0.0.1:9000/stats"
    dim "节点发现: http://127.0.0.1:9000/discover"
    dim "停止所有: ./run_mas.sh stop"
    echo ""

    # 等待所有子进程
    wait
}

cmd_stop() {
    info "停止所有节点进程..."
    local killed=0
    for port in 9000 $(seq 8001 8010); do
        local pid
        pid=$(lsof -ti:"$port" 2>/dev/null)
        if [ -n "$pid" ]; then
            kill "$pid" 2>/dev/null && {
                dim "已停止 port $port (PID $pid)"
                ((killed++))
            }
        fi
    done
    [ "$killed" -gt 0 ] && success "已停止 ${killed} 个进程" || info "没有运行中的节点"
}

cmd_clean() {
    warn "即将清理实验结果（不含数据集缓存）"
    read -r -p "  确认？[y/N] " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || { info "已取消"; return; }

    rm -f  "$PROJECT_DIR"/results/exp*.csv
    rm -f  "$PROJECT_DIR"/results/figures/*.png
    rm -rf "$PROJECT_DIR"/evalscope_data/
    rm -rf "$PROJECT_DIR"/results/evalscope/
    success "清理完成（generated_dataset.json 已保留）"
}

cmd_clean_all() {
    warn "即将清理所有结果（含数据集缓存）"
    read -r -p "  确认？[y/N] " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || { info "已取消"; return; }

    rm -f  "$PROJECT_DIR"/results/exp*.csv
    rm -f  "$PROJECT_DIR"/results/figures/*.png
    rm -f  "$PROJECT_DIR"/results/generated_dataset.json
    rm -rf "$PROJECT_DIR"/evalscope_data/
    rm -rf "$PROJECT_DIR"/results/evalscope/
    success "全量清理完成"
}

cmd_status() {
    header "📊 MAS 项目状态"

    echo -e "  ${BOLD}项目目录${NC}  $PROJECT_DIR"
    echo -e "  ${BOLD}Conda 环境${NC} $CONDA_ENV"
    echo ""

    # 数据集
    local ds="$PROJECT_DIR/results/generated_dataset.json"
    if [ -f "$ds" ]; then
        local rounds
        rounds=$(python -c "import json; d=json.load(open('$ds')); print(len(d))" 2>/dev/null)
        success "数据集缓存  ${rounds} 轮  ($(du -sh "$ds" 2>/dev/null | cut -f1))"
    else
        warn "数据集缓存  不存在（首次运行时自动生成）"
    fi

    # 实验结果
    local csv_count
    csv_count=$(ls "$PROJECT_DIR"/results/exp*.csv 2>/dev/null | wc -l | tr -d ' ')
    local png_count
    png_count=$(ls "$PROJECT_DIR"/results/figures/*.png 2>/dev/null | wc -l | tr -d ' ')
    echo -e "  实验 CSV  ${csv_count} 个 | 图表 PNG  ${png_count} 个"

    # evalscope 数据
    local manifest="$PROJECT_DIR/evalscope_data/manifest.json"
    if [ -f "$manifest" ]; then
        success "evalscope 数据  已导出  ($PROJECT_DIR/evalscope_data/)"
    else
        dim "evalscope 数据  未导出（运行 ./run_mas.sh eval 会自动导出）"
    fi

    # 节点状态
    echo ""
    local running=0
    for port in 9000 $(seq 8001 8005); do
        lsof -ti:"$port" &>/dev/null && ((running++))
    done
    [ "$running" -gt 0 ] \
        && success "分布式节点  ${running} 个进程运行中" \
        || dim "分布式节点  未启动"
}

cmd_help() {
    echo -e "${BOLD}${GREEN}"
    echo "  MAS · 博弈驱动的去中心化语义机制"
    echo -e "${NC}"
    echo -e "${BOLD}实验命令${NC}"
    echo "  exp [1|2|3|all] [选项]   运行指定实验"
    echo "    --rounds N               迭代轮数（默认15）"
    echo "    --agents N               节点数（默认5）"
    echo "    --nodes  N               每轮 AEIC 节点数（默认3）"
    echo "    --plot                   生成论文图表"
    echo ""
    echo "  plot                       运行全部实验 + 生成图表"
    echo "  regen [--nodes N]          强制重新生成数据集"
    echo ""
    echo -e "${BOLD}数据命令${NC}"
    echo "  data [--n N] [--nodes N] [--preview] [--export]"
    echo "    --n N        每场景生成几轮（默认1）"
    echo "    --nodes N    每轮节点数（默认3）"
    echo "    --preview    打印随机样例"
    echo "    --export     同时导出为 evalscope 格式"
    echo ""
    echo -e "${BOLD}evalscope 评测${NC}"
    echo "  eval [mcq|qa|all] [选项]  运行共识理解能力评测"
    echo "    --task  mcq              三分类准确率（无需 judge）"
    echo "    --task  qa               分析质量（LLM judge 打分）"
    echo "    --model <name>           被测模型名"
    echo "    --api-url <url>          模型 API URL"
    echo "    --limit N                只测前 N 条（调试用）"
    echo "    --reexport               强制重新导出评测数据"
    echo ""
    echo -e "${BOLD}分布式节点${NC}"
    echo "  nodes [N]                  启动 N 个节点进程（默认3）+ 注册中心"
    echo "  stop                       停止所有节点进程"
    echo ""
    echo -e "${BOLD}维护${NC}"
    echo "  status                     查看项目状态"
    echo "  clean                      清理实验结果（保留数据集）"
    echo "  clean-all                  清理全部（含数据集缓存）"
    echo "  help                       显示本帮助"
    echo ""
    echo -e "${BOLD}典型用法${NC}"
    echo -e "${DIM}"
    echo "  ./run_mas.sh exp all --plot         # 跑完三个实验 + 出图"
    echo "  ./run_mas.sh exp 2 --nodes 4        # 用4节点跑语义方法对比"
    echo "  ./run_mas.sh regen --nodes 4        # 重新生成4节点数据集"
    echo "  ./run_mas.sh eval mcq --limit 5     # 快速验证 evalscope MCQ"
    echo "  ./run_mas.sh eval all               # 完整评测（MCQ + QA judge）"
    echo "  ./run_mas.sh nodes 3                # 本地起3节点分布式网络"
    echo -e "${NC}"
}

# ── 主入口 ─────────────────────────────────────────────

main() {
    check_conda
    local cmd="${1:-help}"

    # 不需要激活环境的命令
    case "$cmd" in
        help|-h|--help) cmd_help;  exit 0 ;;
        status)         cmd_status; exit 0 ;;
        stop)           cmd_stop;   exit 0 ;;
        clean)          cmd_clean;  exit 0 ;;
        clean-all)      cmd_clean_all; exit 0 ;;
    esac

    # 需要激活环境的命令
    activate_env

    case "$cmd" in
        exp)       cmd_exp      "${@:2}" ;;
        plot)      cmd_plot     "${@:2}" ;;
        regen)     cmd_regen    "${@:2}" ;;
        data)      cmd_data     "${@:2}" ;;
        eval)      cmd_eval     "${@:2}" ;;
        nodes)     cmd_nodes    "${@:2}" ;;
        *)
            warn "未知命令: $cmd"
            echo ""
            cmd_help
            exit 1
            ;;
    esac
}

main "$@"
