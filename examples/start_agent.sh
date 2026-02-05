#!/bin/bash
# 启动Agent节点

if [ $# -lt 3 ]; then
    echo "用法: $0 <端口> <模型> <角色> [Registry地址]"
    echo ""
    echo "示例:"
    echo "  $0 8001 qwen solver"
    echo "  $0 8002 deepseek reviewer"
    echo "  $0 8003 qwen initiator http://192.168.1.100:9000"
    echo ""
    echo "参数说明:"
    echo "  端口: Agent监听的端口号"
    echo "  模型: qwen 或 deepseek"
    echo "  角色: solver(求解者) / reviewer(评审者) / initiator(发起者)"
    echo "  Registry地址: 可选，默认为 http://127.0.0.1:9000"
    exit 1
fi

PORT=$1
MODEL=$2
ROLE=$3
REGISTRY=${4:-"http://127.0.0.1:9000"}

echo "================================"
echo "启动 Agent 节点"
echo "================================"
echo "端口: $PORT"
echo "模型: $MODEL"
echo "角色: $ROLE"
echo "Registry: $REGISTRY"
echo "================================"

cd "$(dirname "$0")/../src"

python agent_node.py --port $PORT --model $MODEL --role $ROLE --registry $REGISTRY
