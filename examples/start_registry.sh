#!/bin/bash
# 启动Registry Center

echo "================================"
echo "启动 Registry Center"
echo "================================"

cd "$(dirname "$0")/../src"

python registry_center.py --port 9000 --host 0.0.0.0
