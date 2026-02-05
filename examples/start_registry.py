"""
启动Registry Center的便捷脚本

使用方法：
    python start_registry.py              # 默认端口9000
    python start_registry.py --port 8000  # 自定义端口
"""

import sys
import os

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from registry_center import app
import uvicorn
import argparse

def main():
    parser = argparse.ArgumentParser(description="启动Registry Center")
    parser.add_argument("--port", type=int, default=9000, help="端口号（默认9000）")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址（默认0.0.0.0）")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🌐 Registry Center 启动")
    print("="*70)
    print(f"   端口: {args.port}")
    print(f"   地址: http://{args.host}:{args.port}")
    print("   职责: 服务发现 + 任务公告板 + 历史存储")
    print("   注意: 此服务不做调度决策，仅提供基础设施")
    print("="*70)
    print("\n📋 等待Agent加入网络...\n")
    print("提示: 按Ctrl+C停止服务\n")
    
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n✓ Registry已停止")

if __name__ == "__main__":
    main()
