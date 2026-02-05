"""
快速开始示例 - 演示如何使用分布式多智能体系统

使用方法：
1. 先启动Registry: python start_registry.py
2. 在另一个终端启动此脚本: python quick_start.py
"""

import asyncio
import httpx
import time
import sys
import os

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent_node import DistributedAgent

REGISTRY_URL = "http://127.0.0.1:9000"

async def check_registry():
    """检查Registry是否运行"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{REGISTRY_URL}/health", timeout=2.0)
            return response.status_code == 200
    except:
        return False

async def wait_for_registry():
    """等待Registry启动"""
    print("等待Registry启动...")
    for i in range(10):
        if await check_registry():
            print("✓ Registry已就绪")
            return True
        await asyncio.sleep(1)
    
    print("❌ 无法连接到Registry")
    print(f"   请确保Registry运行在 {REGISTRY_URL}")
    print("   可以运行: python start_registry.py")
    return False

async def demo_basic_workflow():
    """演示基本工作流程"""
    
    print("\n" + "="*70)
    print("演示：分布式多智能体协作")
    print("="*70 + "\n")
    
    # 步骤1：检查Registry
    if not await wait_for_registry():
        return
    
    # 步骤2：创建Agent节点
    print("\n步骤1：创建Agent节点...")
    print("-" * 70)
    
    # Agent A - Solver (发起者)
    agent_a = DistributedAgent(
        port=8001,
        model="qwen",
        role="initiator",
        registry_url=REGISTRY_URL
    )
    await agent_a.join_network()
    
    # Agent B - Reviewer
    agent_b = DistributedAgent(
        port=8002,
        model="deepseek",
        role="reviewer",
        registry_url=REGISTRY_URL
    )
    await agent_b.join_network()
    
    # Agent C - Solver
    agent_c = DistributedAgent(
        port=8003,
        model="qwen",
        role="solver",
        registry_url=REGISTRY_URL
    )
    await agent_c.join_network()
    
    print("\n✓ 网络已建立，3个Agent在线\n")
    
    await asyncio.sleep(2)
    
    # 步骤3：Agent A发起任务
    print("\n步骤2：Agent A 发起任务...")
    print("-" * 70)
    
    task_desc = "审核企业贷款申请（100万额度）"
    task_data = {
        "assumptions": "初级评估模型",
        "evidence": ["口头陈述", "简单资产证明"],
        "inference": "主观判断",
        "conclusion": "待定"
    }
    
    result = await agent_a.initiate_task(
        task_desc=task_desc,
        task_data=task_data,
        goal="在5轮内达成ESS共识，收益U>55"
    )
    
    # 步骤4：展示结果
    print("\n步骤3：查看结果")
    print("-" * 70)
    
    if result['status'] == 'success':
        print(f"\n✓ 任务成功完成！")
        print(f"   任务ID: {result['task_id']}")
        print(f"   决策: {result['result'].get('decision', 'unknown')}")
        print(f"   收益U: {result['result'].get('utility', 0):.2f}")
        print(f"   轮次: {result['result'].get('rounds', 0)}")
        print(f"   耗时: {result['elapsed_time']:.2f}秒")
        
        print(f"\n📊 收益历史: {result['result'].get('history', [])}")
    else:
        print(f"\n❌ 任务失败: {result.get('reason', 'unknown')}")
    
    # 步骤5：查看统计信息
    print("\n步骤4：查看Agent统计信息")
    print("-" * 70)
    
    for agent in [agent_a, agent_b, agent_c]:
        memory = agent.memory_manager.get_agent_memory(agent.port)
        trust = agent.memory_manager.consensus_memory.get_trust(agent.port)
        stats = memory.get_stats()
        
        print(f"\nAgent {agent.port} ({agent.role}):")
        print(f"  信任分: {trust:.1f}")
        print(f"  成功任务: {stats['successful_tasks']}")
        print(f"  失败任务: {stats['failed_tasks']}")
        print(f"  成功率: {stats['success_rate']:.2%}")
    
    print("\n" + "="*70)
    print("演示完成！")
    print("="*70 + "\n")

async def demo_multiple_tasks():
    """演示多任务处理"""
    
    print("\n" + "="*70)
    print("演示：处理多个任务")
    print("="*70 + "\n")
    
    if not await wait_for_registry():
        return
    
    # 创建Agent
    agent_a = DistributedAgent(8001, "qwen", "initiator", REGISTRY_URL)
    agent_b = DistributedAgent(8002, "deepseek", "reviewer", REGISTRY_URL)
    agent_c = DistributedAgent(8003, "qwen", "solver", REGISTRY_URL)
    
    await agent_a.join_network()
    await agent_b.join_network()
    await agent_c.join_network()
    
    # 定义多个任务
    tasks = [
        {
            "desc": "小额贷款审核（5万）",
            "data": {
                "assumptions": "基础模型",
                "evidence": ["身份证", "银行卡"],
                "inference": "快速验证",
                "conclusion": "待定"
            }
        },
        {
            "desc": "中型企业贷款（50万）",
            "data": {
                "assumptions": "标准模型",
                "evidence": ["营业执照", "财务报表"],
                "inference": "标准流程",
                "conclusion": "待定"
            }
        },
        {
            "desc": "大型项目融资（500万）",
            "data": {
                "assumptions": "高级模型",
                "evidence": ["完整尽调", "担保物"],
                "inference": "深度分析",
                "conclusion": "待定"
            }
        }
    ]
    
    results = []
    for i, task in enumerate(tasks, 1):
        print(f"\n处理任务 {i}/{len(tasks)}: {task['desc']}")
        result = await agent_a.initiate_task(task['desc'], task['data'])
        results.append(result)
        await asyncio.sleep(1)
    
    # 统计
    print("\n" + "="*70)
    print("多任务处理统计")
    print("="*70)
    
    successful = sum(1 for r in results if r.get('status') == 'success')
    avg_utility = sum(r['result'].get('utility', 0) for r in results if r.get('status') == 'success') / len(results)
    avg_rounds = sum(r['result'].get('rounds', 0) for r in results if r.get('status') == 'success') / len(results)
    
    print(f"\n成功率: {successful}/{len(tasks)} ({successful/len(tasks):.1%})")
    print(f"平均收益: {avg_utility:.2f}")
    print(f"平均轮次: {avg_rounds:.1f}")
    
    print("\n" + "="*70 + "\n")

async def main():
    """主函数"""
    
    print("\n" + "="*70)
    print("分布式多智能体系统 - 快速开始")
    print("="*70)
    print("\n选择演示模式：")
    print("  1. 基本工作流程（单任务）")
    print("  2. 多任务处理")
    print("  3. 全部运行")
    
    choice = input("\n请选择 (1/2/3，默认1): ").strip() or "1"
    
    if choice == "1":
        await demo_basic_workflow()
    elif choice == "2":
        await demo_multiple_tasks()
    elif choice == "3":
        await demo_basic_workflow()
        await demo_multiple_tasks()
    else:
        print("无效选择")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n中断执行")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
