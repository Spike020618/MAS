"""
rag_integration_example.py - RAG系统与start.py的集成示例

这个文件展示了如何将RAG系统集成到你现有的博弈驱动系统中
"""

import asyncio
import os
import sys
import argparse

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from mas.rag import (
    LocalRAGDatabase,
    RAGWorkflow,
    RAGSyncManager,
    MultiAgentCoordinator,
    WeightLearningIntegration,
    ExperimentRunner,
    ResultsAnalyzer,
)


# ════════════════════════════════════════════════════════════════════════════
# 示例1：快速开始 - 单个任务分配
# ════════════════════════════════════════════════════════════════════════════

async def example1_simple_allocation():
    """
    最简单的使用示例：初始化系统并分配一个任务
    """
    print("\n" + "=" * 80)
    print("示例1：简单任务分配")
    print("=" * 80)

    # 初始化RAG系统
    rag_db = LocalRAGDatabase(storage_path="./rag_storage_example1")
    await rag_db.initialize()

    # 注册一些Agent
    agents = [
        {"agent_id": 1, "name": "Agent_1", "task_types": ["review"], "success_rate": 0.85},
        {"agent_id": 2, "name": "Agent_2", "task_types": ["review"], "success_rate": 0.80},
        {"agent_id": 3, "name": "Agent_3", "task_types": ["review"], "success_rate": 0.90},
    ]

    for agent in agents:
        await rag_db.register_agent(**agent)

    # 创建工作流
    workflow = RAGWorkflow(rag_db)

    # 分配一个任务
    task_request = {
        "task_id": "task_001",
        "task_type": "review",
        "description": "代码审查：检查代码质量和安全性",
    }

    state = await workflow.allocate_task(
        task_request=task_request,
        task_embedding=await rag_db.embedding.embed(task_request["description"]),
    )

    print(f"\n✅ 任务分配完成")
    print(f"  - 分配的Agent: {state.selected_agents}")
    print(f"  - 分配决策: {state.allocation_decision}")
    print(f"  - 成功评分: {state.all_scores.get('final_score', 0):.4f}")

    await rag_db.close()
    return state


# ════════════════════════════════════════════════════════════════════════════
# 示例2：带权重学习的任务分配
# ════════════════════════════════════════════════════════════════════════════

async def example2_task_allocation_with_learning():
    """
    执行多个任务并通过反馈自动学习权重
    """
    print("\n" + "=" * 80)
    print("示例2：任务分配 + 自动权重学习")
    print("=" * 80)

    # 初始化系统
    rag_db = LocalRAGDatabase(storage_path="./rag_storage_example2")
    await rag_db.initialize()

    # 注册Agent
    agents = [
        {"agent_id": 1, "name": "ReviewExpert", "task_types": ["review"], "success_rate": 0.85},
        {"agent_id": 2, "name": "PlanningMaster", "task_types": ["planning"], "success_rate": 0.80},
    ]
    for agent in agents:
        await rag_db.register_agent(**agent)

    # 创建工作流
    workflow = RAGWorkflow(rag_db)

    # 创建学习集成器
    learner = WeightLearningIntegration(
        rag_database=rag_db,
        rag_workflow=workflow,
        learning_rate=0.01,
    )

    print(f"\n初始权重: {learner.weight_learner.weights}")

    # 执行5个任务，逐步学习改进
    for i in range(5):
        print(f"\n【任务{i+1}】")

        # 执行任务
        result = await learner.execute_task_with_learning(
            task_request={
                "task_id": f"task_{i:03d}",
                "task_type": "review",
                "description": f"代码审查任务{i+1}",
            }
        )

        # 模拟反馈：第一个Agent分配更高分数
        if i < 3:
            success_score = 0.85 + i * 0.03  # 性能逐步改进
        else:
            success_score = 0.90 + (i - 3) * 0.02

        # 处理反馈（自动触发权重学习）
        feedback = await learner.process_feedback_with_learning(
            record_id=f"rec_{i:03d}",
            success_score=success_score,
            agent_ids=result.get("allocated_agents", []),
            feedback_text="任务完成",
        )

        print(f"  分配Agent: {result.get('allocated_agents')}")
        print(f"  反馈分数: {success_score:.2f}")
        print(f"  更新后权重: {feedback['updated_weights']}")

    # 获取最终学习状态
    status = await learner.get_learning_status()
    print(f"\n【最终学习状态】")
    print(f"  总任务数: {status['performance_metrics']['total_tasks']}")
    print(f"  成功任务数: {status['performance_metrics']['successful_tasks']}")
    print(f"  平均成功分数: {status['performance_metrics']['avg_success_score']:.4f}")
    print(f"  学习稳定性: {status['convergence']['learning_stability']:.4f}")

    await rag_db.close()
    return status


# ════════════════════════════════════════════════════════════════════════════
# 示例3：对比实验
# ════════════════════════════════════════════════════════════════════════════

async def example3_comparison_experiment():
    """
    运行三种算法的对比实验
    """
    print("\n" + "=" * 80)
    print("示例3：三算法对比实验")
    print("=" * 80)

    # 初始化系统
    rag_db = LocalRAGDatabase(storage_path="./rag_storage_example3")
    await rag_db.initialize()

    # 运行对比实验
    runner = ExperimentRunner(rag_db)
    results = await runner.run_experiment(
        num_agents=4,
        num_tasks=30,
        seed=42,
    )

    # 分析结果
    analyzer = ResultsAnalyzer()
    metrics_list = []

    print(f"\n【分析各算法性能】")
    for algo_name in ["greedy", "rag", "rag_learning"]:
        if algo_name in results:
            metrics = analyzer.compute_metrics(
                results[algo_name]["results"],
                algo_name,
            )
            metrics_list.append(metrics)

    # 对比分析
    comparison = analyzer.compare_algorithms(metrics_list)

    # 生成报告
    report = analyzer.generate_report(metrics_list, comparison)
    print("\n" + report)

    # 显示赢家
    print(f"\n🏆 总体赢家: {comparison['winner']}")

    await rag_db.close()
    return results, comparison


# ════════════════════════════════════════════════════════════════════════════
# 示例4：多Agent协调
# ════════════════════════════════════════════════════════════════════════════

async def example4_multi_agent_coordination():
    """
    使用多Agent协调器进行跨Agent任务分配
    """
    print("\n" + "=" * 80)
    print("示例4：多Agent协调")
    print("=" * 80)

    # 初始化系统
    rag_db = LocalRAGDatabase(storage_path="./rag_storage_example4")
    await rag_db.initialize()

    # 注册Agent
    agents = [
        {"agent_id": 1, "name": "Agent1", "task_types": ["review"], "success_rate": 0.85},
        {"agent_id": 2, "name": "Agent2", "task_types": ["review"], "success_rate": 0.80},
        {"agent_id": 3, "name": "Agent3", "task_types": ["review"], "success_rate": 0.90},
    ]
    for agent in agents:
        await rag_db.register_agent(**agent)

    # 创建工作流和同步管理器
    workflow = RAGWorkflow(rag_db)
    sync_manager = RAGSyncManager(agent_id=0, agent_name="Coordinator")

    # 注册Agent到同步管理器
    for agent in agents:
        await sync_manager.register_agent(
            agent_id=agent["agent_id"],
            agent_name=agent["name"],
            task_types=agent["task_types"],
            success_rate=agent["success_rate"],
        )

    # 创建协调器
    coordinator = MultiAgentCoordinator(
        agent_id=0,
        agent_name="Coordinator",
        rag_workflow=workflow,
        sync_manager=sync_manager,
    )

    # 分配任务（包含本地和远程选项）
    task_request = {
        "task_id": "task_001",
        "task_type": "review",
        "description": "需要进行代码审查和性能优化",
    }

    result = await coordinator.allocate_task_with_sync(
        task_request=task_request,
        task_embedding=await rag_db.embedding.embed(task_request["description"]),
        enable_remote=True,
    )

    print(f"\n✅ 多Agent协调完成")
    print(f"  - 分配类型: {result.get('type')}")
    print(f"  - 选定Agent: {result.get('selected_agents')}")

    # 发送反馈
    if result.get("selected_agents"):
        target_agent = result["selected_agents"][0]
        await coordinator.send_feedback_to_agent(
            target_agent_id=target_agent,
            record_id="rec_001",
            success_score=0.95,
            feedback_text="任务完成得很好",
        )
        print(f"  - 反馈已发送给Agent {target_agent}")

    await rag_db.close()
    return result


# ════════════════════════════════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(description="RAG系统集成示例")
    parser.add_argument(
        "--example",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="运行指定示例 (1-4)",
    )
    args = parser.parse_args()

    if args.example == 1:
        await example1_simple_allocation()
    elif args.example == 2:
        await example2_task_allocation_with_learning()
    elif args.example == 3:
        await example3_comparison_experiment()
    elif args.example == 4:
        await example4_multi_agent_coordination()

    print("\n✅ 示例完成！\n")


if __name__ == "__main__":
    asyncio.run(main())
