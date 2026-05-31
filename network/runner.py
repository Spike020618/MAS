"""
P2P Multi-Agent Consensus Demo — 单机多协程启动器

Usage:
    python -m network.runner [--agents N] [--question "..."]
"""
from __future__ import annotations

import asyncio
import argparse
import logging
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from network.registry import RegistryServer
from network.agent_node import AgentNode

logger = logging.getLogger("runner")

MODELS = ["gpt-4o", "claude-3.5-sonnet", "gemini-pro", "deepseek-v3", "qwen-2.5"]


async def run_demo(
    n_agents: int = 3,
    question: str = "Should autonomous AI systems be given legal personhood?",
    base_port: int = 9000,
):
    """启动 Registry + N 个 Agent，执行一次完整的 P2P 共识流程"""
    print("\n" + "=" * 60)
    print("  P2P Multi-Agent Consensus Demo")
    print("=" * 60)

    # Phase 0: Start Registry
    registry = RegistryServer(port=base_port)
    registry_runner = await registry.start()
    print(f"\n[Registry] Started on port {base_port}")

    # Phase 1: Start Agent Nodes
    agents: list[AgentNode] = []
    runners = []
    for i in range(n_agents):
        agent = AgentNode(
            agent_id=f"agent-{i}",
            model=MODELS[i % len(MODELS)],
            port=base_port + 1 + i,
            registry_url=f"http://localhost:{base_port}",
        )
        runner = await agent.start()
        agents.append(agent)
        runners.append(runner)
        print(f"[Agent-{i}] Registered ({agent.model}), port {base_port + 1 + i}")

    # Small delay for all agents to discover each other
    await asyncio.sleep(0.5)
    for agent in agents:
        await agent._discover_peers()

    # Phase 2: Publish task
    task = {
        "task_id": "demo-001",
        "question": question,
        "context": {"domain": "AI ethics", "type": "multi-hop"},
    }
    print(f"\n[Registry] Task published: \"{question[:60]}...\"")

    # Phase 3: Each agent generates AEIC and broadcasts
    t_start = time.time()
    await asyncio.gather(*[agent.on_task(task) for agent in agents])
    print()
    for agent in agents:
        print(f"[{agent.agent_id}] Generated AEIC record ({len(agent.my_record['E'])} evidence items)")

    # Phase 4: Wait for record exchange
    await asyncio.gather(*[agent.wait_for_records(timeout=10.0) for agent in agents])
    print(f"\n[All] Records exchanged ({n_agents}×{n_agents-1} = {n_agents*(n_agents-1)} messages)")

    # Phase 5: Distributed Stackelberg consensus
    print("\n--- Stackelberg Consensus ---")
    results = await asyncio.gather(*[agent.run_consensus() for agent in agents])

    t_elapsed = time.time() - t_start

    # Summary
    print("\n" + "=" * 60)
    print("  Consensus Results")
    print("=" * 60)
    total_msgs = sum(r["messages_sent"] + r["messages_received"] for r in results)
    avg_theta = sum(r["final_theta"] for r in results) / n_agents
    avg_sim = sum(r["evaluation"]["avg_similarity"] for r in results) / n_agents
    max_rounds = max(r["rounds"] for r in results)
    final_energy = sum(r["final_energy"] for r in results) / n_agents

    for r in results:
        print(f"  [{r['agent_id']}] θ={r['final_theta']:.4f}, "
              f"sim={r['evaluation']['avg_similarity']:.4f}, "
              f"decision={r['evaluation']['decision']}, "
              f"rounds={r['rounds']}")

    print(f"\n  Avg θ:         {avg_theta:.4f}")
    print(f"  Avg Similarity: {avg_sim:.4f}")
    print(f"  Final Energy:   {final_energy:.6f}")
    print(f"  Convergence:    {max_rounds} rounds")
    print(f"  Total Messages: {total_msgs}")
    print(f"  Wall Time:      {t_elapsed:.2f}s")
    print(f"  Decision:       {results[0]['evaluation']['decision']}")
    print(f"\n{'✓' if final_energy < 0.01 else '○'} Consensus {'reached' if final_energy < 0.01 else 'in progress'}"
          f" in {max_rounds} rounds, {total_msgs} messages exchanged")

    # Cleanup
    for agent in agents:
        await agent.close()
    for r in runners:
        await r.cleanup()
    await registry_runner.cleanup()

    return results


def main():
    parser = argparse.ArgumentParser(description="P2P Multi-Agent Consensus Demo")
    parser.add_argument("--agents", "-n", type=int, default=3, help="Number of agents")
    parser.add_argument("--question", "-q", type=str,
                        default="Should autonomous AI systems be given legal personhood?",
                        help="Task question")
    parser.add_argument("--port", "-p", type=int, default=9000, help="Base port")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    # Suppress aiohttp access logs
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)

    asyncio.run(run_demo(n_agents=args.agents, question=args.question, base_port=args.port))


if __name__ == "__main__":
    main()
