"""
共识模块 - Stackelberg Consensus Game的核心实现

包含：
- stackelberg.py: Stackelberg Consensus Game框架
- consensus.py: 共识引擎
- hybrid_semantic_engine.py: 混合语义理解
- agentverse.py: 多智能体流程框架
"""

from .stackelberg import StackelbergConsensusGame, StackelbergScheduler, AgentBid
from .consensus import ConsensusEngine, run_consensus_simulation

__all__ = [
    'StackelbergConsensusGame',
    'StackelbergScheduler',
    'AgentBid',
    'ConsensusEngine',
    'run_consensus_simulation',
]
