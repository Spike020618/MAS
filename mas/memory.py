"""
Memory System - 多智能体记忆管理

实现：
1. 短期记忆（当前对话）
2. 长期记忆（历史案例）
3. 共识记忆（已达成的共识）
4. 情景记忆（交互历史）
"""

import time
import json
from typing import List, Dict, Optional
from collections import deque
from datasketch import MinHash

class AgentMemory:
    """Agent的记忆系统"""
    
    def __init__(self, max_short_term=50, max_episodic=100):
        # 短期记忆（工作记忆）
        self.short_term = deque(maxlen=max_short_term)
        
        # 长期记忆（持久化案例）
        self.long_term = {}
        
        # 情景记忆（交互历史）
        self.episodic = deque(maxlen=max_episodic)
        
        # 统计信息
        self.stats = {
            "total_memories": 0,
            "successful_tasks": 0,
            "failed_tasks": 0
        }
    
    def add_to_short_term(self, msg: str, role: str, metadata: Dict = None):
        """添加到短期记忆"""
        memory_item = {
            "role": role,
            "content": msg,
            "timestamp": time.time(),
            "round": len(self.short_term),
            "metadata": metadata or {}
        }
        
        self.short_term.append(memory_item)
        self.stats["total_memories"] += 1
    
    def add_to_episodic(self, task_desc: str, result: Dict, utility: float):
        """添加到情景记忆"""
        episode = {
            "task": task_desc,
            "result": result,
            "utility": utility,
            "timestamp": time.time(),
            "success": utility > 55  # ESS阈值
        }
        
        self.episodic.append(episode)
        
        if episode["success"]:
            self.stats["successful_tasks"] += 1
        else:
            self.stats["failed_tasks"] += 1
    
    def add_to_long_term(self, key: str, value: Dict):
        """添加到长期记忆"""
        self.long_term[key] = {
            "value": value,
            "created_at": time.time(),
            "accessed_count": 0
        }
    
    def retrieve_from_long_term(self, key: str) -> Optional[Dict]:
        """从长期记忆检索"""
        if key in self.long_term:
            self.long_term[key]["accessed_count"] += 1
            self.long_term[key]["last_accessed"] = time.time()
            return self.long_term[key]["value"]
        return None
    
    def search_similar_episodes(self, task_desc: str, top_k: int = 3) -> List[Dict]:
        """检索相似的历史案例"""
        if not self.episodic:
            return []
        
        # 使用MinHash计算相似度
        query_minhash = self._text_to_minhash(task_desc)
        
        similarities = []
        for episode in self.episodic:
            ep_minhash = self._text_to_minhash(episode['task'])
            similarity = query_minhash.jaccard(ep_minhash)
            similarities.append((similarity, episode))
        
        # 返回最相似的top_k个
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in similarities[:top_k]]
    
    def _text_to_minhash(self, text: str) -> MinHash:
        """将文本转换为MinHash"""
        m = MinHash(num_perm=128)
        for i in range(len(text) - 1):
            m.update(text[i:i+2].encode('utf8'))
        return m
    
    def get_short_term_context(self, last_n: int = 10) -> str:
        """获取短期记忆的上下文"""
        recent = list(self.short_term)[-last_n:]
        context = "\n".join([
            f"[{m['role']}]: {m['content']}"
            for m in recent
        ])
        return context
    
    def clear_short_term(self):
        """清空短期记忆"""
        self.short_term.clear()
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        success_rate = 0
        if self.stats["successful_tasks"] + self.stats["failed_tasks"] > 0:
            success_rate = self.stats["successful_tasks"] / (
                self.stats["successful_tasks"] + self.stats["failed_tasks"]
            )
        
        return {
            **self.stats,
            "success_rate": success_rate,
            "short_term_size": len(self.short_term),
            "episodic_size": len(self.episodic),
            "long_term_size": len(self.long_term)
        }


class ConsensusMemory:
    """全局共识记忆（共享知识库）"""
    
    def __init__(self):
        # 已达成共识的事实
        self.consensus_facts = {}
        
        # 信任分数（Agent信誉）
        self.trust_scores = {}
    
    def add_consensus(self, task_id: str, facts: Dict, utility: float, participants: List[int]):
        """添加共识事实"""
        if utility > 55:  # 只存储达成ESS的共识
            self.consensus_facts[task_id] = {
                "facts": facts,
                "utility": utility,
                "participants": participants,
                "timestamp": time.time()
            }
    
    def get_consensus(self, task_id: str) -> Optional[Dict]:
        """获取共识事实"""
        return self.consensus_facts.get(task_id)
    
    def update_trust(self, agent_id: int, utility: float):
        """更新Agent信任分"""
        if agent_id not in self.trust_scores:
            self.trust_scores[agent_id] = 50  # 初始分50
        
        # 根据收益调整信任分
        if utility > 55:
            self.trust_scores[agent_id] += 5
        elif utility < 0:
            self.trust_scores[agent_id] -= 3
        
        # 限制在[0, 100]
        self.trust_scores[agent_id] = max(0, min(100, self.trust_scores[agent_id]))
    
    def get_trust(self, agent_id: int) -> float:
        """获取Agent信任分"""
        return self.trust_scores.get(agent_id, 50)
    
    def get_top_trusted_agents(self, top_k: int = 5) -> List[tuple]:
        """获取信任分最高的Agent"""
        sorted_agents = sorted(
            self.trust_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_agents[:top_k]


class MemoryManager:
    """内存管理器（协调所有记忆系统）"""
    
    def __init__(self):
        self.agent_memories = {}  # {agent_id: AgentMemory}
        self.consensus_memory = ConsensusMemory()
    
    def get_agent_memory(self, agent_id: int) -> AgentMemory:
        """获取Agent的记忆系统"""
        if agent_id not in self.agent_memories:
            self.agent_memories[agent_id] = AgentMemory()
        return self.agent_memories[agent_id]
    
    def record_task_result(
        self, 
        agent_id: int, 
        task_desc: str, 
        result: Dict, 
        utility: float
    ):
        """记录任务结果"""
        # 更新Agent记忆
        memory = self.get_agent_memory(agent_id)
        memory.add_to_episodic(task_desc, result, utility)
        
        # 更新信任分
        self.consensus_memory.update_trust(agent_id, utility)
    
    def get_similar_experiences(
        self, 
        agent_id: int, 
        task_desc: str, 
        top_k: int = 3
    ) -> List[Dict]:
        """检索相似经验"""
        memory = self.get_agent_memory(agent_id)
        return memory.search_similar_episodes(task_desc, top_k)
    
    def export_memory(self, filepath: str):
        """导出记忆到文件"""
        data = {
            "agent_memories": {
                aid: {
                    "stats": mem.get_stats(),
                    "episodic": list(mem.episodic)
                }
                for aid, mem in self.agent_memories.items()
            },
            "consensus_facts": self.consensus_memory.consensus_facts,
            "trust_scores": self.consensus_memory.trust_scores,
            "timestamp": time.time()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_memory(self, filepath: str):
        """从文件加载记忆"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 恢复共识记忆
            self.consensus_memory.consensus_facts = data.get("consensus_facts", {})
            self.consensus_memory.trust_scores = data.get("trust_scores", {})
            
            print(f"✓ 记忆已从 {filepath} 加载")
            return True
        except FileNotFoundError:
            print(f"⚠️  记忆文件 {filepath} 不存在")
            return False
