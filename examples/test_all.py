"""
测试脚本 - 验证系统各组件是否正常工作

运行此脚本前，请确保：
1. Registry已启动：python src/registry_center.py
2. 至少2个Agent已启动
"""

import asyncio
import httpx
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

REGISTRY_URL = "http://127.0.0.1:9000"

class TestSuite:
    def __init__(self):
        self.passed = 0
        self.failed = 0
    
    async def test_registry_connection(self):
        """测试1: Registry连接"""
        print("\n[测试1] Registry连接测试...")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{REGISTRY_URL}/health", timeout=2.0)
                if response.status_code == 200:
                    print("  ✓ Registry连接正常")
                    self.passed += 1
                    return True
                else:
                    print(f"  ✗ Registry响应异常: {response.status_code}")
                    self.failed += 1
                    return False
        except Exception as e:
            print(f"  ✗ 无法连接到Registry: {e}")
            self.failed += 1
            return False
    
    async def test_agent_discovery(self):
        """测试2: Agent发现"""
        print("\n[测试2] Agent发现测试...")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{REGISTRY_URL}/discover", timeout=2.0)
                result = response.json()
                
                agent_count = result['total']
                print(f"  发现 {agent_count} 个Agent:")
                for agent_id, info in result['agents'].items():
                    print(f"    - {agent_id}: {info['role']} ({info['model']})")
                
                if agent_count >= 2:
                    print(f"  ✓ Agent数量充足 ({agent_count}个)")
                    self.passed += 1
                    return True
                else:
                    print(f"  ⚠️  Agent数量不足 ({agent_count}个，建议至少2个)")
                    self.failed += 1
                    return False
        except Exception as e:
            print(f"  ✗ 发现失败: {e}")
            self.failed += 1
            return False
    
    async def test_consensus_engine(self):
        """测试3: 共识引擎"""
        print("\n[测试3] 共识引擎测试...")
        try:
            from consensus.consensus import ConsensusEngine
            
            engine = ConsensusEngine()
            
            # 测试数据
            proposal_a = {
                "assumptions": "高信用模型V3",
                "evidence": ["资产证明", "流水单", "纳税记录"],
                "inference": "多维交叉验证",
                "conclusion": "批准"
            }
            
            proposal_b = {
                "assumptions": "高信用模型V3",
                "evidence": ["资产证明", "流水单"],
                "inference": "标准验证",
                "conclusion": "批准"
            }
            
            result = engine.evaluate_game(proposal_a, proposal_b)
            
            print(f"  前提相似度: {result['sim_a']:.2f}")
            print(f"  证据重叠率: {result['sim_e']:.2f}")
            print(f"  推理一致性: {result['sim_i']:.2f}")
            print(f"  结论对齐度: {result['sim_c']:.2f}")
            print(f"  综合收益U: {result['utility']:.2f}")
            
            if result['utility'] > 0:
                print("  ✓ 共识引擎工作正常")
                self.passed += 1
                return True
            else:
                print("  ✗ 收益计算异常")
                self.failed += 1
                return False
        except Exception as e:
            print(f"  ✗ 共识引擎错误: {e}")
            import traceback
            traceback.print_exc()
            self.failed += 1
            return False
    
    async def test_memory_system(self):
        """测试4: 记忆系统"""
        print("\n[测试4] 记忆系统测试...")
        try:
            from memory import MemoryManager
            
            manager = MemoryManager()
            
            # 测试短期记忆
            memory = manager.get_agent_memory(8001)
            memory.add_to_short_term("测试消息", "solver")
            
            # 测试情景记忆
            manager.record_task_result(
                agent_id=8001,
                task_desc="测试任务",
                result={"conclusion": "测试"},
                utility=60
            )
            
            # 测试信任分
            trust = manager.consensus_memory.get_trust(8001)
            
            print(f"  短期记忆: {len(memory.short_term)} 条")
            print(f"  情景记忆: {len(memory.episodic)} 条")
            print(f"  信任分: {trust:.1f}")
            
            print("  ✓ 记忆系统工作正常")
            self.passed += 1
            return True
        except Exception as e:
            print(f"  ✗ 记忆系统错误: {e}")
            self.failed += 1
            return False
    
    async def test_task_planner(self):
        """测试5: 任务规划"""
        print("\n[测试5] 任务规划测试...")
        try:
            from task_planner import TaskPlanner
            
            planner = TaskPlanner()
            
            user_request = {
                "description": "测试任务分解",
                "goal": "验证规划器功能"
            }
            
            plan = await planner.decompose(user_request)
            
            print(f"  子任务数: {len(plan.subtasks)}")
            print(f"  复杂度: {plan.complexity}")
            
            for i, subtask in enumerate(plan.subtasks, 1):
                print(f"    {i}. {subtask['description']}")
            
            print("  ✓ 任务规划工作正常")
            self.passed += 1
            return True
        except Exception as e:
            print(f"  ✗ 任务规划错误: {e}")
            self.failed += 1
            return False
    
    async def test_expert_recruiter(self):
        """测试6: 专家招募"""
        print("\n[测试6] 专家招募测试...")
        try:
            from expert_recruiter import ExpertRecruiter
            
            recruiter = ExpertRecruiter(REGISTRY_URL)
            
            # 模拟招募
            experts = await recruiter.recruit_experts(
                task_desc="信用审核任务",
                current_utility=None,
                trust_scores=None
            )
            
            print(f"  招募专家数: {len(experts)}")
            for expert in experts:
                print(f"    - Agent {expert['port']}: {expert['assigned_role']}")
            
            if len(experts) > 0:
                print("  ✓ 专家招募工作正常")
                self.passed += 1
                return True
            else:
                print("  ⚠️  未能招募到专家（可能是没有Agent在线）")
                self.failed += 1
                return False
        except Exception as e:
            print(f"  ✗ 专家招募错误: {e}")
            self.failed += 1
            return False
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("="*70)
        print("系统组件测试")
        print("="*70)
        
        # 测试1-2: 需要Registry
        await self.test_registry_connection()
        await self.test_agent_discovery()
        
        # 测试3-5: 本地组件
        await self.test_consensus_engine()
        await self.test_memory_system()
        await self.test_task_planner()
        
        # 测试6: 需要Registry + Agents
        await self.test_expert_recruiter()
        
        # 总结
        print("\n" + "="*70)
        print("测试总结")
        print("="*70)
        print(f"通过: {self.passed}")
        print(f"失败: {self.failed}")
        print(f"成功率: {self.passed/(self.passed+self.failed)*100:.1f}%")
        print("="*70 + "\n")
        
        if self.failed == 0:
            print("✓ 所有测试通过！系统运行正常。")
        else:
            print(f"⚠️  有 {self.failed} 项测试失败，请检查配置。")

async def main():
    suite = TestSuite()
    await suite.run_all_tests()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n测试中断")
    except Exception as e:
        print(f"\n测试异常: {e}")
        import traceback
        traceback.print_exc()
