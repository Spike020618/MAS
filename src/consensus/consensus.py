"""
共识引擎 - 改进版本（集成混合语义引擎）

原始版本使用四个独立的字符匹配算法（SimHash、MinHash、NCD、Cosine）
改进版本使用混合语义引擎，结合BERT语义理解、领域知识库和MinHash

Architecture:
  旧版本: 字符匹配 → 相似度计算
  新版本: 文本 → BERT向量语义 + 领域知识规则 + MinHash集合 → 融合 → 相似度

优势:
  ✓ 识别同义词（"批准" = "同意"）
  ✓ 理解语义关系（"资产" ≈ "财产"）
  ✓ 捕获反义词（"批准" ≠ "拒绝"）
  ✓ 完全本地化（零网络依赖）
  ✓ 论文故事清晰（多层次语义融合）
"""

import pandas as pd
import json
from typing import Dict, Any, Optional
from hybrid_semantic_engine import HybridSemanticEngine


# ================= 改进的共识引擎 =================

class ConsensusEngine:
    """
    博弈驱动的共识引擎 - 使用混合语义引擎
    
    核心改进:
      1. 用混合语义引擎替代四个独立算子
      2. 保持博弈论框架不变（ESS、收益函数等）
      3. 引入多层次语义理解的可解释性
    """
    
    def __init__(self, reward: float = 100, cost: float = 25, 
                 enable_bert: bool = True, device: str = 'cpu'):
        """
        初始化改进的共识引擎
        
        Args:
            reward: 系统奖励 (R)
            cost: 验证成本 (C)
            enable_bert: 是否启用BERT引擎
            device: 运行设备 ('cpu' 或 'cuda')
        """
        # 初始化混合语义引擎
        self.semantic_engine = HybridSemanticEngine(
            enable_bert=enable_bert, 
            device=device
        )
        
        # 博弈参数
        self.R = reward  # 系统奖励
        self.C = cost    # 验证成本
        
        # 权重配置（对应论文：跨层融合语义表示函数）
        self.w = {
            'A': 0.2,  # 前提层
            'E': 0.3,  # 证据层
            'I': 0.2,  # 推理层
            'C': 0.3   # 结论层
        }
        
        print("\n" + "="*70)
        print("共识引擎 - 混合语义版本")
        print("="*70)
        print(f"奖励系数 R: {self.R}")
        print(f"成本系数 C: {self.C}")
        print(f"权重配置: A={self.w['A']}, E={self.w['E']}, "
              f"I={self.w['I']}, C={self.w['C']}")
        print("="*70 + "\n")

    def evaluate_game(self, row_a: Dict, row_b: Dict, 
                     verbose: bool = False) -> Dict[str, Any]:
        """
        计算博弈收益 U = F(row_a, row_b) * R - C
        
        使用混合语义引擎替代原来的四个算子
        
        Args:
            row_a: 提议A (包含 assumptions, evidence, inference, conclusion)
            row_b: 提议B (同上)
            verbose: 是否输出详细信息
            
        Returns:
            评估结果字典
        """
        # 调用混合语义引擎
        result = self.semantic_engine.evaluate_game(row_a, row_b)
        
        # 保持与旧版本兼容的输出格式
        return {
            'sim_a': result['sim_a'],
            'sim_e': result['sim_e'],
            'sim_i': result['sim_i'],
            'sim_c': result['sim_c'],
            'total_score': result['total_score'],
            'utility': result['utility'],
            'method': result['method'],
            'debug': result['debug']
        }

    def make_decision(self, utility: float) -> str:
        """
        基于博弈收益做出决策（演化稳定策略）
        
        Args:
            utility: 博弈收益U
            
        Returns:
            策略选择
        """
        if utility > 55:
            return "ESS_Consensus"
        elif utility > 0:
            return "Audit_Required"
        else:
            return "Inconsistent_Reject"


# ================= 演化流水线 =================

def run_simulation(file_a: str, file_b: str, 
                  enable_bert: bool = True,
                  verbose: bool = False) -> Optional[pd.DataFrame]:
    """
    执行共识演化模拟
    
    Args:
        file_a: Agent A的数据文件
        file_b: Agent B的数据文件
        enable_bert: 是否启用BERT引擎
        verbose: 是否输出详细日志
        
    Returns:
        模拟结果DataFrame
    """
    try:
        df_a = pd.read_csv(file_a)
        df_b = pd.read_csv(file_b)
    except FileNotFoundError as e:
        print(f"❌ 错误：{e}")
        print("   请确保 agent_a.csv 和 agent_b.csv 文件存在")
        return None

    # 初始化引擎
    engine = ConsensusEngine(enable_bert=enable_bert)
    logs = []

    # 按行（回应序列）进行博弈演化分析
    for i in range(min(len(df_a), len(df_b))):
        res = engine.evaluate_game(df_a.iloc[i], df_b.iloc[i], verbose=verbose)
        
        # 演化稳定策略判定
        strategy = engine.make_decision(res['utility'])
        
        log_entry = {
            'Round': i + 1,
            'Task': df_a.iloc[i].get('id', f'Task_{i+1}'),
            'sim_a': res['sim_a'],
            'sim_e': res['sim_e'],
            'sim_i': res['sim_i'],
            'sim_c': res['sim_c'],
            'total_score': res['total_score'],
            'utility': res['utility'],
            'Decision': strategy
        }
        
        # 可选：添加方法信息
        if verbose:
            log_entry['method'] = json.dumps(res['method'], ensure_ascii=False)
        
        logs.append(log_entry)
        
        # 输出进度
        if verbose or (i + 1) % 10 == 0:
            print(f"  轮次 {i+1}: U={res['utility']:.2f} → {strategy}")

    result_df = pd.DataFrame(logs)
    
    print("\n" + "="*70)
    print("📊 共识演化统计")
    print("="*70)
    print(f"总轮次: {len(result_df)}")
    print(f"达成共识 (ESS): {len(result_df[result_df['Decision'] == 'ESS_Consensus'])} 轮")
    print(f"需要审计: {len(result_df[result_df['Decision'] == 'Audit_Required'])} 轮")
    print(f"不一致拒绝: {len(result_df[result_df['Decision'] == 'Inconsistent_Reject'])} 轮")
    print(f"平均得分: {result_df['total_score'].mean():.4f}")
    print(f"平均收益: {result_df['utility'].mean():.2f}")
    print("="*70 + "\n")
    
    return result_df


# ================= 对比分析（旧版本 vs 新版本） =================

def compare_with_legacy(file_a: str, file_b: str) -> pd.DataFrame:
    """
    对比原始版本和改进版本的结果
    
    Args:
        file_a: Agent A数据
        file_b: Agent B数据
        
    Returns:
        对比结果
    """
    print("\n" + "="*70)
    print("🔄 原始版本 vs 混合语义版本 对比")
    print("="*70)
    
    try:
        df_a = pd.read_csv(file_a)
        df_b = pd.read_csv(file_b)
    except FileNotFoundError:
        print("❌ 数据文件不存在")
        return None
    
    # 仅运行第一行作为演示
    sample_a = df_a.iloc[0]
    sample_b = df_b.iloc[0]
    
    print("\n📝 样本数据:")
    print(f"  前提A: {sample_a.get('assumptions', 'N/A')}")
    print(f"  前提B: {sample_b.get('assumptions', 'N/A')}")
    print(f"\n  推理A: {sample_a.get('inference', 'N/A')}")
    print(f"  推理B: {sample_b.get('inference', 'N/A')}")
    print(f"\n  结论A: {sample_a.get('conclusion', 'N/A')}")
    print(f"  结论B: {sample_b.get('conclusion', 'N/A')}")
    
    # 新版本评估
    engine = ConsensusEngine(enable_bert=True)
    result = engine.evaluate_game(sample_a, sample_b)
    
    print("\n✨ 混合语义引擎结果:")
    print(f"  前提相似度 (sim_a): {result['sim_a']:.4f}")
    print(f"    → {result['method']['assumptions']}")
    print(f"\n  证据相似度 (sim_e): {result['sim_e']:.4f}")
    print(f"    → {result['method']['evidence']}")
    print(f"\n  推理相似度 (sim_i): {result['sim_i']:.4f}")
    print(f"    → {result['method']['inference']}")
    print(f"\n  结论相似度 (sim_c): {result['sim_c']:.4f}")
    print(f"    → {result['method']['conclusion']}")
    
    print(f"\n  综合得分: {result['total_score']:.4f}")
    print(f"  博弈收益: {result['utility']:.2f}")
    
    strategy = engine.make_decision(result['utility'])
    print(f"\n  🎯 决策: {strategy}")
    
    print("\n" + "="*70)
    
    return result


# ================= 主程序 =================

if __name__ == "__main__":
    import sys
    import os
    
    # 确定数据文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_a = os.path.join(script_dir, "agent_a.csv")
    file_b = os.path.join(script_dir, "agent_b.csv")
    
    print("\n🚀 启动共识引擎（混合语义版本）\n")
    
    # 执行流水线
    results = run_simulation(file_a, file_b, enable_bert=True, verbose=False)
    
    if results is not None:
        # 输出结果汇总
        print("\n📊 跨层语义共识演化博弈日志:")
        print(results[['Round', 'sim_a', 'sim_e', 'sim_i', 'sim_c', 
                       'total_score', 'utility', 'Decision']].to_string(index=False))
        
        # 保存结果用于论文图表绘制
        output_file = os.path.join(script_dir, "simulation_output_hybrid.csv")
        results.to_csv(output_file, index=False)
        print(f"\n✓ 结果已保存到: {output_file}")
