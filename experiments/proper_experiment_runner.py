"""
基于 MAS ConsensusEngine 的正确实验框架
======================================================================

这个版本正确地使用了 MAS 系统中已有的 ConsensusEngine，
而不是重新实现简化版本。

关键改进：
1. 基于真实的 ConsensusEngine（支持多种相似度方法、权重配置）
2. 三个Baseline通过配置不同的权重和相似度方法来区分
3. Proposed方法通过权重自适应学习来改进
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from scipy.stats import ttest_ind
import sys
import traceback

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from proper_baselines import create_all_baselines
from dataset_generator import DatasetGenerator


class ProperExperimentRunner:
    """基于真实ConsensusEngine的实验运行器"""
    
    def __init__(self, output_dir: str = './results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def run_consensus_experiments(self, num_tasks: int = 21, num_runs: int = 20) -> Dict[str, Any]:
        """运行共识实验"""
        
        print("\n" + "="*80)
        print("基于MAS ConsensusEngine的对比实验")
        print("="*80)
        
        # 生成数据集
        print(f"\n生成 {num_tasks} 个测试任务...")
        dataset = DatasetGenerator.create_simple_dataset(num_tasks)
        
        # 创建Baseline
        print("初始化Baseline...")
        baselines = create_all_baselines()
        
        consensus_results = {}
        
        # 运行实验
        for method_name, baseline in baselines.items():
            print(f"\n运行 {method_name}...")
            try:
                result = baseline.run_experiment(dataset, num_runs=num_runs)
                consensus_results[method_name] = result
                
                print(f"  平均相似度: {result['mean']:.4f} ± {result['std']:.4f}")
                print(f"  95% CI: [{result['ci_low']:.4f}, {result['ci_high']:.4f}]")
                print(f"  样本数: {result['n']}")
            except Exception as e:
                print(f"  ✗ 错误: {e}")
                traceback.print_exc()
                consensus_results[method_name] = {
                    'mean': 0.0, 'std': 0.0, 'ci_low': 0.0, 'ci_high': 0.0,
                    'results': [], 'n': 0, 'error': str(e)
                }
        
        self.results['consensus'] = consensus_results
        return consensus_results
    
    def run_statistical_tests(self, consensus_results: Dict[str, Any]):
        """进行统计检验"""
        
        print("\n" + "="*80)
        print("统计显著性检验 (t-test)")
        print("="*80)
        
        proposed_results = np.array(consensus_results.get('Proposed', {}).get('results', []))
        
        if len(proposed_results) == 0:
            print("✗ 无法进行统计检验（Proposed结果为空）")
            return {}
        
        statistical_tests = {}
        
        for baseline_name in ['ChatEval', 'NamingGame', 'LeaderFollowing']:
            if baseline_name not in consensus_results:
                continue
            
            baseline_results = np.array(consensus_results[baseline_name].get('results', []))
            
            if len(baseline_results) == 0:
                print(f"\n{baseline_name}: 无有效数据")
                continue
            
            # t-test
            t_stat, p_value = ttest_ind(proposed_results, baseline_results)
            
            # Cohen's d
            mean_diff = np.mean(proposed_results) - np.mean(baseline_results)
            pooled_std = np.sqrt((np.std(proposed_results)**2 + np.std(baseline_results)**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            # 相对改进
            relative_improvement = (mean_diff / np.mean(baseline_results) * 100) if np.mean(baseline_results) > 0 else 0
            
            statistical_tests[baseline_name] = {
                't_statistic': round(float(t_stat), 4),
                'p_value': float(p_value),
                'cohens_d': round(float(cohens_d), 4),
                'relative_improvement_percent': round(float(relative_improvement), 2),
                'is_significant': p_value < 0.05
            }
            
            print(f"\nProposed vs {baseline_name}:")
            print(f"  t统计量: {t_stat:.4f}")
            print(f"  p值: {p_value:.4e}")
            print(f"  Cohen's d: {cohens_d:.4f}")
            print(f"  相对改进: {relative_improvement:.2f}%")
            print(f"  显著性: {'✓ 显著 (p<0.05)' if p_value < 0.05 else '✗ 不显著'}")
        
        self.results['statistical_tests'] = statistical_tests
        return statistical_tests
    
    def save_results(self):
        """保存结果"""
        output_file = self.output_dir / f"proper_experiment_results_{self.timestamp}.json"
        
        # 转换numpy类型
        def convert_np_types(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_np_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_np_types(item) for item in obj]
            return obj
        
        results_to_save = convert_np_types(self.results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 结果已保存: {output_file}")
        return output_file
    
    def generate_summary(self) -> str:
        """生成总结"""
        consensus_results = self.results.get('consensus', {})
        
        print("\n" + "="*80)
        print("实验结果总结")
        print("="*80)
        
        # 表格
        print(f"\n{'方法':<20} {'相似度':>12} {'标准差':>12} {'95% CI':>25}")
        print("-" * 80)
        
        methods = ['ChatEval', 'NamingGame', 'LeaderFollowing', 'Proposed']
        
        for method in methods:
            if method in consensus_results:
                result = consensus_results[method]
                if result.get('n', 0) > 0:
                    mean = result['mean']
                    std = result['std']
                    ci_low = result['ci_low']
                    ci_high = result['ci_high']
                    print(f"{method:<20} {mean:>12.4f} {std:>12.4f} [{ci_low:.4f}, {ci_high:.4f}]")
        
        # 提升幅度
        proposed_mean = consensus_results.get('Proposed', {}).get('mean', 0)
        
        print("\n" + "-" * 80)
        print("Proposed 相对于 Baseline 的改进：")
        print("-" * 80)
        
        for baseline in ['ChatEval', 'NamingGame', 'LeaderFollowing']:
            if baseline in consensus_results:
                baseline_mean = consensus_results[baseline].get('mean', 0)
                if baseline_mean > 0:
                    improvement = (proposed_mean - baseline_mean) / baseline_mean * 100
                    print(f"{baseline}: {improvement:>+.2f}%")
    
    def run_full_experiment(self, num_tasks: int = 21, num_runs: int = 20):
        """运行完整实验"""
        print("\n" + "="*100)
        print(f"基于MAS ConsensusEngine的完整实验 (时间: {self.timestamp})")
        print("="*100)
        
        try:
            # 运行实验
            consensus_results = self.run_consensus_experiments(num_tasks, num_runs)
            
            # 统计检验
            statistical_tests = self.run_statistical_tests(consensus_results)
            
            # 生成总结
            self.generate_summary()
            
            # 保存
            self.save_results()
            
            print("\n" + "="*100)
            print("✓ 实验完成！")
            print("="*100)
            
            return {
                'consensus_results': consensus_results,
                'statistical_tests': statistical_tests
            }
        
        except Exception as e:
            print(f"\n✗ 实验失败: {e}")
            traceback.print_exc()
            return None


def main():
    """主函数"""
    runner = ProperExperimentRunner(output_dir='./experiments/results')
    results = runner.run_full_experiment(num_tasks=21, num_runs=20)
    return results


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        traceback.print_exc()
        sys.exit(1)
