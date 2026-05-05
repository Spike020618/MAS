"""
完整的实验运行框架
======================================================================

支持：
  • 多种基线方法的对比（ChatEval, NamingGame, LeaderFollowing, Proposed）
  • 统计显著性检验（t-test, Mann-Whitney U）
  • 效果大小计算（Cohen's d, effect size）
  • 置信区间计算
  • 详细的实验日志
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
from scipy.stats import ttest_ind, mannwhitneyu
import sys
import warnings

# 导入baseline方法
from baselines import get_all_baselines
from dataset_generator import DatasetGenerator


class ExperimentRunner:
    """完整的实验运行和分析框架"""
    
    def __init__(self, output_dir: str = './experiments/results', verbose: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.verbose = verbose
    
    def _log(self, msg: str, level: str = 'INFO'):
        """条件性打印日志"""
        if self.verbose:
            prefix = f"[{level}]" if level != "INFO" else ""
            print(f"{prefix} {msg}")
    
    def run_consensus_experiments(
        self, 
        num_tasks: int = 21, 
        num_runs: int = 20,
        num_nodes_per_task: int = None
    ) -> Dict[str, Any]:
        """
        运行多节点语义共识实验
        
        Args:
            num_tasks:         任务数
            num_runs:          每个任务的运行次数
            num_nodes_per_task: 每个任务的节点数
        
        Returns:
            所有方法的实验结果
        """
        self._log("\n" + "="*88)
        self._log(f"开始数据生成和实验运行 (时间戳: {self.timestamp})")
        self._log("="*88)
        
        # 生成数据集
        self._log(f"\n[数据生成] 创建 {num_tasks} 个任务数据集...")
        dataset = DatasetGenerator.create_simple_dataset(
            num_tasks, 
            num_nodes_per_task or 3
        )
        self._log(f"✓ 数据生成完成，共 {len(dataset)} 个任务")
        
        # 获取所有方法
        methods = get_all_baselines()
        self._log(f"\n[方法列表] 将对比 {len(methods)} 种方法:")
        for name in methods.keys():
            self._log(f"  • {name}")
        
        # 运行每个方法的实验
        consensus_results = {}
        self._log("\n" + "="*88)
        self._log("运行实验")
        self._log("="*88)
        
        for method_name, method in methods.items():
            self._log(f"\n[运行中] {method_name}...", level="RUN")
            try:
                result = method.run_experiment(dataset, num_runs=num_runs)
                consensus_results[method_name] = result
                
                # 打印中间结果
                self._log(f"  ✓ 完成")
                self._log(f"    平均相似度: {result['mean']:.4f} ± {result['std']:.4f}")
                self._log(f"    95% CI:     [{result['ci_low']:.4f}, {result['ci_high']:.4f}]")
                self._log(f"    样本数:     {result['n']}")
            except Exception as e:
                self._log(f"  ✗ 失败: {e}", level="ERROR")
                warnings.warn(f"{method_name} 执行失败: {e}")
        
        self.results['consensus'] = consensus_results
        return consensus_results
    
    def run_statistical_tests(self, consensus_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        进行统计显著性检验和效果大小计算
        
        Args:
            consensus_results: 共识实验结果
        
        Returns:
            统计检验结果
        """
        self._log("\n" + "="*88)
        self._log("统计显著性检验")
        self._log("="*88)
        
        if 'Proposed' not in consensus_results:
            self._log("✗ 未发现 Proposed 方法结果", level="ERROR")
            return {}
        
        proposed_results = np.array(consensus_results['Proposed']['results'])
        statistical_tests = {}
        
        baselines = ['ChatEval', 'NamingGame', 'LeaderFollowing']
        
        for baseline_name in baselines:
            if baseline_name not in consensus_results:
                continue
                
            baseline_results = np.array(consensus_results[baseline_name]['results'])
            
            # t-test (parametric)
            t_stat, p_value_t = ttest_ind(proposed_results, baseline_results)
            
            # Mann-Whitney U test (non-parametric)
            u_stat, p_value_u = mannwhitneyu(proposed_results, baseline_results)
            
            # Cohen's d (效果大小)
            mean_diff = np.mean(proposed_results) - np.mean(baseline_results)
            pooled_std = np.sqrt((np.std(proposed_results, ddof=1)**2 + 
                                 np.std(baseline_results, ddof=1)**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            # 相对改进百分比
            baseline_mean = np.mean(baseline_results)
            relative_improvement = (mean_diff / baseline_mean * 100) if baseline_mean > 0 else 0
            
            statistical_tests[baseline_name] = {
                't_statistic': round(float(t_stat), 4),
                'p_value_ttest': float(p_value_t),
                'u_statistic': round(float(u_stat), 4),
                'p_value_mannwhitneyu': float(p_value_u),
                'cohens_d': round(float(cohens_d), 4),
                'relative_improvement_percent': round(float(relative_improvement), 2),
                'is_significant_ttest': p_value_t < 0.05,
                'is_significant_mwu': p_value_u < 0.05,
            }
            
            # 打印详细结果
            sig_marker = "✓" if (p_value_t < 0.05 or p_value_u < 0.05) else "✗"
            self._log(f"\n[{sig_marker}] Proposed vs {baseline_name}")
            self._log(f"    t-test:       t={t_stat:.4f}, p={p_value_t:.4f}")
            self._log(f"    Mann-Whitney: U={u_stat:.4f}, p={p_value_u:.4f}")
            self._log(f"    Cohen's d:    {cohens_d:.4f}")
            self._log(f"    相对改进:     {relative_improvement:+.2f}%")
        
        self.results['statistical_tests'] = statistical_tests
        return statistical_tests
    
    def generate_summary_table(self) -> str:
        """
        生成总结表格，格式为能直接插入论文的 LaTeX
        """
        consensus_results = self.results.get('consensus', {})
        
        if not consensus_results:
            return "无可用的共识实验结果"
        
        methods = ['ChatEval', 'NamingGame', 'LeaderFollowing', 'Proposed']
        
        self._log("\n" + "="*88)
        self._log("实验结果总结表")
        self._log("="*88 + "\n")
        
        # 打印格式化的表格到控制台
        print(f"{'方法':<20} {'相似度':>12} {'标准差':>12} {'95% CI':>30}")
        print("-" * 88)
        
        for method in methods:
            if method in consensus_results:
                result = consensus_results[method]
                mean = result['mean']
                std = result['std']
                ci_low = result['ci_low']
                ci_high = result['ci_high']
                
                print(f"{method:<20} {mean:>12.4f} {std:>12.4f} [{ci_low:.4f}, {ci_high:.4f}]")
        
        # 计算相对性能
        proposed_mean = consensus_results.get('Proposed', {}).get('mean', 0)
        
        print("\n" + "-" * 88)
        print("相对于 Proposed 的改进：")
        print("-" * 88)
        
        for baseline in ['ChatEval', 'NamingGame', 'LeaderFollowing']:
            if baseline in consensus_results:
                baseline_mean = consensus_results[baseline]['mean']
                improvement = ((proposed_mean - baseline_mean) / baseline_mean * 100) \
                    if baseline_mean > 0 else 0
                direction = "↑" if improvement > 0 else "↓" if improvement < 0 else "="
                print(f"{baseline:<20} {improvement:>+8.2f}% {direction}")
        
        return ""
    
    def save_results(self) -> Path:
        """保存实验结果到 JSON 文件"""
        output_file = self.output_dir / f"experiment_results_{self.timestamp}.json"
        
        # 转换 numpy 类型为 Python 原生类型
        def convert_np_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
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
        
        self._log(f"\n✓ 结果已保存: {output_file}")
        return output_file
    
    def run_full_experiment(
        self, 
        num_tasks: int = 21, 
        num_runs: int = 20,
        num_nodes_per_task: int = 3
    ) -> Dict[str, Any]:
        """
        运行完整的实验流程
        
        Args:
            num_tasks:         任务数
            num_runs:          每个任务的运行次数  
            num_nodes_per_task: 每个任务的节点数
        
        Returns:
            最终结果字典
        """
        print("\n" + "="*100)
        print(f"🧪 多智能体语义共识对比实验流水线")
        print("="*100)
        
        # 第一步：运行共识实验
        consensus_results = self.run_consensus_experiments(
            num_tasks=num_tasks, 
            num_runs=num_runs,
            num_nodes_per_task=num_nodes_per_task
        )
        
        # 第二步：進行统计检验
        statistical_tests = self.run_statistical_tests(consensus_results)
        
        # 第三步：生成总结表格
        self.generate_summary_table()
        
        # 第四步：保存结果
        self.save_results()
        
        print("\n" + "="*100)
        print("✅ 实验流程完成！")
        print("="*100)
        
        return self.results


if __name__ == '__main__':
    try:
        runner = ExperimentRunner(output_dir='./experiments/results')
        runner.run_full_experiment(num_tasks=21, num_runs=20)
    except Exception as e:
        print(f"\n✗ 实验发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
