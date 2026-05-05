#!/usr/bin/env python3
"""
多智能体语义共识对比实验 - 完整流水线
======================================================================

运行流程：
  1️⃣  生成测试数据集
  2️⃣  运行4种方法的对比实验（ChatEval, NamingGame, LeaderFollowing, Proposed）
  3️⃣  进行统计显著性检验
  4️⃣  生成论文级别的可视化和报告
  5️⃣  输出 LaTeX 表格代码

使用: python run_experiments.py
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """主函数 - 按步骤运行完整实验流程"""
    
    print("\n" + "="*100)
    print("🧪 多智能体语义共识对比实验 - 完整流水线")
    print("="*100)
    
    # ========================================================================
    # 第一步：运行完整实验
    # ========================================================================
    print("\n📊 第一步: 数据生成和实验运行...")
    print("-" * 100)
    
    try:
        from experiment_runner import ExperimentRunner
        
        runner = ExperimentRunner(
            output_dir='./experiments/results',
            verbose=True
        )
        results = runner.run_full_experiment(
            num_tasks=21, 
            num_runs=20,
            num_nodes_per_task=3
        )
        
        print("\n✅ 第一步完成：实验运行成功")
        
    except Exception as e:
        print(f"\n❌ 第一步失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================================================
    # 第二步：结果分析和可视化
    # ========================================================================
    print("\n📈 第二步: 结果分析和可视化...")
    print("-" * 100)
    
    try:
        from results_analyzer import ResultsAnalyzer
        
        # 找到最新的结果文件
        results_dir = Path('./experiments/results')
        results_files = list(results_dir.glob('experiment_results_*.json'))
        
        if results_files:
            latest_results_file = max(results_files, key=lambda p: p.stat().st_mtime)
            
            analyzer = ResultsAnalyzer(str(latest_results_file))
            
            # 生成 LaTeX 表格
            print("\n📋 生成 LaTeX 表格代码...")
            latex_tables = analyzer.generate_latex_tables()
            
            # 保存 LaTeX 代码到文件
            latex_file = results_dir / 'latex_tables.tex'
            with open(latex_file, 'w', encoding='utf-8') as f:
                f.write(latex_tables)
            print(f"✓ LaTeX 表格已保存到: {latex_file}")
            print("\n可以直接复制以下代码到论文：")
            print("-" * 100)
            print(latex_tables)
            print("-" * 100)
            
            # 生成可视化图表
            print("\n📊 生成可视化图表...")
            analyzer.plot_comparison_bar_chart()
            analyzer.plot_ci_plot()
            
            # 生成分析报告
            print("\n📝 生成分析报告...")
            analyzer.save_analysis_report()
            
            print("\n✅ 第二步完成：分析和可视化成功")
        else:
            print("⚠️  警告: 未找到实验结果文件")
            return False
            
    except ImportError as e:
        print(f"\n⚠️  警告: {e}")
        print("  (这可能是因为matplotlib未安装，但不影响数据分析)")
    except Exception as e:
        print(f"\n❌ 第二步失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================================================
    # 完成
    # ========================================================================
    print("\n" + "="*100)
    print("✅ 完整实验流程已完成！")
    print("="*100)
    
    print("\n📂 输出文件位置:")
    print("  • 原始数据:    ./experiments/results/experiment_results_*.json")
    print("  • 对比图表:    ./experiments/results/comparison_bar_chart.png")
    print("  • 置信区间图:  ./experiments/results/ci_plot.png")
    print("  • 分析报告:    ./experiments/results/analysis_report.md")
    print("  • LaTeX 表格:  ./experiments/results/latex_tables.tex")
    
    print("\n📚 下一步:")
    print("  1. 查看 analysis_report.md 了解详细结果")
    print("  2. 将生成的图表插入论文")
    print("  3. 复制 LaTeX 表格代码到论文对应位置")
    print("  4. 根据需要调整实验参数重新运行")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
