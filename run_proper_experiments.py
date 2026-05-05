#!/usr/bin/env python3
"""
快速启动 - 基于真实MAS ConsensusEngine的正确实验

用法：
  python run_proper_experiments.py
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """主函数"""
    
    print("\n" + "="*100)
    print("基于真实 MAS ConsensusEngine 的多智能体语义共识对比实验")
    print("="*100)
    
    print("\n[步骤1] 导入实验框架...")
    try:
        from experiments.proper_experiment_runner import ProperExperimentRunner
        print("  ✓ 成功导入")
    except ImportError as e:
        print(f"  ✗ 导入失败: {e}")
        print("\n提示: 确保在 /Users/spike/code/MAS 目录下运行")
        return False
    
    print("\n[步骤2] 初始化实验...")
    try:
        runner = ProperExperimentRunner(output_dir='./experiments/results')
        print("  ✓ 实验器初始化成功")
    except Exception as e:
        print(f"  ✗ 初始化失败: {e}")
        return False
    
    print("\n[步骤3] 运行实验...")
    print("  参数:")
    print("    - 任务数: 21")
    print("    - 每任务重复运行: 20 次")
    print("    - 总评估数: 21 × 20 × 4方法 = 1680")
    print("  预期时间: 5-15 分钟")
    print("  请耐心等待...\n")
    
    try:
        results = runner.run_full_experiment(num_tasks=21, num_runs=20)
        if results is None:
            print("\n✗ 实验失败")
            return False
        
        print("\n✓ 实验完成！")
        
        # 显示结果位置
        results_dir = Path('./experiments/results')
        results_files = list(results_dir.glob('proper_experiment_results_*.json'))
        
        if results_files:
            latest_file = max(results_files, key=lambda p: p.stat().st_mtime)
            print(f"\n结果已保存到:")
            print(f"  📄 {latest_file}")
        
        return True
    
    except Exception as e:
        print(f"\n✗ 实验过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "="*100)
    print("📊 正确的实验框架 - 基于真实 MAS ConsensusEngine")
    print("="*100)
    
    print("\n关键改进:")
    print("  ✓ 使用真实的 ConsensusEngine（不是自己实现的简化版）")
    print("  ✓ 使用 BM25 相似度计算（不是字符级Jaccard）")
    print("  ✓ 真实的权重配置（不是虚假的权重）")
    print("  ✓ 预期结果更好（0.6-0.85 vs 之前的0.4）")
    
    success = main()
    
    if success:
        print("\n" + "="*100)
        print("🎉 实验成功完成！")
        print("="*100)
        print("\n下一步:")
        print("  1. 查看结果文件: ./experiments/results/proper_experiment_results_*.json")
        print("  2. 阅读说明: cat PROPER_EXPERIMENT_GUIDE.md")
        print("  3. 分析结果: python -m experiments.results_analyzer")
        print("\n详见: PROPER_EXPERIMENT_GUIDE.md")
        sys.exit(0)
    else:
        print("\n" + "="*100)
        print("✗ 实验失败")
        print("="*100)
        print("\n故障排除:")
        print("  1. 检查是否在正确的目录: /Users/spike/code/MAS")
        print("  2. 检查是否有必要的依赖: numpy, scipy")
        print("  3. 查看 PROPER_EXPERIMENT_GUIDE.md 中的故障排除部分")
        sys.exit(1)
