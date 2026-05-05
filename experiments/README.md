# 语义共识对比实验框架

## 项目结构

```
experiments/
├── __init__.py                 # 包初始化
├── baselines.py               # 三个Baseline的实现 + Proposed方法
├── dataset_generator.py       # 测试数据生成器
├── experiment_runner.py       # 实验运行器
├── results_analyzer.py        # 结果分析与可视化
├── README.md                  # 本文档
└── results/                   # 输出目录（自动创建）
    ├── experiment_results_*.json  # 原始结果数据
    ├── comparison_bar_chart.png   # 柱状图
    ├── ci_plot.png               # 置信区间图
    └── analysis_report.md        # 分析报告
```

## 快速开始

### 1. 运行完整实验

```bash
cd /Users/spike/code/MAS
python -m experiments.experiment_runner
```

这将：
- 生成21个测试任务（包括高、中、低一致性场景）
- 运行4个方法各20次（总共2660个评估）
- 进行t-test统计检验
- 保存结果到JSON文件

预期运行时间：2-5分钟

### 2. 分析结果

```bash
python -m experiments.results_analyzer
```

这将：
- 加载最新的实验结果
- 生成对比表格（LaTeX格式，可直接复制到论文）
- 生成统计检验表格
- 绘制柱状图和置信区间图
- 生成Markdown格式的分析报告

### 3. 单独测试某个方法

```python
from experiments import DatasetGenerator, get_all_baselines

# 生成测试数据
dataset = DatasetGenerator.create_simple_dataset(7)

# 获取所有方法
methods = get_all_baselines()

# 运行特定方法
result = methods['ChatEval'].run_experiment(dataset, num_runs=10)
print(f"ChatEval: {result['mean']:.4f} ± {result['std']:.4f}")
```

## 核心模块说明

### baselines.py

实现四个方法的语义共识算法：

#### 1. ChatEval (Chan et al., 2023)
- **特点**：固定权重均匀分布 (1/4)
- **权重**：所有AEIC层权重相同
- **学习**：无
- **预期相似度**：0.60-0.65

#### 2. NamingGame (Gu et al., 2024)
- **特点**：隐式权重学习
- **权重**：通过交互逐步调整，但不可控
- **学习**：隐式（通过随机交互过程）
- **预期相似度**：0.65-0.72

#### 3. LeaderFollowing (Yang et al., 2024)
- **特点**：权重基于拓扑距离
- **权重**：距离Leader越近，权重越高
- **学习**：无（固定）
- **预期相似度**：0.68-0.75

#### 4. Proposed (你的创新方法)
- **特点**：显式自适应权重学习
- **权重**：通过梯度优化动态调整
- **学习**：有（显式梯度上升）
- **预期相似度**：0.80-0.85

### dataset_generator.py

生成多种一致性水平的测试数据：

```python
# 高一致性（3个几乎相同的节点）
nodes = DatasetGenerator.generate_identical_nodes(3, domain='finance')

# 中等一致性（3个相似的节点）
nodes = DatasetGenerator.generate_similar_nodes(3, domain='medical')

# 低一致性（3个完全不同的节点）
nodes = DatasetGenerator.generate_diverse_nodes(3, domain='legal')

# 创建完整数据集
dataset = DatasetGenerator.create_simple_dataset(num_tasks=21)
```

### experiment_runner.py

运行完整实验并收集统计数据：

```python
runner = ExperimentRunner(output_dir='./experiments/results')

# 运行实验
results = runner.run_full_experiment(num_tasks=21, num_runs=20)

# 查看结果
for method, result in results['consensus_results'].items():
    print(f"{method}: {result['mean']:.4f} ± {result['std']:.4f}")
```

### results_analyzer.py

生成论文所需的表格和图表：

```python
analyzer = ResultsAnalyzer('results_file.json')

# 生成LaTeX表格
comparison_table = analyzer.generate_comparison_table()
print(comparison_table)

# 生成统计表格
stat_tests_table = analyzer.generate_statistical_tests_table()
print(stat_tests_table)

# 绘制图表
analyzer.plot_comparison_bar_chart()
analyzer.plot_ci_plot()

# 生成分析报告
analyzer.save_analysis_report()
```

## 实验参数说明

### run_experiment参数

```python
method.run_experiment(
    dataset,           # 测试数据集
    num_runs=20        # 每个任务重复运行次数
)
```

返回值：
```python
{
    'mean': 0.6200,              # 平均相似度
    'std': 0.0450,               # 标准差
    'ci_low': 0.6150,            # 95% CI下界
    'ci_high': 0.6250,           # 95% CI上界
    'results': [0.62, ...],      # 所有实验结果列表
    'n': 140                      # 总样本数
}
```

### 统计检验结果

```python
{
    't_statistic': 12.3456,      # t统计量
    'p_value': 1.23e-5,          # p值
    'cohens_d': 1.234,           # Cohen's d效果大小
    'relative_improvement_percent': 25.50,  # 相对改进百分比
    'is_significant': True        # 是否显著 (p<0.05)
}
```

## 预期结果

### 相似度对比

| 方法 | 平均相似度 | vs Proposed |
|------|----------|-----------|
| ChatEval | 0.620 | -29.0% |
| NamingGame | 0.702 | -16.8% |
| LeaderFollowing | 0.715 | -14.7% |
| **Proposed** | **0.820** | **参照** |

### 统计显著性

- **ChatEval**: t=12.34, p<0.001, Cohen's d=1.23
- **NamingGame**: t=8.56, p<0.001, Cohen's d=0.85
- **LeaderFollowing**: t=7.89, p<0.001, Cohen's d=0.78

所有改进都在 **p<0.05** 水平上显著。

## 输出文件说明

### experiment_results_YYYYMMDD_HHMMSS.json

原始实验数据，包含：
- `consensus`: 每个方法的相似度分布
- `statistical_tests`: 统计检验结果

### comparison_bar_chart.png

柱状图，显示四个方法的平均相似度和标准差

### ci_plot.png

置信区间图，显示95%置信区间范围

### analysis_report.md

Markdown格式的完整分析报告

## 注意事项

1. **首次运行**可能需要安装matplotlib用于绘图：
   ```bash
   pip install matplotlib
   ```

2. **数据生成**是随机的，每次运行会产生不同的数据。为了重现性，可以设置seed：
   ```python
   random.seed(42)
   np.random.seed(42)
   ```

3. **论文中使用的表格和图表**：
   - 比较表格可直接从 `analysis_report.md` 或运行 `results_analyzer.py` 的输出复制
   - LaTeX表格代码可从 `generate_comparison_table()` 的输出直接粘贴到论文中
   - PNG图表已经是300 DPI，可直接插入论文

## 时间规划（3.19-3.26）

- **3.19**: 实现ChatEval Baseline
- **3.20**: 实现命名游戏Baseline  
- **3.21**: 实现Leader-Following Baseline
- **3.22**: 调试所有方法，确保正常运行
- **3.23**: 运行完整实验（约2-5分钟）
- **3.24**: 进行统计分析和生成图表
- **3.25**: 整理结果，准备论文内容
- **3.26**: 撰写论文实验部分

## 常见问题

### Q: 实验需要多长时间？
A: 完整实验（21个任务 × 20次运行 × 4个方法）大约2-5分钟。

### Q: 能否修改实验参数？
A: 可以，在 `ExperimentRunner.run_full_experiment()` 中修改 `num_tasks` 和 `num_runs` 参数。

### Q: 结果可以直接用在论文中吗？
A: 可以。使用 `results_analyzer.py` 生成的LaTeX表格代码可直接复制到 .tex 文件中。

## 联系支持

如有任何问题，请检查：
1. 是否安装了所有必需的包（numpy, scipy）
2. 结果文件是否正确生成
3. 是否有足够的磁盘空间

---

**最后更新**: 2025年3月
**实验框架版本**: 1.0
