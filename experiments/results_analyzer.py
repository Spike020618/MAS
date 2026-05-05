"""
结果分析与可视化 v2
======================================================================

生成论文级别的图表和分析报告：
  • 柱状对比图（with error bars）
  • 置信区间对比图
  • 统计显著性标记
  • LaTeX 表格代码
  • 详细的 Markdown 报告
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import sys

# 尝试导入matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    # 设置字体为支持英文的字体，避免中文字符警告
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial']
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib未安装，将不生成图表文件")

