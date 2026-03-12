# 运行单个实验
./run_mas.sh start exp1 --rounds 20 --agents 5

# 运行所有实验并生成图表
./run_mas.sh start all --plot

# 只生成图表（基于现有结果）
./run_mas.sh plot

# 重新生成数据集
./run_mas.sh regen

# 清理所有结果文件
./run_mas.sh clean

# 查看项目状态
./run_mas.sh status

# 显示帮助
./run_mas.sh help