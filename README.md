# MAS Graduation Project

本仓库围绕一条完整证据链组织：**多智能体语义共识创新方法 → 机制实验 → 影视搜索推荐系统验证**。

## 目录

```text
MAS/
├── experiments/                 # 机制实现、实验、数据、结果与测试
│   ├── core/                    # AEIC、共识、路由与机制控制器
│   ├── mechanism/               # follower-response 正式实验
│   ├── results/                 # 冻结结果与分析产物（默认不提交 Git）
│   ├── data/                    # 原始/处理后实验数据（默认不提交 Git）
│   ├── benchmarks/              # 评测集定义
│   ├── scripts/                 # 数据构建与验证脚本
│   ├── tests/                   # 机制相关测试
│   ├── legacy/                  # 保留用于追溯的旧实验代码
│   └── legacy_tools/            # 旧结果解析工具
├── system/                      # 影视搜索推荐系统实现与系统测试
│   ├── game_consensus_recsys/   # 后端、工作流和 API
│   └── frontend/                # 前端原型
├── docs/                        # 论文、汇报和研究说明
│   ├── papers/zh/               # 中文论文（权威中文版）
│   ├── papers/en/               # 英文论文
│   ├── presentations/midterm/   # 中期汇报 HTML PPT
│   ├── research/                # 研究设计、审计、计划与台账
│   ├── references/              # 参考论文与检索快照
│   └── templates/               # LaTeX / Word 模板
├── requirements.txt
└── .env                         # 本地密钥，不提交 Git
```

## 常用命令

```bash
# 机制与系统测试
python3 -m unittest experiments.tests.test_mechanism
python3 -m unittest experiments.tests.test_deepseek_recsys system.test_game_consensus_recsys

# 系统环境检查与演示
./system/game_consensus_recsys/run.sh doctor
./system/game_consensus_recsys/run.sh demo

# 机制冻结结果复算
python3 experiments/mechanism/recompute_frozen.py --help
```

论文入口：

- 中文：`docs/papers/zh/main.tex`
- 英文：`docs/papers/en/main.tex`
- 中期汇报：`docs/presentations/midterm/index.html`

研究结论和系统完成度以 `docs/research/` 下的最新审查与结构文档为准；`experiments/legacy/` 仅用于历史追溯，不作为当前论文主证据。
