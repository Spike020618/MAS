# 当前方法论文插图

独立HTML仅有图形、标签和图例；无下方解释段落。相邻SVG为矢量版本，preview.png为浏览器检查用快照。图题/图注不嵌入图稿，交论文统一处理。

首批：fig02-method-framework.html、fig06-game-coordination.html、fig08-system-architecture.html。构建命令：`python3 docs/figures/current-method/build_figures.py`。

2026-10-29视觉修订：图2已按对象级论文插图重绘，文档、成员、主张关系、核查分配和局部补丁均用具体图形表达。图2独立生成命令：`python3 docs/figures/current-method/redraw_framework.py`。图6、图8仍是待重绘的旧版，不能视为已完成视觉修订。

对应实现：方法图依据experiments/consensus_mechanism/engine.py与allocation.SetAllocator；图6使用固定前提、成员条件贡献、同核查者交互与可行最佳响应，不表示全局最优或已验证增益。系统图依据system/game_consensus_recsys/workflow.py与http_server.py；研究集合协调模块在实验包内，不虚画成已接入系统。

后续保留原编号：1研究结构、3认知与主张、4核验更新、5匹配分支、7状态演化、9业务交互、10同任务截图、11实验设计。12—14待对应封存结果，禁止虚构曲线和案例；表格与公式另行排版。本轮没有启动实验或读取未完成实验成绩。
