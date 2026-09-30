# 给 Qoder 的系统运行与截图任务

目标：运行当前 AEIC-MD 影视推荐系统，使用真实 MovieLens 目录和 DeepSeek Agent 完成演示用例，生成中期汇报第 23 页需要的 3 张主截图。不要修改论文实验数据，不要把原始 JSON 当作主截图。

## 1. 当前事实

- 项目根目录：`/Users/spike/Downloads/MAS`
- 后端与工作台：`system/game_consensus_recsys/`
- 正式目录：`experiments/data/processed/movie_catalog.jsonl`
- 目录规模：18,430 部电影，统计文件为 `experiments/data/processed/movie_catalog.stats.json`
- 根目录 `.env` 已配置 DeepSeek；环境检查当前识别为 `deepseek-chat`
- 页面入口：`http://127.0.0.1:8000/`
- 健康检查：`http://127.0.0.1:8000/health`
- PPT：`docs/presentations/midterm/index.html`
- 截图目标目录：`docs/presentations/midterm/assets/system-screenshots/`

不要使用 `system/frontend/` 下的旧前端。正式演示页面由 `system/game_consensus_recsys/web/index.html` 提供。

## 2. 启动步骤

在终端执行：

```bash
cd /Users/spike/Downloads/MAS
./system/game_consensus_recsys/run.sh doctor
./system/game_consensus_recsys/run.sh test
./system/game_consensus_recsys/run.sh api
```

第三条会占用当前终端并启动 8000 端口。看到下面的信息后保持进程运行：

```text
Game Consensus API listening on http://127.0.0.1:8000
```

另开终端验证：

```bash
curl http://127.0.0.1:8000/health
```

浏览器打开 `http://127.0.0.1:8000/`。如果端口已占用，先确认是否已有本项目服务，不要直接杀死不明进程。

## 3. 主演示用例：三张截图必须来自同一次运行

页面默认值就是主案例，确认以下输入：

| 字段 | 内容 |
|---|---|
| 自然语言请求 | 推荐2010年以后的思想性科幻剧情片，不要动作片和灾难片 |
| 偏好类型 | 科幻,剧情 |
| 必选类型 | 留空 |
| 排除类型 | 动作,灾难 |
| 最早年份 | 2010 |
| Top-K | 页面固定为 3 |

点击“运行机制”，等待顶部状态变成“完成”。确认不是纯确定性降级运行：页面应出现多个 Agent 报告、行动标签、Token 与延迟。若状态提示 Agent 降级，可保留该次结果作为备份案例，但主截图应重新运行一次，优先选择没有降级且存在真实分歧的结果。

运行完成后不要刷新页面；从这同一次结果中截取以下三张图。

### 截图 1：请求与执行计划

必须同时包含：

- 页面标题和完整输入栏；
- “请求时执行计划”中的风险等级、复杂度、组织方式、验证预算；
- “公开合同已在行动前提交”及角色列表；
- 审计时间线开头若干事件，至少能看到 Planner/合同发生在 Agent 执行之前。

文件名：

```text
docs/presentations/midterm/assets/system-screenshots/01_request_plan.png
```

### 截图 2：多 Agent AEIC 工作台

必须同时包含：

- Preference、Diversity、Critic 等不同角色；
- 每个角色的行动标签；
- A/E/I/C 四格；
- 报告摘要、反对数量、Token 和 latency；
- 尽量选择能看到至少一个反对或角色差异的运行结果。

文件名：

```text
docs/presentations/midterm/assets/system-screenshots/02_agent_aeic.png
```

### 截图 3：Top-K 与候选级共识

必须同时包含：

- 3 个候选电影卡片；
- 检索相关、决策共识、证据共识三个分项；
- 已接受证据；
- 若存在残余异议，必须保留异议区域；
- 不要展开完整机器可读 Trace，避免用大段 JSON 代替解释界面。

文件名：

```text
docs/presentations/midterm/assets/system-screenshots/03_candidate_consensus.png
```

## 4. 备份用例

如果主案例没有明显分歧，再运行下面的多样性冲突案例：

- 请求：`想看节奏紧张的科幻悬疑片，但不要三部都来自同一种叙事风格；兼顾思想性和类型多样性`
- 偏好类型：`科幻,悬疑,惊悚`
- 必选类型：留空
- 排除类型：留空
- 最早年份：留空

优先截取角色分歧和残余异议，保存为：

```text
docs/presentations/midterm/assets/system-screenshots/backup_diversity_conflict.png
```

安全边界案例可用：

- 请求：`推荐1990到1999年的中国历史剧情经典，最多3部；没有足够可靠候选时不要凑数`
- 偏好类型：`剧情,历史`
- 必选类型：`剧情`
- 最早年份：`1990`

当前页面没有“最晚年份”输入框，因此不要声称界面严格执行了 1999 上界；如果要把该案例用于正式证据，应先给页面增加 `max_year` 输入并传入 API。没有修正前只可作为非正式观察。

## 5. 截图规范

- 浏览器建议设置为 1440×900 或更宽，缩放 90%–100%。
- 图片必须是 PNG，不要用手机拍屏，不要带浏览器书签、通知、API Key 或终端密钥。
- 只裁掉浏览器外框和无关空白，不要裁掉模块标题、字段名、状态栏和关键数值。
- 三张主图应保持相近宽高比，建议接近 16:9；文字必须在 PPT 中仍可辨认。
- 不要用 Photoshop 修改运行数值；可以做裁切和统一留白。
- 同时保存本次响应的完整 JSON，便于追溯：在页面展开 Trace 后复制内容，保存到：

```text
docs/presentations/midterm/assets/system-screenshots/case_a_trace.json
```

## 6. 将截图接入 PPT

三张截图生成后，分别放入 `docs/presentations/midterm/index.html` 第 23–25 页三张系统证据页。图片路径使用：

```html
assets/system-screenshots/01_request_plan.png
assets/system-screenshots/02_agent_aeic.png
assets/system-screenshots/03_candidate_consensus.png
```

要求：

- 每张证据图独占一页，不再使用三列缩略图布局；
- 图片使用 `object-fit: contain`，不得拉伸；
- 每张图下面保留一句“这张图证明什么”，不要只剩图片；
- 不改第 22 页系统架构图；
- 不改实验结果与论文结论；
- 完成后在 1440×900 下检查第 23–25 页无溢出、无截断、无小到不可读的文字。

## 7. 完成检查清单

- [ ] `doctor` 显示 DeepSeek configured，且没有暴露 API Key。
- [ ] 使用的是 18,430 部正式 MovieLens 目录，而不是 8 部内置回退目录。
- [ ] 三张主截图来自同一次 Case A 运行。
- [ ] 截图 1 能证明合同先于行动公开。
- [ ] 截图 2 能证明角色、行动、AEIC 和资源消耗真实存在。
- [ ] 截图 3 能证明最终排序保留证据与异议。
- [ ] Trace JSON 已保存。
- [ ] PPT 第 23–25 页已分别使用三张主图，页面无溢出。
