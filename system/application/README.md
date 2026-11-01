# 映知：影视搜索与协作推荐系统

统一应用入口，独立于实验目录。默认使用已配置的低价 Qwen Flash（关闭思考）；未配置时可运行本地目录模式。旧系统源码保留。

## 启动与验收

```bash
bash scripts/run_system.sh start     # http://127.0.0.1:8010
bash scripts/run_system.sh lan       # 手机扫码需同一局域网，监听0.0.0.0
bash scripts/run_system.sh test
```

Python默认`.venv-native-benchmark/bin/python`；可用MAS_PYTHON覆盖。前端沿用已有React/Vite依赖，第一次缺依赖时在system/frontend/frontend运行npm install。PORT可覆盖端口。

配置读取根目录`.env`/`.env.local`与进程环境。Qwen键名MAS_DASHSCOPE_API_KEY，GLM键名MAS_GLM_API_KEY。密钥不会发给浏览器。模型单价取项目现有配置，费用是估算值。界面可选本地、Qwen Flash、GLM Flash；每次最多3次核查、2元。默认3次、0.1元。

## 研究接口

`EngineRequest` → `Engine.run(request, emit, checkpoint, cancelled)` → `mas.engine.result.v1`。

- request：当前需求、已解析约束、当前用户偏好快照、固定候选与来源、方法、模型、预算。
- emit：阶段、说明、结构化细节。用于即时进度、成员意见、预算分配、局部修改与停止。
- checkpoint：回执读写；发送前写intent，成功后写complete，服务拒绝写rejected。
- cancelled：取消标记。已发出的请求等待返回并保存回执，然后停止后续行动。
- result：影片、来源、初始/最终排序、成员报告、局部变化、资源用量、引擎能力和版本。

应用层不依赖贡献模型特征、博弈求解器内部变量、训练文件或实验gold。变动研究内部实现时保持契约兼容；新增模态/交互需求才扩展带版本的契约。

默认`CatalogEvidenceEngine`是应用级目录证据适配：目录/主题标签/偏好检索，有限预算下的启发式核查价值、边际贡献势函数选择，模型选择证据片段ID，程序绑定原文后局部调分。它没有假称加载Climate-FEVER的训练模型；影视排序和QA分类的响应空间不同，不能直接混用。其game与central使用同一可行域和势函数；UI比较是系统方法展示，不是论文效果评测。关系语义支持性仍由模型判断，来源绑定不是事实真实性认证。

切换原系统真实AEIC工作流（仅提供原流程对应的fixed方法）：

```bash
MAS_ENGINE_PLUGIN=system.application.legacy_adapter:LegacyAEICAdapter PORT=8011 bash scripts/run_system.sh start
```

接入后续研究版本：实现同一接口，在服务端设置`MAS_ENGINE_PLUGIN=your.module:YourEngine`。引擎能力决定界面可用方法，界面不导入研究模块。恢复任务会检查引擎ID；引擎实现有变化时必须升级ID。

## 功能与边界

- 搜索推荐：18,430条现有MovieLens目录，关键词、主题标签、偏好和基于原片标签的相似推荐。必选/排除/年份是硬过滤，偏好是软加分。中文常用片名通过本地别名表映射。
- 多轮：持久会话、约束继承、短追问、换一批、清除条件、用户反馈、收藏与已看/不喜欢排除。
- 详情：真实目录评分及其5分制、标签、IMDb链接与来源。无海报用明确标识的文字封面；无简介明确显示未提供。
- 比较：最多4部影片横向比较；方法对比固定原任务候选/偏好/模型/预算快照，只改变分配方法。
- 语音：浏览器SpeechRecognition输入与speechSynthesis播报。依赖浏览器支持及麦克风权限；手机局域网HTTP通常不能使用麦克风，文字/图片仍可用。
- 图片：macOS Vision本地OCR，提取海报/截图文字，用户可修正后检索。没有文字的画面需补描述，不声称已实现通用画面识别。上传限5MB，临时文件用后删除。
- 扫码：15分钟、限单会话的访问令牌，手机和桌面通过该会话同步；不使用旧全局latest_result。桌面与手机需访问同一个LAN服务。
- 运行：任务进度、取消、回执恢复、导出、费用、诊断。unknown请求不会自动重发。已完成模型回执可复用；重启将正在执行的任务标为interrupted。
- 研究展示：实时协作与封存实验回放分开，回放沿用已有封存校验器。
- 管理：首个本地工作空间具备目录导入权限。JSON导入写入output/system_app/catalog.jsonl副本，不修改实验输入。匿名工作空间令牌保存在浏览器本地；本系统是本地/局域网毕设应用，未设计公网账号服务。

## 存储

`output/system_app/application.sqlite`：profile/session/task，SQLite WAL；原始模型回执只保存在服务端。`MAS_SYSTEM_DATA`可改数据目录，测试使用临时目录。目录版本写入任务，恢复使用原候选快照。

## 验收顺序

1. 输入“推荐温暖的科幻电影，不要恐怖”，检查结果、约束和工作台核查事件。
2. 输入“2010年以后，换一批”，检查年份及已返回影片排除。
3. 收藏、标记看过，再推荐；刷新页面后查看历史与收藏。
4. 选两部影片比较，查看来源和评分尺度。
5. 上传包含片名的图片，修改识别文字并提交检索。
6. LAN模式扫码，从手机发送文字，桌面对话中出现同一任务。
7. 运行时取消，等待当前请求结算后恢复，检查成功回执没有再次发送。
8. 同状态运行另一方法，工作台并列比较结果与成本。
9. 查看封存实验回放、导出任务JSON，管理页导入一条带来源的影片记录。
10. 使用上面的MAS_ENGINE_PLUGIN切换原AEIC适配器，验证页面保持兼容。
