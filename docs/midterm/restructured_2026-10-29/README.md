# 中期报告重构与图示构建

正文风格参考用户提供的 /Users/spike/Downloads/中期报告.pdf，版式沿用 /Users/spike/Downloads/LwMidReport.docx。上一版项目Word保留，采用其已修复的学校模板包作为工作副本，正文全部重写。

content.py为结构化正文；build_figures.py生成七幅HTML/SVG图；render_figures.cjs使用隔离的无头浏览器将本地HTML图转成高分辨率PNG；build_report.py在原模板包中替换正文并嵌入图示。使用Codex内置Python/Node运行。图片生成后再构建Word。

最终交付位于output/midterm/restructured_2026-10-29。检查记录为template_verification.json与delivery_quality.json，所有17页已渲染检查。正文、图示与系统截图均不改变正在运行的实验。

系统新增research_view.py、web/workspace.html及三个HTTP入口，并在原首页增加链接。系统真实搜索/证据展开通过本地确定性目录检查；封存回放由完整生成seal及case hash控制。线上模型推荐仍走原工作流，未将实验回放误称为在线新博弈部署。
