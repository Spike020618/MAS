# 开题对齐扩写版

输入：用户的开题报告、学校LwMidReport.docx模板、参考中期PDF、当前项目实现和已封存研究记录。

生成顺序（项目根目录，使用捆绑Python与Node）：

1. `build_figures.py` 生成SVG及初始HTML。
2. `render_figures.cjs` 使用安装的Chrome渲染SVG为PNG并检查文字边界。
3. 从上版保留实际界面截图fig08_workspace.png。
4. `finalize_gallery.py` 按正文顺序组织11幅图。
5. `build_report.py` 写入学校Word包，并导出Markdown、内容JSON及模板校验。
6. 文档skill的render_docx.py配合tmp/midterm_20261029/fonts.conf渲染，检查全部页。

正文来源在content.py，文献在references.py，修订原则在输出目录开题对齐与修订说明.md。修改正文后重新生成并渲染，不直接覆盖冻结实验文件。此次工作没有执行付费实验。
