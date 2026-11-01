import sys,json,html
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent));from content import FIG_ORDER,FIG_NUMBER
ROOT=Path(__file__).resolve().parents[3];out=ROOT/'output/midterm/expanded_2026-10-29'
meta={x['id']:x for x in json.loads((out/'figures_manifest.json').read_text())}
meta['fig08_workspace']={'title':'影视检索与证据工作台界面','description':'本地确定性目录演示的实际界面，不代表模型实验效果。'}
style='body{margin:0;background:#f3f5f8;color:#203047;font:16px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}main{max-width:1240px;margin:auto;padding:40px 24px}section{background:white;border:1px solid #dde3eb;padding:24px;margin:28px 0;border-radius:8px}svg,img{width:100%;height:auto}h1{font-size:30px}h2{font-size:21px}p{color:#65758a}nav{display:flex;gap:12px;flex-wrap:wrap}a{color:#426b9a}footer{padding:20px}@media print{header,nav,footer{display:none}section{break-after:page;border:0;margin:0;padding:0}@page{size:A4 landscape;margin:12mm}}'
parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>中期报告配套图册</title><style>'+style+'</style><main><header><h1>博弈驱动的去中心化语义共识机制及在影视推荐中的应用</h1><p>与中期报告一致的图序；SVG为可编辑矢量图，图10为实际界面截图。</p></header><nav>']
parts+=['<a href="#'+f+'">图'+str(FIG_NUMBER[f])+' '+html.escape(meta[f]['title'])+'</a>'for f in FIG_ORDER];parts+=['</nav>']
for f in FIG_ORDER:
 m=meta[f];asset=(out/'figures'/f'{f}.svg');visual=asset.read_text()if asset.exists()else f'<img src="figures/{f}.png" alt="实际应用界面">'
 parts+=['<section id="'+f+'"><h2>图'+str(FIG_NUMBER[f])+' '+m['title']+'</h2>'+visual+'<p>'+m['description']+'</p><a download href="figures/'+f+('.svg'if asset.exists()else'.png')+'">下载原图</a></section>']
parts+=['<footer>来源：项目实际方案、实现与已封存记录，2026-10-29。</footer></main></html>'];(out/'研究图册.html').write_text(''.join(parts))
