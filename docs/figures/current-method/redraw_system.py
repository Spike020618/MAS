"""Nested system plate: interface, processing pipeline, audit inset."""
import redraw_framework as p
from pathlib import Path
from build_figures import Figure, BLUE, TEAL, AMBER, GRAY
p.f=Figure('影视检索推荐系统架构',850)
f=p.f
s=(Path(__file__).parent/'fig02-method-framework.svg').read_text()
f.parts.append(s[s.index('<style>.small'):s.index('</defs>',s.index('<style>.small'))+7])
rect,txt,line,use,raw=p.rect,p.txt,p.line,p.use,p.raw
O='#d77632'
def frame(x,y,w,h,title):
 rect(x,y,w,h,'white',O,True);rect(x,y,w,29,O,O);txt(x+9,y+21,title,'white','stage')
def db(x,y,label):
 raw(f'<path d="M{x} {y}v65c0 16 70 16 70 0v-65" fill="#e2ebd8" stroke="#638744" stroke-width="2"/><ellipse cx="{x+35}" cy="{y}" rx="35" ry="11" fill="#bfd4a7" stroke="#638744" stroke-width="2"/><path d="M{x} {y+22}c0 15 70 15 70 0m-70 22c0 15 70 15 70 0" fill="none" stroke="#638744"/>')
 txt(x-3,y+103,label)
frame(20,20,365,715,'① 用户交互与结果浏览')
# A real interface schematic, not a paragraph describing an interface.
rect(38,69,329,410,'#f5f7fb',BLUE);rect(38,69,329,33,BLUE,BLUE)
txt(52,92,'影视搜索', 'white');rect(51,119,248,34,'white',GRAY);txt(60,142,'自然语言需求 / 显式偏好');rect(307,119,46,34,'#deebe7',TEAL);txt(315,142,'搜')
for x,c in [(54,'#426e80'),(153,'#8f795d'),(252,'#677b95')]:
 rect(x,177,85,124,c,c);raw(f'<path d="M{x+32} 220l26 17-26 17z" fill="white" opacity=".75"/>');txt(x,325,'影片卡片',BLUE)
txt(54,365,'年份 · 类型 · 推荐依据')
rect(52,387,299,35,'white',TEAL);txt(67,411,'查看最近一次检索过程 ↗',TEAL)
rect(52,434,299,28,'white',GRAY);txt(64,454,'侧栏：语音输入 / 图片上传')
f.arrow([(200,479),(200,521)])
rect(45,533,315,169,'#f0f6f4',TEAL);txt(59,558,'过程回放',TEAL,'head')
for y,lab in [(587,'请求与计划'),(622,'角色报告与依据'),(657,'候选审计与最终排序')]:
 raw(f'<circle cx="63" cy="{y-5}" r="4" fill="{TEAL}"/>');txt(79,y,lab)
line(63,586,63,651,TEAL)

frame(408,20,972,378,'② 请求规划、目录检索与并行评审')
rect(428,67,185,94,'#edf2fa',BLUE);txt(439,91,'SearchRequest',BLUE);txt(439,119,'偏好 / 必选 / 排除');txt(439,146,'HTTP · WebSocket')
f.arrow([(613,113),(651,113)])
rect(661,67,176,94,'#edf2fa',BLUE);txt(672,91,'执行计划',BLUE);txt(672,119,'风险 · 预算 · 角色');txt(672,146,'request_id')
f.arrow([(837,113),(875,113)])
db(888,82,'MovieLens目录')
f.arrow([(967,113),(1004,113)])
rect(1014,67,343,94,'#eff5e9','#638744');txt(1027,92,'目录检索 → 短名单冻结','#638744');txt(1027,120,'影片ID | 标题 | 年份 | 类型');txt(1027,147,'所有评审者使用同一候选集')
f.arrow([(1190,161),(1190,187),(901,187),(901,217)])
rect(430,225,927,143,'#f3f5fa',BLUE,True)
for x,title,sub in [(480,'检索 / 偏好 / 约束','本地规则'),(802,'偏好 / 多样性 / 批评','DeepSeek API')]:
 for j in range(3):use('agent',x+j*76,265,BLUE,.65)
 txt(x-20,318,title,BLUE);txt(x+25,349,sub)
f.arrow([(1080,276),(1131,276)])
for j in range(3):p.doc(1144+j*56,242,'r'+str(j+1),TEAL)
txt(1142,350,'候选级AEIC报告',TEAL)

frame(408,422,596,313,'③ 约束门控、候选共识与归档')
rect(425,471,157,161,'#fff7e9',AMBER);txt(436,496,'硬约束门控',AMBER)
for y,lab in [(529,'年份'),(563,'必选类型'),(597,'排除类型')]:txt(439,y,'✓ '+lab)
f.arrow([(582,552),(610,552)])
# Candidate-by-reviewer matrix shows the data actually aggregated.
txt(627,478,'候选 × 评审',TEAL)
for r in range(4):
 txt(621,510+r*29,'m'+str(r+1))
 for c in range(4):rect(660+c*31,493+r*29,24,21,['#c6ddd8','#e4edeb','#97bab0'][(r+c)%3],TEAL)
f.arrow([(790,552),(825,552)])
for j,w in enumerate([139,106,77]):
 rect(837,498+j*41,w,25,'#dae6f0',BLUE);txt(843,516+j*41,str(j+1)+'  候选',BLUE)
txt(831,638,'共同 Top-K',BLUE)
f.arrow([(904,646),(904,675),(594,675),(594,650)])
txt(430,700,'JSON Trace：原始报告 · 分值分解 · 用量 · 时延')
f.arrow([(1180,398),(1180,409),(705,409),(705,422)])
f.arrow([(408,671),(385,671)])

frame(1027,422,353,313,'研究模块接入边界')
rect(1044,472,319,179,'#fbf8f0',AMBER,True)
node=p.node
for x,lab in [(1092,'s'),(1201,'a'),(1311,'s′')]:node(x,523,lab,AMBER)
f.arrow([(1115,523),(1177,523)]);f.arrow([(1224,523),(1287,523)])
txt(1058,579,'成员条件贡献 → 集合协调',AMBER)
txt(1058,611,'定向核查 → 局部状态更新',AMBER)
txt(1047,685,'实验机制包；当前未接入业务链路')
line(1004,583,1044,583,AMBER,True)
txt(35,792,'实线：现有业务数据流',TEAL);txt(410,792,'虚线边框：模块分组 / 接入边界',GRAY)
txt(1030,792,'界面为结构示意，非运行截图',GRAY)
f.save('fig08-system-architecture')
