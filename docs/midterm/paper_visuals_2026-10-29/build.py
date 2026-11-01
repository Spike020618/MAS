from pathlib import Path
import html,json
R=Path(__file__).resolve().parents[3];O=R/'output/midterm/paper_visuals_2026-10-29';F=O/'figures'
INK='#222831';GRAY='#77808d';LINE='#c5cad2';BLUE='#507ea7';ORANGE='#ce9356';GREEN='#5d998a';RED='#bd6f75';PURPLE='#8b7daf';PALE='#f6f7f8'
figs=[]
def t(x,y,s,size=24,anchor='middle',fill=INK,weight=400,italic=False):return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}"'+(' font-style="italic" font-family="Times New Roman,serif"'if italic else'')+'>'+html.escape(s)+'</text>'
def path(d,c=GRAY,w=2.2,dash=False,arrow=False):return f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{w}"'+(' stroke-dasharray="7 6"'if dash else'')+(' marker-end="url(#arr)"'if arrow else'')+'/>'
def rect(x,y,w,h,fill='white',stroke=LINE,rx=5,dash=False):return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"'+(' stroke-dasharray="7 6"'if dash else'')+'/>'
def circle(x,y,r,fill='white',stroke=LINE):return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>'
def arr(x1,y1,x2,y2,c=GRAY,dash=False):return path(f'M{x1} {y1} L{x2} {y2}',c,dash=dash,arrow=True)
def panel(x,y,w,h,title,color=BLUE):return rect(x,y,w,h,'white',color,9)+t(x+18,y+34,title,25,'start',INK,600)
def doc(x,y,label,color=BLUE,scale=1):
 b=f'<g transform="translate({x},{y}) scale({scale})">'+rect(0,0,58,72,'white',color,3)+path('M37 0 V18 H58',color)
 for yy in [30,41,52]:b+=path(f'M10 {yy} H46',color,1.6)
 return b+t(29,92,label,23,fill=color,italic=True)+'</g>'
def agent(x,y,label,color=BLUE,scale=1):
 return f'<g transform="translate({x},{y}) scale({scale})">'+path('M0 -36 V-25',color)+circle(0,-40,4,'white',color)+rect(-31,-25,62,43,'white',color,10)+circle(-12,-6,4,color,color)+circle(12,-6,4,color,color)+path('M-12 8 H12',color,2)+path('M-19 20 L-26 43 H26 L19 20',color,2)+t(0,70,label,23,fill=color,italic=True)+'</g>'
def node(x,y,label,color=BLUE,r=24,fill='white'):return circle(x,y,r,fill,color)+t(x,y+8,label,23,fill=color,italic=True)
def badge(x,y,label,color):return circle(x,y,15,'white',color)+t(x,y+6,label,18,fill=color,weight=600)
def listrows(x,y,ids,color=BLUE,highlight=None,w=150):
 b=''
 for i,s in enumerate(ids):
  active=highlight==i;b+=rect(x,y+i*43,w,35,'#edf5f1'if active else'white',GREEN if active else LINE,3)+t(x+18,y+i*43+24,str(i+1),17,'start',GRAY)+t(x+w/2+8,y+i*43+25,s,22,fill=GREEN if active else color,italic=True)
 return b

def save(id,title,caption,b,w=1280,h=540):
 svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc"><title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(caption)}</desc><defs><marker id="arr" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M1 1 L8 4.5 L1 8" fill="none" stroke="{GRAY}" stroke-width="1.4"/></marker></defs><rect width="{w}" height="{h}" fill="white"/><g font-family="Arial,PingFang SC,Microsoft YaHei,sans-serif">{b}</g></svg>'''
 (F/(id+'.svg')).write_text(svg);figs.append(dict(id=id,title=title,caption=caption,width=w,height=h))
# 1 Overview: visual entities and a single feedback loop.
b=''
for x,w,title,c in [(15,290,'(a) 局部认知',BLUE),(335,290,'(b) 贡献估计',PURPLE),(655,290,'(c) 博弈协调',ORANGE),(975,290,'(d) 共识更新',GREEN)]:b+=panel(x,25,w,355,title,c)
for yy,k in [(122,1),(250,2)]:
 b+=doc(40,yy-36,'e'+str(k),BLUE,.65)+agent(138,yy,'i'+str(k),BLUE,.7)+arr(88,yy-8,107,yy-8)+node(245,yy,'c'+str(k),BLUE,22)+arr(163,yy-8,215,yy-8)
b+=t(156,350,'资料 → 成员 → 主张',22)
b+=doc(365,112,'e₁',BLUE,.7)+doc(465,112,'e₂',BLUE,.7)+arr(388,185,410,222)+arr(488,185,470,222)
b+=node(440,248,'ĝ',PURPLE,32)+node(540,248,'ĥ',PURPLE,32)+arr(440,218,440,190,PURPLE,True)
b+=t(480,328,'单份增量 · 联合交互',22)
for x,y,label in [(700,140,'a₁'),(800,115,'a₂'),(890,180,'a₃')]:b+=node(x,y,label,ORANGE,27)
b+=path('M728 135 Q760 95 776 110',ORANGE,arrow=True)+path('M818 135 Q863 132 879 152',ORANGE,arrow=True)+path('M870 197 Q780 235 715 166',ORANGE,arrow=True)
b+=t(800,275,'uᵢ = Φ(a) − Φ(a₋ᵢ,∅)',23,italic=True)+t(800,328,'共享预算  K · B · M',22)
b+=node(1018,155,'cₖ',RED,23)+arr(1044,155,1085,155,GREEN)+node(1120,155,'cₖ′',GREEN,23)+badge(1155,126,'✓',GREEN)
b+=listrows(1020,224,['m₁','m₄','m₃'],GREEN,1,w=192)
for x in [309,629,949]:b+=arr(x,200,x+22,200)
b+=path('M1120 380 V445 H160 V386',GREEN,2.5,arrow=True)+t(640,433,'更新状态 sₜ₊₁ 后重新规划',24,fill=GREEN)
b+=t(640,500,'共同答案  /  共同排序',25,weight=600)
save('fig01_overview','证据贡献博弈共识方法总体框架','成员基于局部资料提出主张，单份与联合贡献用于构造共享预算下的行动博弈。核查结果以局部更新进入共同决策，执行后依据新状态重新规划。图中对象为机制示意。',b,h=530)
# 2 Semantic object graph: distinguish scope, contradiction, update.
b=panel(20,20,360,430,'(a) 来源与范围',BLUE)+panel(410,20,460,430,'(b) 主张—证据关系',PURPLE)+panel(900,20,360,430,'(c) 公共状态',GREEN)
for yy,label,symbol,c in [(115,'当前任务','e₁',BLUE),(235,'历史用户','e₂',ORANGE),(355,'目录事实','e₃',GREEN)]:
 b+=doc(55,yy-32,symbol,c,.7)+t(210,yy+4,label,24)+rect(130,yy+30,160,29,'white',c,14)+t(210,yy+51,'scope',19,fill=c,italic=True)
for x,y,k,c in [(475,140,'e₁',BLUE),(475,345,'e₃',GREEN),(665,140,'c₁',BLUE),(665,345,'c₂',RED),(810,240,'dₜ',PURPLE)]:b+=node(x,y,k,c,29)
b+=arr(506,140,632,140,BLUE)+t(570,121,'支持',20,fill=BLUE)+arr(506,345,632,345,GREEN)+t(570,326,'反驳',20,fill=RED)+path('M665 171 V310',RED,2.2,True)+t(695,245,'冲突',20,fill=RED)+arr(691,153,789,220)+arr(689,328,791,262,GRAY,True)
b+=rect(940,95,280,130,'#f7f9fa',LINE)+t(1080,130,'sₜ',28,italic=True)+t(1080,173,'Cₜ · Vₜ · dₜ',24,italic=True)+t(1080,204,'Hₜ · Aₜ',24,italic=True)
b+=rect(940,285,280,105,'white',GREEN)+t(1080,325,'主张身份不变',24)+t(1080,364,'状态与评估分离',22)
save('fig02_state','主张证据关联与公共状态表示','来源范围区分当前任务、历史用户与目录事实。引用将资料片段关联到主张，核查关系与当前决策分别保存；历史记录保留被反驳主张，不将其重新作为活动依据。',b,h=475)
# 3 Local state transition with rank-boundary operation.
b=panel(20,20,385,395,'(a) 更新前  sₜ',BLUE)+panel(445,20,350,395,'(b) 指定核查',ORANGE)+panel(835,20,425,395,'(c) 更新后  sₜ₊₁',GREEN)
for x,k in [(95,'c₁'),(210,'c₂'),(325,'c₃')]:b+=node(x,112,k,RED if k=='c₂'else BLUE,26)
b+=listrows(55,182,['m₁','m₂','m₃'],BLUE,w=165)+rect(250,268,115,40,'white',LINE)+t(307,295,'m₄',23,italic=True)+t(304,246,'Top10 外',18,fill=GRAY)
b+=agent(620,115,'核查者',ORANGE,.8)+doc(494,225,'eₐ',BLUE,.75)+doc(690,225,'eᵦ',BLUE,.75)+arr(544,245,596,221)+arr(690,245,649,221)
b+=t(620,345,'反驳 c₂',25,fill=RED)+t(620,383,'swap(m₂,m₄)',23,italic=True)
for x,k in [(910,'c₁'),(1045,'c₂'),(1180,'c₃')]:b+=node(x,112,k,RED if k=='c₂'else BLUE,26)
b+=path('M1025 93 L1066 132 M1025 132 L1066 93',RED,2.5)+t(1045,160,'退出活动集',18,fill=RED)
b+=listrows(885,182,['m₁','m₄','m₃'],BLUE,1,w=165)+rect(1100,268,115,40,'white',LINE)+t(1157,295,'m₂',23,italic=True)+t(1157,246,'Top10 外',18,fill=GRAY)
b+=arr(407,215,440,215)+arr(798,215,830,215)
b+=t(640,466,'图中仅画出决策边界的局部片段；其余主张与候选顺序保留。',22,fill=GRAY)
save('fig03_update','证据核验驱动的局部共识更新','指定主张的反驳改变其活动状态，合法补丁在Top10边界内外交换一项。c₁与c₃保持原状态，历史意见和引用仍可追溯。候选与主张身份均为示意符号。',b,h=500)
# 4 Matched acquisition as controlled branching, not rectangle prose.
b=rect(20,20,1240,85,'#f5f6f8',LINE)+t(640,57,'固定 s · c · i',27,italic=True)+t(640,88,'同一初始状态、目标与核查者',22)
for x in [160,480,800,1120]:b+=path(f'M640 105 V145 H{x} V179',GRAY,arrow=True)
for x,label,docs,decision in [(160,'空集',[], 'Y∅'),(480,'单份 A',['eₐ'],'Yₐ'),(800,'单份 B',['eᵦ'],'Yᵦ'),(1120,'联合 AB',['eₐ','eᵦ'],'Yₐᵦ')]:
 b+=t(x,205,label,25,weight=600)
 if not docs:b+=circle(x,263,28,'white',GRAY)+t(x,273,'∅',35,fill=GRAY)
 for j,k in enumerate(docs):b+=doc(x-22+(j-.5*(len(docs)-1))*70,232,k,BLUE,.7)
 b+=agent(x,375,'',GREEN,.7)+arr(x,322,x,342)+t(x,451,decision,28,italic=True)
b+=path('M160 471 V500 H1120 V471',GRAY)+arr(640,500,640,535)
b+=t(640,575,'g(A)=Q(Yₐ)−Q(Y∅)     h(A,B)=Q(Yₐᵦ)−Q(Yₐ)−Q(Yᵦ)+Q(Y∅)',25,italic=True)
save('fig04_matched','匹配分支的贡献学习设计','固定初始条件，仅改变追加资料。空集控制再次思考，单份与联合分支分别构造增量和交互；生成封存后计算质量标签。箭头描述采集依赖，不代表实际效果大小。',b,h=610)
# 5 Set game visually shows overlap, assignment and best response.
b=panel(20,20,400,430,'(a) 证据并集',BLUE)+panel(450,20,370,430,'(b) 行动分配',ORANGE)+panel(850,20,410,430,'(c) 可行最佳响应',GREEN)
b+=f'<ellipse cx="175" cy="220" rx="105" ry="112" fill="#eaf0f7" stroke="{BLUE}" stroke-width="2"/><ellipse cx="280" cy="220" rx="105" ry="112" fill="#f6eee5" fill-opacity=".75" stroke="{ORANGE}" stroke-width="2"/>'
b+=t(130,225,'e₁',29,fill=BLUE,italic=True)+t(227,225,'e₂',29,fill=PURPLE,italic=True)+t(325,225,'e₃',29,fill=ORANGE,italic=True)+t(160,365,'重复来源计一次',24)+t(300,395,'e₁ ⊕ e₃',25,italic=True)
for yy,label,color in [(120,'i₁',BLUE),(240,'i₂',GREEN),(360,'i₃',PURPLE)]:b+=agent(510,yy,label,color,.6)
for yy,label in [(120,'{e₁,e₂}'),(240,'{e₃}'),(360,'∅')]:b+=rect(605,yy-32,175,55,'white',LINE)+t(692,yy+4,label,26,italic=True)+arr(542,yy-6,598,yy-6)
b+=t(1055,115,'a',29,italic=True)+arr(1090,106,1180,106,GREEN)+t(1200,115,'a′',29,italic=True)
b+=rect(897,155,317,120,'#f3f8f5',GREEN)+t(1055,197,'Δuᵢ = ΔΦ',29,italic=True)+t(1055,245,'K · B · M',27,italic=True)
b+=path('M1090 310 C1200 310 1215 390 1118 395',GREEN,arrow=True)+path('M1015 395 C914 395 914 310 1010 310',GREEN,arrow=True)+t(1055,350,'逐成员替代',23)+t(1055,423,'δ ≤ ε',28,italic=True)
save('fig05_game','集合价值与边际贡献协调','证据并集消除重复计值，联合项表达互补或干扰。成员在共享资源下替换动作，边际效用变化等于共同潜势变化；终止后检查最大可行偏离。该图未表示代理均衡必然提高真实任务质量。',b,h=475)
# 6 Lifecycle: compact actual cycle with budget tokens.
b='';centers=[(180,125,'定位',BLUE),(640,125,'估值',PURPLE),(1100,125,'分配',ORANGE),(1100,355,'核查',ORANGE),(640,355,'提交',GREEN),(180,355,'读出',BLUE)]
for x,y,label,c in centers:b+=circle(x,y,54,'white',c)+t(x,y+9,label,27,fill=c,weight=600)
for a,z in [(0,1),(1,2),(2,3),(3,4),(4,5)]:
 x,y,*_=centers[a];xx,yy,*_=centers[z];b+=arr(x+(60 if xx>x else -60 if xx<x else 0),y+(60 if yy>y else 0),xx+(-60 if xx>x else 60 if xx<x else 0),yy+(-60 if yy>y else 0))
b+=t(180,215,'cₖ',26,italic=True)+t(640,215,'ĝᵢ · ĥᵢ',26,italic=True)+t(1100,215,'a*',26,italic=True)+t(1100,445,'r · refs · patch',24,italic=True)+t(640,445,'sₜ → sₜ₊₁',26,italic=True)+t(180,445,'d_T',26,italic=True)
b+=path('M640 295 V260 H180 V184',GREEN,2.4,True,True)+t(425,247,'剩余预算 > 0',22,fill=GREEN)
b+=t(640,512,'停止：预算耗尽 / 无合法行动 / 不再追加',23,fill=GRAY)
save('fig06_execution','预算约束下的在线执行闭环','每次只落实一个核查，提交后按剩余预算回到目标定位与估值。停止后读取最终共同决策，未知或无效输出按统一执行规则记录。',b,h=550)
# 7 experiment isolation: datasets as separate tracks and strategy selector.
b=panel(20,20,365,445,'(a) 学习与冻结',PURPLE)+panel(420,20,840,200,'(b) 端到端比较',BLUE)+panel(420,250,840,215,'(c) 机制诊断',GREEN)
for x,k in [(80,'s'),(180,'A'),(280,'AB')]:b+=doc(x,105,k,PURPLE,.8)
b+=path('M105 220 H305 M205 220 V255',PURPLE,arrow=True)+rect(75,280,255,60,'#f3f0f7',PURPLE)+t(202,318,'收益模型  ĝ · ĥ',25,fill=PURPLE)+t(202,405,'按原题 / 会话隔离',23)
for x,label,c in [(520,'问答',BLUE),(790,'会话推荐',ORANGE),(1060,'多跳问答',GREEN)]:
 b+=doc(x-20,88,'',c,.55)+t(x,173,label,24)
b+=arr(385,310,412,310,PURPLE,True)
for x,label,symbol,c in [(560,'关系核验','✓ / × / ?',GREEN),(840,'匹配动作','∅ · A · B · AB',BLUE),(1115,'消融与预算','K → K+1',ORANGE)]:
 b+=t(x,350,symbol,27,fill=c,italic=True)+t(x,410,label,24)
b+=t(840,525,'生成 → 封存 → 评分',26)+t(203,525,'训练标签',22,fill=GRAY)
save('fig07_experiment','实验分层与数据隔离设计','学习阶段拟合并冻结贡献模型，端到端任务比较共同决策，独立诊断分别考察核验、行动价值和组件。训练与评价按原题或会话隔离，标签在生成封存后用于评分。',b,h=565)
# 8 System: diagrammatic browser, catalog, roles, service boundary.
b=rect(20,70,210,295,'white',BLUE)+rect(20,70,210,35,'#eaf0f7',BLUE)+circle(36,87,3,BLUE,BLUE)+circle(49,87,3,BLUE,BLUE)+circle(62,87,3,BLUE,BLUE)+rect(43,129,160,31,'white',LINE)
for i in range(3):b+=rect(43,182+i*49,40,34,'#edf1f5',LINE)+path(f'M95 {193+i*49} H198 M95 {207+i*49} H172',GRAY,1.7)
b+=t(125,407,'用户工作台',25)
b+=panel(280,20,725,440,'服务与共识处理',GRAY)
b+=rect(315,92,655,55,'#f7f8f9',LINE)+t(640,128,'请求解析与任务规划',25)+arr(640,150,640,180)
b+=doc(335,220,'目录',BLUE,.7)
for x,label,c in [(505,'偏好',BLUE),(650,'批评',PURPLE),(795,'核验',GREEN)]:b+=agent(x,230,label,c,.7)
b+=arr(390,230,458,230)+path('M505 300 V325 H840 M650 300 V325 M795 300 V325',GRAY)+arr(840,325,840,355)
b+=rect(455,360,470,62,'#eef5f1',GREEN)+t(690,400,'约束门控 · 共同排序',25,fill=GREEN)
b+=arr(230,180,275,180)+path('M455 391 H255 V315 H234',GRAY,arrow=True)
b+=rect(1050,92,210,125,'white',GRAY)+t(1155,132,'执行轨迹',25)+t(1155,177,'请求 · 报告 · 结果',19)
b+=rect(1050,295,210,125,'white',ORANGE,5,True)+t(1155,337,'研究核查策略',24)+t(1155,380,'在线适配接口',21,fill=GRAY)
b+=arr(1005,140,1045,140)+arr(1045,358,1009,358,ORANGE,True)
b+=t(640,519,'实线：应用已实现路径     虚线：研究机制接入边界',22,fill=GRAY)
save('fig08_system','影视推荐系统架构与机制接口','工作台经服务层连接目录、并行角色、确定性门控和共同排序，执行记录供追踪与回放。研究核查策略在独立运行器中实现，虚线表示在线适配接口，未将其画为已部署的应用路径。',b,h=550)
(O/'figures.json').write_text(json.dumps(figs,ensure_ascii=False,indent=2))
print('figures',len(figs))
