"""Object-level SVG framework; no experimental values are invented."""
from build_figures import Figure, BLUE, TEAL, AMBER, GRAY, INK

f = Figure('多智能体证据贡献协调框架', 850)
f.parts.append('''<style>.small{font-size:16px}.head{font-size:21px} .stage{font-size:19px;font-weight:600} text{font-size:18px}</style>
<defs>
<g id="agent" fill="white" stroke="currentColor" stroke-width="2"><rect x="-23" y="-17" width="46" height="34" rx="7"/><path d="M0-17v-10m-8 44v8m16-8v8m-31-32h-7m53 0h-7"/><circle cx="0" cy="-29" r="3"/><circle cx="-9" cy="-3" r="3"/><circle cx="9" cy="-3" r="3"/><path d="M-9 9h18"/></g>
<g id="doc" fill="white" stroke="currentColor" stroke-width="1.7"><path d="M0 0h42l12 13v61H0z"/><path d="M42 0v13h12M9 25h34M9 34h34M9 43h25M9 58h34"/><path d="M8 45h33" stroke-width="7" opacity=".22"/></g>
<g id="lock" fill="white" stroke="currentColor" stroke-width="2"><rect x="0" y="13" width="24" height="21" rx="3"/><path d="M5 13V7a7 7 0 0114 0v6"/><circle cx="12" cy="23" r="2"/></g>
</defs>''')

def raw(s): f.parts.append(s)
def rect(x,y,w,h,fill='white',stroke=GRAY,dash=False):
    raw(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}"'+(' stroke-dasharray="6 4"' if dash else '')+'/>')
def line(x,y,a,b,color=GRAY,dash=False):
    raw(f'<path d="M{x} {y}L{a} {b}" stroke="{color}" fill="none" stroke-width="1.7"'+(' stroke-dasharray="4 3"' if dash else '')+'/>')
def use(kind,x,y,color=BLUE,scale=1):raw(f'<use href="#{kind}" transform="translate({x} {y}) scale({scale})" style="color:{color}"/>')
def txt(x,y,s,color=None,cls='small'): f.text(x,y,s,cls,color)
def stage(x,y,w,h,label,color):
    rect(x,y,w,h,'white',color,True);rect(x,y,w,32,color,color);txt(x+10,y+23,label,'white','stage')
def node(x,y,s,color=BLUE,r=19):
    raw(f'<circle cx="{x}" cy="{y}" r="{r}" fill="white" stroke="{color}" stroke-width="2"/>');txt(x-10,y+6,s,color)
def doc(x,y,s,color=TEAL):use('doc',x,y,color);txt(x+12,y+69,s,color)

stage(20,20,330,470,'① 局部资料与独立判断',BLUE)
stage(370,20,400,470,'② 主张组织与预算协调',AMBER)
stage(790,20,590,470,'③ 定向核查与共识更新',TEAL)

# Persistent concrete objects: source documents, reviewers and linked reports.
txt(40,83,'资料目录',BLUE,'head')
for x,s in [(48,'e₁'),(142,'e₂'),(236,'e₃')]:doc(x,101,s)
for x,s in [(75,'1'),(169,'2'),(263,'3')]:
    f.arrow([(x,182),(x,207)]);use('agent',x,239);txt(x-25,281,'成员 '+s)
    f.arrow([(x,291),(x,315)])
    rect(x-39,326,78,102,'#f2f6fb',BLUE)
    txt(x-28,349,'报告 r'+s,BLUE)
    for yy,lab in [(371,'c'+s),(394,'↳ e'+s)]:txt(x-26,yy,lab)
    line(x-27,410,x+25,410,BLUE)
txt(43,464,'同一模型 · 不同局部可见资料')
f.arrow([(350,356),(390,356)])

# Public graph: distinct claims and cited source nodes.
txt(391,82,'公共状态 sₜ',AMBER,'head')
for x,y,s in [(430,123,'c₁'),(558,123,'c₂'),(685,123,'c₃')]:node(x,y,s,BLUE)
for x,y,s in [(430,210,'e₁'),(558,210,'e₂'),(685,210,'e₃')]:node(x,y,s,TEAL,17)
for x in [430,558,685]:line(x,143,x,192,TEAL)
line(451,123,537,123,AMBER,True);txt(478,112,'冲突',AMBER)
txt(401,253,'固定目标 c₂',AMBER);txt(580,253,'状态版本 vₜ')
line(390,268,750,268)
txt(391,297,'成员条件行动菜单',AMBER)
for x,y,s in [(425,338,'1'),(425,390,'2'),(425,442,'3')]:
    use('agent',x,y,BLUE,.55)
    line(x+20,y,482,y,AMBER)
    for xx,lab in [(485,'∅'),(552,'{eₐ}'),(637,'{eₐ,eᵦ}')]:
        rect(xx,y-16,56 if xx<600 else 100,31,'#fff7eb',AMBER);txt(xx+9,y+6,lab)
f.arrow([(770,350),(805,350)])

# Assignment visibly changes one action, instead of another abstract box.
txt(811,82,'可行最佳响应',TEAL,'head')
for x,lab in [(840,'i₁'),(936,'i₂'),(1032,'i₃')]:
    use('agent',x,125,BLUE,.65);txt(x-9,160,lab)
for x,lab in [(840,'∅'),(936,'eₐ'),(1032,'eᵦ')]:node(x,210,lab,TEAL,22)
line(840,143,840,186,GRAY,True);line(936,143,936,186,TEAL);line(1032,143,1032,186,GRAY,True)
raw('<path d="M952 140Q1000 150 1020 187" fill="none" stroke="#308c88" stroke-width="2"/>')
txt(1110,125,'共享预算',TEAL)
for j in range(4):rect(1110+j*47,145,35,20,'#dceee9' if j<2 else 'white',TEAL)
txt(1107,190,'调用 / 资料 / 费用')
txt(811,252,'同源去重',TEAL);txt(950,252,'联合核查',TEAL);txt(1110,252,'成本约束',TEAL)
line(810,268,1360,268)
doc(820,296,'eₐ');doc(892,296,'eᵦ');f.arrow([(954,332),(993,332)])
use('agent',1025,330,TEAL,.8);txt(997,376,'核查者 i₂')
f.arrow([(1052,332),(1093,332)])
rect(1100,291,255,104,'#f1f7f4',TEAL)
txt(1112,314,'c₂ × 来源 × vₜ',TEAL)
rect(1112,327,229,20,'#d7e9dd',TEAL);txt(1118,343,'“原文定位片段”',TEAL)
txt(1112,380,'findings → 局部补丁')
f.arrow([(1230,395),(1230,419)])
txt(813,457,'c₁ 保留',GRAY);node(974,448,'c₂′',TEAL,21);txt(1032,457,'c₃ 保留',GRAY)
f.arrow([(1110,448),(1203,448)])
txt(1210,454,'共同决策',TEAL)

# Online state feedback is a single explicit return path.
f.arrow([(1230,490),(1230,526),(565,526),(565,490)],'sₜ₊₁ · 预算递减 · 重新分配',747,518)

stage(20,559,1360,246,'④ 离线匹配采集与价值学习（测试前冻结）',GRAY)
txt(40,620,'同一状态 · 目标 · 核查者',BLUE)
use('agent',96,672,BLUE,.7);node(174,672,'c₂',BLUE);f.arrow([(200,672),(250,672)])
for x,lab in [(266,'∅'),(381,'A'),(496,'B'),(611,'AB')]:
    rect(x,627,94,86,'#f5f8fa',GRAY);txt(x+32,649,lab,TEAL)
    for j in range(3):line(x+13,665+j*11,x+79,665+j*11,GRAY)
    txt(x+8,741,'Q('+lab+')')
txt(267,777,'匹配调用；∅ 不获得新资料')
f.arrow([(713,675),(761,675)])
txt(779,646,'单份增量',TEAL);txt(779,674,'Q(A) − Q(∅)')
txt(779,714,'联合交互',TEAL);txt(779,742,'Q(AB) − Q(A) − Q(B) + Q(∅)')
f.arrow([(1094,683),(1143,683)])
rect(1155,627,150,113,'#f2f6fb',BLUE)
txt(1167,651,'特征 × 系数',BLUE)
for row in range(3):
    for col in range(5):rect(1168+col*24,664+row*20,18,14,'#c7dbe9' if (row+col)%3 else '#6f9fbe',BLUE)
use('lock',1320,664,BLUE)
txt(1152,777,'冻结 ĝ、ĥ · 在线只读',BLUE)
f.arrow([(1360,683),(1390,683),(1390,280),(754,280),(754,310)],dashed=True)
f.save('fig02-method-framework')
