"""Reuse object primitives without executing experiments."""
import redraw_framework as p
from build_figures import Figure, BLUE, TEAL, AMBER, GRAY
p.f=Figure('证据贡献博弈与行动协调',760)
f=p.f
# Shared original vector symbols and typography.
f.parts.extend(p.__dict__.get('_unused', []))
svg=(p.__file__)
from pathlib import Path
source=(Path(svg).parent/'fig02-method-framework.svg').read_text()
f.parts.append(source[source.index('<style>.small'):source.index('</defs>',source.index('<style>.small'))+7])
stage,txt,rect,use,node,line=p.stage,p.txt,p.rect,p.use,p.node,p.line
stage(20,20,410,440,'① 成员条件行动与资料归属',BLUE)
stage(450,20,450,440,'② 集合价值与共享可行域',TEAL)
stage(920,20,460,440,'③ 单成员替换与固定状态停止',AMBER)
txt(40,84,'固定状态 sₜ · 固定主张 c · 剩余预算')
for y,i,labels in [(137,1,['∅','{A}','{A,B}']),(230,2,['∅','{A}','{C}']),(323,3,['∅','{B}','{A,B}'])]:
 use('agent',65,y,BLUE,.65);txt(48,y+41,'i'+str(i))
 for x,label in zip([114,200,290],labels):
  rect(x,y-22,72 if x<290 else 112,43,'#f4f7fa',BLUE);txt(x+13,y+5,label)
 line(86,y,112,y,BLUE)
txt(41,411,'同一来源可由不同成员核查')
txt(41,438,'成员能力不同 → 条件价值不同',BLUE)
for x,label in [(505,'A'),(650,'B'),(800,'C')]:node(x,120,label,TEAL,23)
txt(478,175,'i₁:A');txt(568,175,'i₂:A');line(508,150,505,144,TEAL);line(595,151,523,132,TEAL)
txt(478,210,'重复来源：按成员条件贡献取最大值')
rect(474,232,402,83,'#f0f7f5',TEAL)
txt(487,260,'同一核查者：{A,B}',TEAL)
txt(487,290,'单份贡献 + 有符号交互 − 预计成本')
txt(477,348,'Φ(a) = V̂(来源并集, 成员归属) − Ĉ(a)')
for x,label in [(477,'调用'),(612,'唯一资料'),(757,'费用')]:
 txt(x,386,label,TEAL)
 for j in range(3):rect(x+j*31,399,23,17,'#cce3dc' if j<2 else 'white',TEAL)
txt(478,447,'只允许满足三类共享约束的配置')
txt(943,84,'其他成员不变，仅替换成员 i 的行动')
for y,label,fill in [(120,'当前 aᵢ = {A}','#f5f7fa'),(209,'候选 aᵢ′ = {A,B}','#fff5e6')]:
 use('agent',969,y+17,BLUE,.65);rect(1010,y-4,341,56,fill,AMBER);txt(1024,y+29,label)
f.arrow([(1180,173),(1180,201)])
txt(944,305,'uᵢ(a) = Φ(a) − Φ(a₋ᵢ, ∅)')
txt(944,343,'Δuᵢ = ΔΦ > ε',AMBER)
f.arrow([(1180,264),(1180,280)])
rect(942,369,409,65,'#fff7eb',AMBER)
txt(955,394,'接受严格改善的可行替换')
txt(955,420,'无可行正偏离 → 当前状态停止')
stage(20,490,1360,232,'④ 执行一项核查，随后重新规划',GRAY)
use('agent',70,579,BLUE,.8);txt(41,623,'选中成员')
p.doc(145,545,'A');p.doc(216,545,'B');f.arrow([(282,580),(338,580)])
rect(350,545,240,73,'#eff6f3',TEAL);txt(365,573,'定位引文 → findings',TEAL);txt(365,603,'目标主张 × 来源 × 版本')
f.arrow([(590,580),(640,580)])
node(685,580,'c′',TEAL,26);txt(646,637,'局部状态补丁')
f.arrow([(719,580),(772,580)])
rect(785,545,240,73,'#f5f7fa',BLUE);txt(800,573,'sₜ → sₜ₊₁',BLUE);txt(800,603,'预算扣减 · 价值重估')
f.arrow([(1025,580),(1080,580)])
txt(1100,569,'继续：重新分配',AMBER);txt(1100,609,'停止：共同决策',TEAL)
txt(40,693,'固定状态下的可行单边稳定 ≠ 全局最优；更新状态后重新求解')
f.save('fig06-game-coordination')
