"""Restore the agreed report hierarchy and substantive academic exposition."""
import importlib.util,re
from pathlib import Path
from manuscript import BLOCKS as CURRENT,REFS as CURRENT_REFS,SOURCE_REFS
spec=importlib.util.spec_from_file_location('retained_narrative',Path(__file__).with_name('manuscript_before_revision.py'))
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
def locate(blocks,title):return next(i for i,(k,t) in enumerate(blocks) if k.startswith('h') and t==title)
def normalize(blocks,refs):
 def sub(m):return '['+','.join('@'+refs[int(i)-1][0] for i in m.group(1).split(','))+']'
 return [(k,re.sub(r'\[([0-9,]+)\]',sub,t) if isinstance(t,str) else t) for k,t in blocks if k!='ref']
c=normalize(CURRENT,CURRENT_REFS);o=normalize(old.BLOCKS,SOURCE_REFS)
start=locate(c,'二、论文进展情况');end=locate(c,'3. 影视推荐系统设计与实现')
intro_end=locate(o,'二、论文进展情况')
BLOCKS=o[:intro_end]+c[start:end]
# Restore the agreed hierarchy. A system implementation is an engineering
# content, not a second algorithmic innovation.
titles={
'1. 多智能体语义共识方法':('h2','研究内容一：基于证据贡献博弈的多智能体语义共识方法'),
'1.1 任务与公共状态':('h3','（一）问题描述与方法总体框架'),
'1.2 来源身份与核查接口':('h3','（二）面向语义共识的主张与证据表示'),
'1.3 受限更新与决策关联':('h3','（三）证据核验驱动的局部共识更新'),
'1.4 匹配贡献学习':('h3','（四）证据贡献估计与联合价值建模'),
'1.5 集合价值与预算博弈':('h3','（五）预算约束下的多智能体博弈协调'),
'1.6 求解与动态执行':('h4','（1）可行最佳响应与动态重规划'),
'2. 实验设计与阶段性证据':('h3','（六）实验设计与结果分析'),
'2.1 研究问题与对照关系':('h4','（1）实验目的与研究问题'),
'2.2 数据与信息隔离':('h4','（2）实验数据集与预处理'),
'2.3 比较方法与资源控制':('h4','（3）对比方法与实验设置'),
'2.4 指标与统计方法':('h4','（4）评价指标与统计方法'),
'2.5 阶段性结果及归因':('h4','（5）整体性能与行动收益分析'),
'2.6 组件分析与有效性边界':('h4','（6）消融与鲁棒性分析')}
BLOCKS=[titles.get(t,(k,t)) if k.startswith('h') else (k,t) for k,t in BLOCKS]
# The retained system chapter supplies requirements, module responsibilities,
# object interfaces and UI execution, instead of reducing these to audit notes.
s=locate(o,'研究内容二：博弈驱动的多智能体影视推荐系统的设计与实现')
e=locate(o,'四、计划及进度安排')
BLOCKS+=o[s:e]
BLOCKS+=c[locate(c,'四、计划及进度安排'):]
# Preserve the current implementation facts while expanding the exposition.
fixes={
'状态版本用于检测核查是否针对旧状态，不能用一个旧目标的结果覆盖后来已经改变的决策。':'完整状态指纹用于核对核查输入身份；决策版本仅在共同决策实际改变时递增。',
'论文写作按研究问题、方法设计、理论分析、实验评价和系统实现组织。':'毕业论文按研究问题、方法设计、理论分析、实验评价和系统实现组织；研究论文集中讨论共识方法与实验。',
'现有工作流与新增回放相关测试共17项通过，支持对应程序行为的实现。':'现有工作流与回放测试覆盖输入、候选、状态及只读接口等程序行为。',
'两个Ridge回归器分别拟合单份增量与交互，正则系数为10；':'单份资料和联合资料承担不同的信息作用：前者补充当前判断缺少的依据，后者可能连接分散的前提。为分别预测这两类作用，本文采用两个Ridge回归器拟合单份增量与交互，正则系数为10；',
'每次接受要么增加势函数，要么在势相等时减少行动数，字典序严格改进，不会重复配置。':'命题一：固定状态下，可行最佳响应过程有限终止。证明：将势函数值与负行动数构成字典序，每次接受的替换使该序严格增加，因而不会重新访问同一配置。',
'有限配置保证有限终止；考虑程序容差，终点不存在超过10⁻¹⁰的可行单边改善。':'可行配置有限，因此接受替换的序列有限。命题二：在终点完整枚举可行偏离后，若δ(a)≤ε，则配置满足ε意义下的可行单边稳定。程序取ε=10⁻¹⁰。',
}
out=[]
for k,t in BLOCKS:
 if isinstance(t,str):
  for a,b in fixes.items():t=t.replace(a,b)
 out.append((k,t))
BLOCKS=out
used=set()
for k,t in BLOCKS:
 if isinstance(t,str):used.update(re.findall(r'@([A-Za-z][A-Za-z0-9_]*)',t))
REFS=[r for r in SOURCE_REFS if r[0] in used]
index={r[0]:i for i,r in enumerate(REFS,1)}
BLOCKS=[(k,re.sub(r'@([A-Za-z][A-Za-z0-9_]*)',lambda m:str(index[m.group(1)]),t) if isinstance(t,str) else t) for k,t in BLOCKS]
for i,(_,text,url) in enumerate(REFS,1):BLOCKS.append(('ref',f'[{i}] {text}'))
