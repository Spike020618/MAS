"""Generate three self-contained paper figure pages. No experiment execution."""
from pathlib import Path
from html import escape

OUT = Path(__file__).resolve().parent
INK, BLUE, TEAL, AMBER, GRAY = '#23384d', '#3976a9', '#308c88', '#b78039', '#778492'


class Figure:
    def __init__(self, name, height=850):
        self.name, self.height = name, height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1400 {height}" role="img" aria-label="{escape(name)}">',
                      '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10" fill="#778492"/></marker></defs>',
                      '<style>text{font-family:"PingFang SC","Microsoft YaHei",sans-serif;fill:#23384d;font-size:17px}.head{font-size:22px;font-weight:600}.small{font-size:15px}.mono{font-family:"Times New Roman","PingFang SC",serif;font-size:19px}</style>']

    def text(self, x, y, label, cls='', color=None):
        self.parts.append(f'<text x="{x}" y="{y}" class="{cls}"'+(f' style="fill:{color}"' if color else '')+f'>{escape(label)}</text>')

    def zone(self, x, y, w, h, title, color=BLUE, dashed=False):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{color}" fill-opacity=".045" stroke="{color}" stroke-opacity=".6"'+(' stroke-dasharray="7 5"' if dashed else '')+'/>')
        self.text(x+20,y+32,title,'head',color)

    def box(self,x,y,w,h,title,rows=(),color=BLUE,dashed=False):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="white" stroke="{color}"'+(' stroke-dasharray="6 4"' if dashed else '')+'/>')
        self.parts.append(f'<rect x="{x}" y="{y}" width="5" height="{h}" fill="{color}"/>')
        self.text(x+15,y+29,title,'head',color)
        for i,row in enumerate(rows):
            self.text(x+15,y+57+i*25,row,'small')

    def arrow(self,points,label=None,lx=None,ly=None,dashed=False):
        path='M'+' L'.join(f'{x},{y}' for x,y in points)
        self.parts.append(f'<path d="{path}" fill="none" stroke="{GRAY}" stroke-width="1.6" marker-end="url(#arrow)"'+(' stroke-dasharray="6 4"' if dashed else '')+'/>')
        if label:self.text(lx,ly,label,'small')

    def legend(self,y,items):
        x=40
        for label,color,dashed in items:
            self.parts.append(f'<path d="M{x},{y-5}h35" stroke="{color}" stroke-width="2"'+(' stroke-dasharray="6 4"' if dashed else '')+'/>')
            self.text(x+44,y,label,'small');x+=300

    def save(self,stem):
        svg='\n'.join(self.parts+['</svg>'])
        (OUT/(stem+'.svg')).write_text(svg)
        page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+escape(self.name)+'</title><style>html,body{margin:0;background:white}main{max-width:1600px;margin:auto}svg{display:block;width:100%;height:auto}@media print{@page{size:landscape;margin:8mm}main{max-width:none}}</style><main>'+svg+'</main></html>'
        (OUT/(stem+'.html')).write_text(page)


def framework():
    f=Figure('多智能体博弈共识方法总体框架',900)
    f.zone(25,20,1350,235,'离线：匹配分支采集与集合价值学习',TEAL)
    f.box(50,75,240,150,'相同初始状态', ['任务 x · 局部报告 r₁,r₂,r₃','指定前提 c · 核查成员 i','原始资料 eₐ,eᵦ'],TEAL)
    for x,title,rows in [(330,'空集 / 单份 / 联合',['∅   {eₐ}   {eᵦ}   {eₐ,eᵦ}','相同目标 · 同一核查者','匹配调用 · 保留失败']), (650,'真实增量标签',['Q(A) − Q(∅)','hₐᵦ = Q(AB) − Q(A)','                  − Q(B) + Q(∅)']), (1010,'冻结的价值模型',['单份收益 ĝ(s,e,i)','同成员交互 ĥ(s,eₐ,eᵦ,i)','训练 / 校准 → 固定参数'])]:
        f.box(x,75,280 if x!=1010 else 335,150,title,rows,TEAL)
    f.arrow([(290,155),(330,155)]);f.arrow([(610,155),(650,155)]);f.arrow([(930,155),(1010,155)])
    f.zone(25,290,1350,530,'在线：局部判断—贡献协调—受限修订',BLUE)
    f.box(50,350,210,150,'任务与资料', ['问题 / 当前请求','候选决策 · 资料目录','来源ID · 正文 · 可见范围'])
    f.box(310,350,270,150,'成员局部判断', ['成员 1：e₁ → 报告 r₁','成员 2：e₂ → 报告 r₂','成员 3：e₃ → 报告 r₃'])
    f.box(630,350,300,150,'公共状态 sₜ', ['主张 c：内容 / 来源 / 版本','支持 · 反对 · 未决','当前答案 / 共同排序'])
    f.box(980,350,365,150,'成员条件集合价值', ['行动：成员 i × 前提 c × 资料 A','单份贡献 · 同核查者互补','同源去重 · 预计行动成本'],TEAL)
    for a,b in [(260,310),(580,630),(930,980)]:f.arrow([(a,425),(b,425)])
    f.arrow([(1180,225),(1180,350)],'冻结参数',1193,282,True)
    f.box(980,580,365,170,'证据贡献博弈分配', ['可选：停止 / 单份 / 联合核查','约束：调用数 · 唯一资料 · 费用','可行最佳响应 → 行动配置','选中动作 → 执行 → 重新规划'],AMBER)
    f.box(630,580,300,170,'核查与局部更新', ['指定 source / claim / 版本','逐字引文 · 目标级 findings','接受补丁 / 拒绝 / 放弃','保留无关位置与旧依据'])
    f.box(310,580,270,170,'共同决策读出', ['问答：集体答案','推荐：共同候选排序','初始判断 → 最终判断'])
    f.box(50,580,210,170,'执行封存', ['来源与状态版本','动作 / 补丁 / 回退','Token · 费用 · 时延','原始回执与复现日志'])
    f.arrow([(1160,500),(1160,580)]);f.arrow([(980,665),(930,665)]);f.arrow([(630,665),(580,665)]);f.arrow([(310,665),(260,665)])
    f.arrow([(780,580),(780,535),(945,535),(945,325),(1160,325),(1160,350)],'更新状态 · 预算递减 · 重估价值',810,555)
    f.legend(865,[('离线学习',TEAL,False),('在线状态与执行',BLUE,False),('预算协调',AMBER,False),('离线→在线参数',GRAY,True)])
    f.save('fig02-method-framework')


def game():
    f=Figure('证据贡献博弈与行动协调',860)
    f.zone(25,20,410,720,'成员—行动—资料',BLUE)
    f.box(50,80,360,105,'固定分配状态 sₜ',['当前前提 c · 已有依据 Eₜ','剩余调用 b · 唯一资料上限 k'])
    for y,i,rows in [(220,1,['a₁：停止 / 核查 {eₐ} / 核查 {eₐ,eᵦ}']), (345,2,['a₂：停止 / 核查 {eₐ} / 核查 {e𝒸}']), (470,3,['a₃：停止 / 核查 {eᵦ} / 核查 {eₐ,eᵦ}'])]:
        f.box(50,y,360,95,f'成员 {i} · 局部报告 r{i}',rows)
    f.box(50,610,360,100,'证据集合 E(a)', ['相同来源 eₐ：合并计值','同成员联合 {eₐ,eᵦ}：估计交互'],TEAL)
    f.zone(470,20,425,720,'共同目标与可行配置',TEAL)
    f.box(495,80,375,140,'集合价值', ['来源并集 → 去重后的单份贡献','同核查者联合 → 有符号交互项','价值估计 − 预计行动成本'],TEAL)
    f.box(495,265,375,140,'可行域', ['核查次数 ≤ b','唯一资料数 ≤ k','行动估价合计 ≤ 剩余费用'],TEAL)
    f.box(495,450,375,145,'成员边际效用', ['共同目标 Φ(sₜ,a)','uᵢ = Φ(a) − Φ(a₋ᵢ,停止)','只比较满足共享约束的偏离'],TEAL)
    f.box(495,630,375,80,'同目标参照',['中央枚举 / 集合贪心 / 成员响应'],GRAY)
    f.zone(930,20,445,720,'可行最佳响应与执行',AMBER)
    for y,title,rows in [(80,'① 当前配置 a',['成员归属 · 选中行动 · 预计价值']), (220,'② 单成员可行替换',['停止 ↔ 单份 ↔ 联合','其他成员行动固定']), (385,'③ 严格改善则接受',['ΔΦ > ε → 替换当前行动','无可行正偏离 → 固定状态停止']), (575,'④ 执行选中核查',['最高预计边际行动先执行','补丁进入 sₜ₊₁ → 剩余预算重规划'])]:
        f.box(955,y,395,110 if y!=575 else 135,title,rows,AMBER)
    f.arrow([(410,650),(450,650),(450,150),(495,150)])
    f.arrow([(680,220),(680,265)]);f.arrow([(680,405),(680,450)]);f.arrow([(680,595),(680,630)])
    f.arrow([(870,145),(955,145)])
    f.arrow([(1150,190),(1150,220)]);f.arrow([(1150,330),(1150,385)]);f.arrow([(1150,495),(1150,575)])
    f.arrow([(1350,440),(1363,440),(1363,205),(1320,205),(1320,220)])
    f.legend(800,[('成员与资料',BLUE,False),('共同目标 / 共享约束',TEAL,False),('可行响应 / 重规划',AMBER,False)])
    f.save('fig06-game-coordination')


def system():
    f=Figure('影视推荐系统总体架构',900)
    layers=[(25,20,1350,145,'交互层',BLUE),(25,200,1350,145,'服务与执行层',BLUE),(25,380,820,235,'现有推荐处理层',TEAL),(885,380,490,235,'研究机制接入边界',AMBER),(25,655,1350,170,'数据与审计层',GRAY)]
    for x,y,w,h,t,c in layers:f.zone(x,y,w,h,t,c)
    for x,w,t,rows in [(50,390,'影视浏览与文本检索',['影片卡片 · 偏好输入 · Top-K结果']), (480,390,'辅助输入',['语音 / 图片入口（侧栏）']), (910,435,'最近一次过程查看',['角色报告 · 候选审计 · Trace回放'])]:f.box(x,70,w,70,t,rows)
    f.box(50,250,390,70,'HTTP / WebSocket服务',['请求接收 · 状态返回 · 结果推送'])
    f.box(480,250,390,70,'请求与执行规划',['SearchRequest · 风险 / 预算 / 角色'])
    f.box(910,250,435,70,'单次任务与轨迹归档',['request_id · SearchResponse · SearchTrace'])
    f.box(50,440,235,145,'真实目录候选检索',['类型 / 年份 / 排除条件','检索候选 → 冻结短名单','所有角色评审同一批候选'],TEAL)
    f.box(320,440,235,145,'角色评审',['检索 / 偏好 / 约束核验','DeepSeek偏好 / 多样性','批评角色 → AEIC报告'],TEAL)
    f.box(590,440,230,145,'门控与候选共识',['硬约束验证','报告覆盖 · 分值分解','共同Top-K → 审计轨迹'],TEAL)
    f.box(910,440,435,145,'集合价值与动态协调（待接入）',['experiments/consensus_mechanism','成员条件收益 · 证据去重 / 互补','局部核查 → 补丁 → 状态重规划'],AMBER,True)
    f.arrow([(285,510),(320,510)]);f.arrow([(555,510),(590,510)])
    f.arrow([(440,285),(480,285)])
    f.arrow([(245,140),(245,250)]);f.arrow([(675,320),(675,355),(165,355),(165,440)])
    f.arrow([(1130,440),(1130,345)],'适配器 / 状态映射',1142,365,True)
    f.arrow([(1130,250),(1130,140)])
    f.arrow([(820,510),(860,510),(860,335),(1080,335),(1080,320)])
    f.box(50,710,390,85,'影片目录与实体映射',['MovieLens目录 · 影片ID · 类型 / 年份'],GRAY)
    f.box(480,710,390,85,'模型访问与规则配置',['DeepSeek API · 本地回退 · 公开合同'],GRAY)
    f.box(910,710,435,85,'可追溯运行记录',['原始报告 · 用量 / 时延 · JSON Trace'],GRAY)
    f.arrow([(165,710),(165,585)]);f.arrow([(675,710),(675,640),(435,640),(435,585)]);f.arrow([(800,585),(800,630),(1130,630),(1130,710)])
    f.legend(870,[('现有系统路径',TEAL,False),('研究模块待接入',AMBER,True),('服务与数据依赖',GRAY,False)])
    f.save('fig08-system-architecture')


if __name__=='__main__':
    import runpy
    runpy.run_path(str(OUT/'redraw_framework.py'), run_name='__main__')
    runpy.run_path(str(OUT/'redraw_game.py'), run_name='__main__')
    runpy.run_path(str(OUT/'redraw_system.py'), run_name='__main__')
