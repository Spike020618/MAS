"""Summarize sealed batches without modifying predictions or invoking models."""
import argparse,json
from pathlib import Path
from experiments.consensus_recommendation.common import create
from experiments.evidence_consensus.runner_protocol_v3 import score
from experiments.evidence_consensus.paired_statistics_v3 import paired as paired_statistics
NAMES={'single':'一个Agent依次阅读并判断','independent':'三个Agent独立核查后汇总','fixed':'按固定规则指派核查','score_greedy':'直接按预计净收益选择核查','central':'中央全局优化核查分配','game':'边际贡献博弈分配核查'}
def report(root):
 lines=['# v3完整批结果','', '真实语义支持与用户满意度未由结构合同认证；结果属于覆盖合格的开发任务。',''];combined={};total=0
 for mode in ('heuristic','learned'):
  s=score(root/mode);physical=json.loads((root/mode/'physical_summary.json').read_text());total+=physical['configured_known_cny'];combined[mode]={'summary':s,'physical':physical}
  lines.extend([f'## {"预设贡献估计"if mode=="heuristic"else"学习贡献估计"}','','|具体做法|问答正确/总数|推荐命中/目标数|比独立汇总改善/损害题数|核查调用数|','|---|---:|---:|---:|---:|'])
  for method,a in s['methods'].items():
   paired=a['paired_vs_independent'];lines.append(f"|{NAMES[method]}|{a.get('qa_correct',0)}/{a.get('qa_tasks',0)}|{a.get('rec_hits_at_10',0)}/{a.get('rec_targets',0)}|{paired['improved_tasks']}/{paired['harmed_tasks']}|{a.get('review_calls',0)}|")
  plan=json.loads((root/mode/'plan.json').read_text());cases=[json.loads((root/mode/'cases'/f'case_{i}.json').read_text())for i in range(plan['count'])];same=0
  for c in cases:
   source_sets=[]
   for method in ('game','central'):
    a=c['methods'][method]['allocation'];source_sets.append({a['estimated_actions'][q]['unit']for _,q in a['actions']})
   same+=source_sets[0]==source_sets[1]
  tasks=json.loads((root/mode/'inputs.json').read_text());labels=json.loads((root/mode/'labels.after_generation.json').read_text());comparisons={ref:paired_statistics(tasks,cases,labels,reference=ref)for ref in ('independent','fixed','score_greedy','central')};combined[mode]['paired_game_comparisons']=comparisons
  lines.extend(['','博弈相对独立汇总的配对差异（按原始题目/会话分组重采样）：'])
  for kind,est in comparisons['independent'].items():
   lo,hi=est['cluster_bootstrap_95_percent'];lines.append(f'- {kind}: 差值 {est["difference"]:.3f}，探索性95%区间 [{lo:.3f}, {hi:.3f}]。')
  lines.extend(['',f'博弈与中央所选资料集合相同：{same}/{len(cases)}。',f'本批已知用量配置估价：{physical["configured_known_cny"]:.6f}元；未知计费请求：{len(physical["unknown_intents"])}。',''])
 base_cases={mode:[json.loads(p.read_text())for p in sorted((root/mode/'cases').glob('case_*.json'))]for mode in ('heuristic','learned')};by_mode={mode:{c['task_id']:c for c in rows}for mode,rows in base_cases.items()};controls={m:all(by_mode['heuristic'][tid]['methods'][m]['decision']==by_mode['learned'][tid]['methods'][m]['decision']for tid in by_mode['heuristic'])for m in ('single','independent','fixed')};combined['matched_controls']=controls
 lines.extend(['','跨贡献估计批次的固定对照最终决策一致性：'+json.dumps(controls,ensure_ascii=False)+'。若不一致，需先审查回执复用，不能把批次差异归因于贡献模型。',''])
 excluded=json.loads((root/'inputs/excluded_tasks.json').read_text());lines.extend([f'检索覆盖不足而未进入比较的题数：{len(excluded)}，名单保留在inputs/excluded_tasks.json；不能外推为完整基准成绩。','', '按题配对差异仅为描述性结果，不以小批成绩宣称显著或先进。不因负结果自动改协议、换样本或启动下一批。'])
 acq=root/'acquisition/summary.json'
 if acq.exists():
  training=json.loads(acq.read_text());combined['training']=training;total+=training['physical']['configured_known_cny']
 lines.extend(['',f'上述已保存阶段的已知用量配置估价合计：{total:.6f}元；不包含未运行干扰批，未知计费另报。'])
 create(root/'report.json',combined);path=root/'report.md';text='\n'.join(lines)+'\n'
 if path.exists()and path.read_text()!=text:raise ValueError('report changed; original retained')
 path.write_text(text);print(str(path))
def stress_report(root):
 lines=['# v3固定干扰结果','', '|环境|具体做法|问答正确/总数|推荐命中/目标数|','|---|---|---:|---:|'];data={}
 for environment in ('baseline','missing','redundant','distractor'):
  run=root/'learned'if environment=='baseline'else root/'stress'/environment
  summary=score(run);data[environment]=summary
  for method,a in summary['methods'].items():lines.append(f"|{environment}|{NAMES[method]}|{a.get('qa_correct',0)}/{a.get('qa_tasks',0)}|{a.get('rec_hits_at_10',0)}/{a.get('rec_targets',0)}|")
 lines.extend(['','每个环境的所有方法受到相同干预。词面低匹配不等于语义无关；不能从单批排序推断稳定领先。'])
 create(root/'stress_report.json',data);path=root/'stress_report.md';text='\n'.join(lines)+'\n'
 if path.exists()and path.read_text()!=text:raise ValueError('stress report changed')
 path.write_text(text);print(str(path))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stress-only',action='store_true');a=p.parse_args();stress_report(a.root)if a.stress_only else report(a.root)
