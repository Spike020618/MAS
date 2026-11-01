"""Export recorded metrics and paired results without manufacturing missing values."""
import csv,json
from pathlib import Path
NAMES={'single_strong':'同信息同选择器单Agent','independent':'独立分工汇总','fixed':'固定核查','greedy':'集合边际收益贪心','central':'中央集合价值最优','game':'边际贡献博弈','self_consistency':'独立采样投票适配','self_refine':'Self-Refine适配','mad':'多Agent讨论适配'}

def report(root):
 root=Path(root);rows=[];metrics=[]
 for p in sorted(root.rglob('*summary.json')):
  if 'source_snapshot' in p.parts or p.name=='physical_summary.json':continue
  summary=json.loads(p.read_text());stage=str(p.parent.relative_to(root));rows.append({'stage':stage,'file':str(p),'summary':summary})
  for method,arm in summary.get('methods',{}).items():
   metrics.append({'stage':stage,'method':method,'description':NAMES.get(method,method),'tasks':arm.get('tasks',summary.get('tasks')),'qa_accuracy':arm.get('qa_accuracy'),'rec_recall_at_10':arm.get('rec_recall_at_10',arm.get('recall_at_10')),'macro_f1':arm.get('macro_f1'),'invalid_events':arm.get('invalid_events',arm.get('invalid')),'logical_calls':arm.get('logical_calls'),'prompt_tokens':arm.get('tokens',{}).get('prompt_tokens'),'completion_tokens':arm.get('tokens',{}).get('completion_tokens'),'logical_known_cny':arm.get('logical_known_cny'),'source':str(p)})
 expected=['probe','train','model','main','value','ablations','budget/0','budget/1','budget/2','budget/3','stress/missing','stress/redundant','stress/distractor','transfer','semantic']
 present={r['stage']for r in rows}
 if (root/'benchmark_data/tasks.json').exists():expected+=['benchmark','rank_baselines']
 result={'completed_stage_summaries':rows,'missing_stages':[s for s in expected if s not in present],'missing_results_are_not_zero':True}
 root.mkdir(parents=True,exist_ok=True);(root/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 with (root/'metrics.csv').open('w',newline='')as f:
  fields=list(metrics[0])if metrics else ['stage','method','source'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(metrics)
 lines=['# 集合价值博弈实验汇总','','|阶段|方法|QA正确率|推荐Recall@10|无效响应|逻辑调用|','|---|---|---:|---:|---:|---:|']
 for r in metrics:lines.append('|'+ '|'.join(str(r.get(k))if r.get(k)is not None else '未测'for k in ('stage','description','qa_accuracy','rec_recall_at_10','invalid_events','logical_calls'))+'|')
 lines+=['','未完成阶段：'+('、'.join(result['missing_stages'])or '无'),'','配对区间、逐题修正/损害、分配偏离和实际费用回执见 report.json 与各阶段 summary.json、physical_summary.json。逻辑费用包含共享请求的逻辑归属，不能跨方法相加当实际账单。','外部方法为同后端预算适配，非原论文模型配置复现；公开论文分数不并入同条件胜负。']
 (root/'report.md').write_text('\n'.join(lines)+'\n');return result
