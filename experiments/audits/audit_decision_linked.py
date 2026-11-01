"""Read-only opportunity/judgment/execution audits. Probe phase reads sealed labels only after seal verification; main phase never reads labels. No API calls, no resends, no semantic certification."""
import argparse,collections,json,re
from pathlib import Path
from experiments.consensus_recommendation.common import ROOT,fingerprint
from experiments.consensus_mechanism import core
from experiments.consensus_mechanism.data import labels_for,read

STOP=frozenset('the a an and or of to in is are was were be been being it its this that these those with for on at by as not no yes nor but if then than so such from into over under about after before between during through can could would should shall may might must do does did done has have had having will'.split())
BRIDGE={'min_shared':2,'min_jaccard':.2}
LABEL_KEYS=frozenset('gold label labels answer targets'.split())
NULL_CLASSES=('refute_bridge_no_patch','refute_no_bridge_low_overlap','support_no_patch','unclear_no_patch')

def tokens(text):return [t for t in re.findall(r'[a-z0-9]+',str(text).lower())if len(t)>2 and t not in STOP]
def overlap(a,b):
 sa,sb=set(tokens(a)),set(tokens(b));u=len(sa|sb)
 return {'shared':len(sa&sb),'jaccard':round(len(sa&sb)/u,4)if u else 0.0}
def bridge_proxy(claim_text,question):
 o=overlap(claim_text,question);return {**o,'bridge':o['shared']>=BRIDGE['min_shared']or o['jaccard']>=BRIDGE['min_jaccard']}

def classify_probe_case(case,task,label):
 kind=task['kind'];initial=case['initial_decision'];q1=float(core.quality(initial,label))
 arms={}
 for mode in('ordinary','linked'):
  arm=case['arms'][mode];ev=arm['event'];rep=ev.get('report')or{}
  arms[mode]={'valid':ev.get('valid'),'accepted':ev.get('accepted'),'error':ev.get('error'),'changed':arm['decision']!=initial,'quality':float(core.quality(arm['decision'],label)),'decision_effect':rep.get('decision_effect'),'patch':rep.get('patch'),'verdict':rep.get('verdict'),'claim_excerpt':str(rep.get('claim')or'')[:400]}
 if kind=='qa':
  source_texts=[task['source_catalog'][sid].get('text','')for sid in case['action'].get('source_ids',[])]
  q2={'mode':'qa_semantic_pending','question_overlap':[overlap(t,task['prompt'])for t in source_texts]}
 else:
  pool=[str(c.get('movie_id'))for c in task.get('candidates',[])];top10=[str(x)for x in list(initial)[:10]];targets=[str(t)for t in label['targets']]
  in_pool=[t for t in targets if t in pool];in_top10=[t for t in targets if t in top10]
  q2={'mode':'rec_mechanical','targets':targets,'in_pool':in_pool,'in_top10':in_top10,'outside_top10_in_pool':[t for t in in_pool if t not in top10],'mechanical_opportunity':any(t in pool and t not in top10 for t in targets)}
 return {'task_id':case['task_id'],'kind':kind,'question':task['prompt'],'gold':label,'initial_decision':initial,'q1_initial_quality':q1,'q1_initial_correct':q1>=1.0,'q1_initial_partial':0<q1<1.0,'q2':q2,'arms':arms,'action_target':case['action'].get('target_id'),'action_source_ids':case['action'].get('source_ids',[]),'source_excerpts':{sid:task['source_catalog'][sid].get('text','')[:240]for sid in case['action'].get('source_ids',[])}}

def probe_audit(root):
 root=Path(root);tasks=read(root/'inputs.json');seal=read(root/'generation_complete.json');cases=[read(root/'cases'/f'case_{i}.json')for i in range(len(tasks))]
 if seal['task_ids']!=[t['id']for t in tasks]or seal['hashes']!=[fingerprint(c)for c in cases]:raise ValueError('probe seal mismatch')
 for t,c in zip(tasks,cases):
  if c['task_id']!=t['id']:raise ValueError('case/input mismatch')
 labels=labels_for(tasks,root/'generation_complete.json')
 rows=[classify_probe_case(c,t,labels[c['task_id']])for c,t in zip(cases,tasks)]
 agg={'tasks':len(rows),'q1_initial_correct':sum(r['q1_initial_correct']for r in rows),'q1_initial_partial':sum(r['q1_initial_partial']for r in rows),'q1_initial_zero':sum(r['q1_initial_quality']==0 for r in rows),'rec_mechanical_opportunity':sum(r['q2'].get('mechanical_opportunity',False)for r in rows),'linked_changed':sum(r['arms']['linked']['changed']for r in rows),'ordinary_changed':sum(r['arms']['ordinary']['changed']for r in rows),'linked_valid':sum(bool(r['arms']['linked']['valid'])for r in rows),'ordinary_valid':sum(bool(r['arms']['ordinary']['valid'])for r in rows)}
 return {'model_api_calls':0,'scope':'read-only opportunity/judgment/execution audit of a sealed decision-linked probe; labels read only after seal verification; no semantic certification','root':str(root),'verification':{'seal_verified':True,'labels_after_seal':True},'aggregate':agg,'per_case':rows}

def probe_report(result):
 text=['# 决策相关小批：机会—判断—执行逐例审计（只读）','',f"根目录：{result['root']}；封存校验通过；标签在封存后读取；无API调用；机械重叠只是代理，不作语义认证。",'']
 for r in result['per_case']:
  text+= [f"## {r['task_id']}（{r['kind']}）",f"- 问题：{str(r['question'])[:200]}",f"- gold：{json.dumps(r['gold'],ensure_ascii=False)}",f"- 初始决策：{json.dumps(r['initial_decision'],ensure_ascii=False)[:160]}",f"- Q1 初始质量：{r['q1_initial_quality']}（完全正确={r['q1_initial_correct']}，部分正确={r['q1_initial_partial']}）"]
  if r['q2']['mode']=='rec_mechanical':text+= [f"- Q2 机械层：gold在候选池内={r['q2']['in_pool']}；在Top10内={r['q2']['in_top10']}；池内Top10外（可换入机会）={r['q2']['outside_top10_in_pool']}；机械机会存在={r['q2']['mechanical_opportunity']}"]
  else:text+= [f"- Q2 机械层（QA无机械判定，语义留待裁决）：各指派来源与问题的词面重叠={[o['shared'] for o in r['q2']['question_overlap']]}"]
  for mode in('ordinary','linked'):
   a=r['arms'][mode];eff=a['decision_effect']or{}
   text+= [f"- {mode}臂：valid={a['valid']} accepted={a['accepted']} changed={a['changed']} quality={a['quality']} error={a['error']}",f"  decision_effect：{json.dumps(eff,ensure_ascii=False)[:300]}"]
  text+= [f"- 指派来源摘录：{json.dumps(r['source_excerpts'],ensure_ascii=False)[:600]}",'']
 text+=['','语义裁决（Q2第二层）留给用户：对照问题、gold、初始决策与指派来源，判断资料是否真正支持更好的输出。本审计不代做语义判断。']
 return '\n'.join(text)+'\n'

def classify_review_event(event,claim_text,question,kind):
 rep=event.get('report')or{};patch=rep.get('patch');verdict=rep.get('verdict');before=event.get('before');bp=bridge_proxy(claim_text,question)
 # Patch-bearing events classify by patch properties (same-value / rejected / accepted); the contract error stays recorded. Patch-less events classify by validity then verdict.
 if patch is not None:
  same=str(patch.get('value'))==str(before)if kind=='qa'else patch.get('a')==patch.get('b')
  cls='same_value_patch'if same else('patch_accepted'if event.get('accepted')else 'patch_not_accepted')
 elif not event.get('valid'):cls='invalid_contract'
 elif verdict=='refute':cls='refute_bridge_no_patch'if bp['bridge']else 'refute_no_bridge_low_overlap'
 elif verdict=='support':cls='support_no_patch'
 elif verdict=='unclear':cls='unclear_no_patch'
 else:cls='missing_verdict'
 rationale=str(rep.get('rationale')or'')
 return {'class':cls,'valid':event.get('valid'),'accepted':event.get('accepted'),'error':event.get('error'),'verdict':verdict,'patch':patch,'target_id':rep.get('target_id'),'bridge':bp,'rationale_excerpt':rationale[:400],'slot':event.get('slot'),'decision_reference':any(w in rationale.lower()for w in('answer','decision','ranking','top10','candidate'))}

def main_audit(root):
 root=Path(root);tasks=read(root/'inputs.json')
 for t in tasks:
  if LABEL_KEYS&set(t):raise ValueError('label-like fields present in inputs.json; main audit must stay label-free')
 byid={t['id']:t for t in tasks};case_paths=sorted((root/'cases').glob('case_*.json'));rows=[];cls_counter=collections.Counter();patch_rows=[];null_events=[];valid_count=0
 for path in case_paths:
  c=read(path);t=byid.get(c['task_id']);g=(c.get('methods')or{}).get('game')
  if t is None or g is None:rows.append({'case':path.name,'task_id':c.get('task_id'),'skipped':'missing task or game method'});continue
  claims=(g.get('state')or{}).get('claims')or{};question=t['prompt'];case_events=[]
  for e in g['events']:
   if e.get('phase')!='review':continue
   tid=(e.get('report')or{}).get('target_id')or(e.get('action')or{}).get('target_id');claim_text=str((claims.get(tid)or{}).get('text',''))
   row=classify_review_event(e,claim_text,question,t['kind']);row['task_id']=c['task_id'];row['kind']=t['kind'];case_events.append(row)
   if e.get('valid'):valid_count+=1
   cls_counter[(t['kind'],row['class'])]+=1;cls_counter[row['class']]+=1
   if row['patch'] is not None:patch_rows.append(row)
   elif row['class']in NULL_CLASSES:null_events.append(row)
  rows.append({'case':path.name,'task_id':c['task_id'],'kind':t['kind'],'reviews':len(case_events),'classes':[r['class']for r in case_events]})
 total=sum(r.get('reviews',0)for r in rows)
 strata=collections.defaultdict(list)
 for r in null_events:strata[(r['kind'],r['class'])].append(r)
 sample=[];keys=sorted(strata,key=str)
 while len(sample)<30 and any(strata[k]for k in keys):
  for k in keys:
   if strata[k] and len(sample)<30:sample.append(strata[k].pop(0))
 cross=None;prior=ROOT/'docs/research/update_repair_2026-10-29/audit.json'
 if prior.exists():
  a=read(prior)['game_review'];mine={'reviews':total,'valid':valid_count,'accepted':sum(1for p in patch_rows if p['accepted']),'null_patch':total-len(patch_rows),'patch_events':len(patch_rows)}
  cross={'prior':a,'mine':mine,'matches':{k:mine[k]==a[k]for k in('reviews','valid','accepted','null_patch')}}
 return {'model_api_calls':0,'scope':'label-free execution-side audit of the interrupted main batch; no label reads, no seal fabrication, no semantic certification','root':str(root),'cases':len(rows),'label_read':False,'classes':{str(k):v for k,v in cls_counter.items()},'bridge_settings':BRIDGE,'patch_events':patch_rows,'sample_30_for_semantic_pass':sample,'crosscheck_with_prior_audit':cross,'per_case':rows}

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--mode',choices=['probe','main'],required=True);a.add_argument('--root',type=Path);a.add_argument('--output',type=Path);a.add_argument('--report',type=Path);args=a.parse_args()
 if args.mode=='probe':
  root=args.root or ROOT/'experiments/results/decision_linked_probe_v2';out=args.output or root/'sealed_opportunity_audit.json';result=probe_audit(root);text=probe_report(result)
 else:
  root=args.root or ROOT/'experiments/results/consensus_mechanism_paper_v2/main';out=args.output or ROOT/'docs/research/update_repair_2026-10-29/phaseB_main_execution_audit.json';result=main_audit(root);text=None
 if out.exists():raise ValueError('use a new audit output; original artifacts preserved')
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 if text is not None:
  rep=args.report or root/'sealed_opportunity_audit.md'
  if rep.exists():raise ValueError('use a new report output; original artifacts preserved')
  rep.write_text(text)
 print(json.dumps({k:v for k,v in result.items()if k not in('per_case','patch_events','sample_30_for_semantic_pass','classes')},ensure_ascii=False,indent=2))
