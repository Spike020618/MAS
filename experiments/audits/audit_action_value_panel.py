"""Read sealed action-value panel artifacts only. No API calls, no resends, no semantic certification."""
import argparse,collections,itertools,json
from pathlib import Path
from experiments.consensus_recommendation.common import fingerprint,sha
from experiments.evidence_consensus import protocol_v3 as p

def read(path):return json.loads(Path(path).read_text())

def decompose(cases,labels):
 """Behavioral decomposition plus recomputation of the sealed panel metrics from case data."""
 out={'kinds':{},'per_task':[]}
 for c in cases:
  kind=c['kind'];st=out['kinds'].setdefault(kind,{'tasks':0,'no_initial_claim':0,'branches':0,'invalid_branches':0,'invalid_reasons':collections.Counter(),'valid_outcomes':collections.Counter(),'decision_changes':0,'decision_changes_without_new_evidence':0,'decision_changes_to_gold':0,'decision_change_answers':collections.Counter(),'recomputed':collections.Counter()})
  st['tasks']+=1;gold=labels[c['task_id']];base=c['baseline']['decision']
  if not c['branches']:
   st['no_initial_claim']+=1;out['per_task'].append({'task_id':c['task_id'],'kind':kind,'no_initial_claim':True});continue
  events=c['branches'];reps=sorted({e['replicate']for e in events});lookup={(tuple(e['source_ids']),e['replicate']):e for e in events};sources=list(c['predictions']);pairs=list(itertools.combinations(sources,2))
  q=lambda e:p.quality(e['after_decision'],gold)
  local_changes=0;local_without=0
  for e in events:
   st['branches']+=1
   if not e['valid']:st['invalid_branches']+=1;st['invalid_reasons'][(e.get('error')or 'unknown')[:60]]+=1;continue
   st['valid_outcomes'][f'{bool(e.get("patch_accepted"))}|{str(e.get("patch_reason"))[:40]}']+=1
   if e['after_decision']!=base:
    st['decision_changes']+=1;local_changes+=1
    if not e['source_ids']:st['decision_changes_without_new_evidence']+=1;local_without+=1
    st['decision_change_answers'][json.dumps(e['after_decision'],ensure_ascii=False)[:40]]+=1
    if q(e)>p.quality(base,gold):st['decision_changes_to_gold']+=1
  controls={r:q(lookup[((),r)])for r in reps};single={s:sum(q(lookup[((s,),r)])-controls[r]for r in reps)/len(reps)for s in sources}
  selected=max(sources,key=lambda s:(c['predictions'][s]['gain'],s));uniform=sum(single.values())/len(single)
  st['recomputed']['initial_quality_sum']+=float(p.quality(base,gold));st['recomputed']['predicted_choice_gain_sum']+=single[selected];st['recomputed']['uniform_choice_gain_sum']+=uniform
  st['recomputed']['tasks_with_positive_single_mean']+=any(v>0 for v in single.values());st['recomputed']['tasks_with_varying_single_means']+=max(single.values())-min(single.values())>1e-12
  first=reps[0];later=reps[1:];empirical=max(sources,key=lambda s:(q(lookup[((s,),first)])-controls[first],s));held=sum(q(lookup[((empirical,),r)])-controls[r]for r in later)/len(later);held_uniform=sum(q(lookup[((s,),r)])-controls[r]for s in sources for r in later)/(len(sources)*len(later))
  st['recomputed']['repeat0_choice_later_gain_sum']+=held;st['recomputed']['later_uniform_gain_sum']+=held_uniform
  persistent_flags=[]
  for a,b in pairs:
   rows=[]
   for r in reps:
    ea,eb,eab,e0=lookup[((a,),r)],lookup[((b,),r)],lookup[((a,b),r)],lookup[((),r)];qa,qb,qab,q0=q(ea),q(eb),q(eab),q(e0)
    rows.append({'joint_only_improvement':qab>q0 and qa<=q0 and qb<=q0,'all_valid':all(e['valid']for e in [ea,eb,eab,e0])})
   persistent=all(x['joint_only_improvement']and x['all_valid']for x in rows);st['recomputed']['pairs_joint_only_in_all_repeats_valid']+=persistent;persistent_flags.append(persistent)
  st['recomputed']['tasks_with_repeat_consistent_joint_only_pair']+=any(persistent_flags)
  st['recomputed']['invalid_branches']+=sum(not e['valid']for e in events);st['recomputed']['branches']+=len(events);st['recomputed']['control_quality_changed_between_repeats']+=len(set(controls.values()))>1
  st['recomputed']['single_repeat_absolute_difference_sum']+=sum(abs(q(lookup[((s,),reps[0])])-q(lookup[((s,),reps[1])]))for s in sources)/len(sources)
  if kind=='rec':st['recomputed']['targets']+=len(gold['targets'])
  out['per_task'].append({'task_id':c['task_id'],'kind':kind,'initial_decision':base,'gold':gold,'initial_quality':float(p.quality(base,gold)),'control_quality':{r:float(q(lookup[((),r)]))for r in reps},'decision_changes':local_changes,'decision_changes_without_new_evidence':local_without})
 return out

def audit(root):
 root=Path(root);tasks=read(root/'inputs.json');seal=read(root/'generation_complete.json');plan=read(root/'plan.json');summary=read(root/'summary.json');physical=read(root/'physical_summary.json');labels=read(root/'labels.after_generation.json')
 cases=[read(root/'cases'/f'case_{i}.json')for i in range(len(tasks))]
 if seal['task_ids']!=[t['id']for t in tasks]or seal['hashes']!=[fingerprint(c)for c in cases]:raise ValueError('panel seal mismatch')
 for t,c in zip(tasks,cases):
  if fingerprint(c['baseline']['task'])!=fingerprint(t):raise ValueError('baseline/input mismatch')
 d=decompose(cases,labels);checks={}
 for kind,st in d['kinds'].items():
  ref=summary['kinds'].get(kind,{})
  for k,v in st['recomputed'].items():
   if k in('initial_quality_sum','targets'):continue
   got=ref.get(k,0);checks[f'{kind}.{k}']=abs(float(got)-float(v))<1e-12
 return {'model_api_calls':0,'scope':'read-only audit of a sealed matched-state action value panel; no resend, no relabel, no semantic certification','panel_version':plan['version'],'estimator':plan['estimator'],'model_sha256':plan['model_sha256'],'input_sha256':plan['input_sha256'],'code_sha256':plan.get('source_sha256',{}).get('experiments/action_value_panel.py'),'result_sha256':sha(root/'summary.json'),'verification':{'seal_verified':True,'summary_recompute_checks':checks,'summary_matches_recompute':all(checks.values())},'physical':{'physical_attempts':physical['physical_attempts'],'returned_requests':physical['returned_requests'],'http_status':physical['http_status'],'configured_known_cny':physical['configured_known_cny'],'unknown_intents_do_not_resend':physical['unknown_intents']},'kinds':{k:{kk:(dict(vv)if isinstance(vv,collections.Counter)else vv)for kk,vv in st.items()if kk!='recomputed'}for k,st in d['kinds'].items()},'recomputed_sums':{k:dict(st['recomputed'])for k,st in d['kinds'].items()},'per_task':d['per_task']}

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
 result=audit(args.root);args.output.parent.mkdir(parents=True,exist_ok=True)
 if args.output.exists():raise ValueError('use a new audit output; original artifacts preserved')
 args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items()if k not in('per_task','recomputed_sums')},ensure_ascii=False,indent=2))
