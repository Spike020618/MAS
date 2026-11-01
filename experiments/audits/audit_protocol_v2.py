"""Read sealed v2 artifacts only. No API calls, state updates or semantic certification."""
import argparse,collections,json
from pathlib import Path
from experiments.consensus_recommendation.common import fingerprint
PLACEHOLDER='One relevant, testable premise; a hypothesis rather than a certified fact.'
def read(p):return json.loads(p.read_text())
def audit(root):
 tasks=read(root/'diagnostic/tasks.json');byid={t['id']:t for t in tasks};out={'model_api_calls':0,'modes':{}}
 sets=collections.Counter(tuple(sorted(t['source_catalog']))for t in tasks)
 out['source_packages']={'tasks':len(tasks),'distinct_packages':len(sets),'package_sizes_and_task_counts':[{'sources':list(k),'tasks':v}for k,v in sets.items()],'unique_sources':len({s for t in tasks for s in t['source_catalog']}),'qa_with_training_aggregate':sum(t['kind']=='qa'and any(s.get('kind')=='training_aggregate'for s in t['source_catalog'].values())for t in tasks)}
 for mode in ['heuristic','learned']:
  run=root/mode;seal=read(run/'generation_complete.json');plan=read(run/'plan.json');cases=[read(run/'cases'/f'case_{i}.json')for i in range(plan['count'])]
  if seal['hashes']!=[fingerprint(c)for c in cases]:raise ValueError('seal mismatch')
  stats={};details=[];space=collections.Counter()
  for c in cases:
   t=byid[c['task_id']];ga=c['methods']['game']['allocation'];ca=c['methods']['central']['allocation']
   if {q for _,q in ga.get('actions',[])}=={q for _,q in ca.get('actions',[])}:space['game_central_same_action_set']+=1
   unseen=len(t['source_catalog'])-(0 if t['kind']=='qa'else min(1,len(t['source_catalog'])))
   if unseen<=3:space['unseen_sources_at_most_action_capacity']+=1
   for method,arm in c['methods'].items():
    st=stats.setdefault(method,collections.Counter())
    for e in arm['events']:
     if e.get('valid'):
      st['valid_events']+=1
      st['placeholder_claim_events']+=any(a.get('text')==PLACEHOLDER for a in e['report'].get('A',[]))
     if e.get('patch_reason')!='assigned_source_not_used':continue
     st['assigned_source_rejections']+=1
     sid=e['source_action']['source_id'];report=e['report'];patch=report['C']['proposed_patch'];path=run/'slots'/('request_'+e['request_hash'][:32]+'.raw.json')
     wrapper=read(path);payload=json.loads(wrapper['request']['messages'][1]['content']);ex=payload.get('non_abstain_shape_only_not_a_recommendation',{}).get('patch',{})
     shape_copy=all(patch.get(k)==ex.get(k)for k in ['operation','value','a','b','evidence_ids','expected_version','preserve_ids'])
     row={'task_id':c['task_id'],'method':method,'kind':t['kind'],'slot':e['slot'],'request_hash':e['request_hash'],'assigned_source':sid,'assigned_source_present_in_payload':sid in payload['fetched_sources'],'assigned_source_named_in_assigned_doubt':sid in json.dumps(payload.get('assigned_doubt',{})),'cited_sources':patch.get('evidence_ids',[]),'cites_old_source_only':set(patch.get('evidence_ids',[]))<=set(e['source_action']['before_visible_ids']),'placeholder_claim':any(a.get('text')==PLACEHOLDER for a in report['A']),'patch_matches_supplied_example':shape_copy,'assigned_source_in_report_E':sid in report['E'],'assigned_source_linked_support_refute':any(i.get('source_id')==sid and i.get('relation')in ('support','refute')for i in report['I']),'semantic_verdict':'not_established; no relabel or auto-repair'}
     details.append(row)
     for k in ['assigned_source_present_in_payload','assigned_source_named_in_assigned_doubt','cites_old_source_only','placeholder_claim','patch_matches_supplied_example','assigned_source_in_report_E','assigned_source_linked_support_refute']:st[k]+=row[k]
  out['modes'][mode]={'sealed_cases':len(cases),'action_space':dict(space),'methods':{m:dict(v)for m,v in stats.items()},'rejections':details}
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.root);a.output.parent.mkdir(parents=True,exist_ok=True)
 if a.output.exists():raise ValueError('use a new audit output; original artifacts preserved')
 a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps({**r,'modes':{k:{q:v for q,v in d.items()if q!='rejections'}for k,d in r['modes'].items()}},ensure_ascii=False,indent=2))
