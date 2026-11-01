"""Decision-perturbation recovery probe: deterministically corrupt a verified-correct initial decision, then test whether the linked review detects and repairs the visible evidence-decision inconsistency. Labels are read only after seal."""
import argparse,collections,copy,json
from pathlib import Path
from . import core as c
from .engine import initialize
from .runner import frozen_plan,locked
from .runtime import Runtime
from .data import read,labels_for
from experiments.consensus_recommendation.common import ROOT,create,fingerprint
from experiments.paired_deliberation.native_optimization_runtime import audit

def perturb(task,decision):
 """Label-free deterministic corruption: QA flips to the other supplied option; rec replaces the Top10 head with a fingerprint-chosen pool outsider so the original head drops to rank 11."""
 if task['kind']=='qa':
  other=[o for o in task['options']if o!=decision]
  if not other:return copy.deepcopy(decision),'unperturbed_no_other_option'
  return other[0],f'flipped {decision} -> {other[0]}'
 ranked=[str(x)for x in decision]
 if len(ranked)<2:return copy.deepcopy(decision),'unperturbed_ranking_too_short'
 tail=min(10,len(ranked)-1)
 return [ranked[tail]]+ranked[1:tail]+[ranked[0]]+ranked[tail+1:],f'swapped head {ranked[0]} with rank {tail+1} item {ranked[tail]}'

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'experiments/results/decision_perturb_probe_v1');p.add_argument('--inputs',type=Path,default=ROOT/'experiments/results/consensus_mechanism_paper_v2/data/train.json');p.add_argument('--tasks',type=Path,help='explicit task list: verified-correct initial decisions (QA initial==gold; rec all gold in Top10 with a gold head) selected on the train partition where labels are design-legal');p.add_argument('--run',action='store_true');p.add_argument('--max-cny',type=float,default=.2);a=p.parse_args()
 if not 0<a.max_cny<=.5:raise ValueError('probe hard limit must be 0..0.5 CNY')
 tasks=read(a.tasks)if a.tasks else read(a.inputs)
 if not 2<=len(tasks)<=12:raise ValueError('explicit perturb tasks must be 2..12')
 if not a.run:print(json.dumps({'tasks':len(tasks),'max_calls':len(tasks)*5,'provider':'qwen_flash','max_cny':a.max_cny,'network':False}));return
 from dotenv import load_dotenv
 load_dotenv(ROOT/'.env.local',override=False)
 with locked(a.root):
  plan=frozen_plan(a.root,tasks,'perturb_probe','qwen_flash',1,('ordinary','linked'),max_cny=a.max_cny);api=Runtime(a.root,plan,'qwen_flash');cases=[]
  for i,t in enumerate(tasks):
   dest=a.root/'cases'/f'case_{i}.json'
   if dest.exists():cases.append(read(dest));continue
   state,reports,initial,_=initialize(t,i,api)
   pre=copy.deepcopy(state['decision']);corrupted,note=perturb(t,pre)
   state['decision']=copy.deepcopy(corrupted)
   rows=c.actions(t,state)
   if not rows:raise RuntimeError('no actionable claim; perturb probe not qualified')
   row=min(rows,key=lambda r:fingerprint([t['id'],r['id']]));exposed=copy.deepcopy(state);c.expose(t,exposed,row['source_ids']);arms={}
   for mode,phase in (('ordinary','direct'),('linked','review')):
    task=copy.deepcopy(t);task['metadata']['decision_linked_v1']=mode=='linked'
    after,event=c.invoke(api,f'{i}_{mode}',task,copy.deepcopy(exposed),phase,100,reports,row,reviewer=0)
    arms[mode]={'decision':after['decision'],'event':event}
   case={'task_id':t['id'],'kind':t['kind'],'pre_decision':pre,'perturbation':note,'perturbed_decision':corrupted,'action':row,'initial_events':initial,'arms':arms};create(dest,case);cases.append(case);print('completed perturb',i+1,t['id'],note,flush=True)
  create(a.root/'generation_complete.json',{'task_ids':[t['id']for t in tasks],'hashes':[fingerprint(x)for x in cases]});labels=labels_for(tasks,a.root/'generation_complete.json');summary={'tasks':len(tasks),'methods':{},'physical':audit(a.root,'qwen_flash')}
  for mode in ('ordinary','linked'):
   pairs=[(c.quality(x['perturbed_decision'],labels[x['task_id']]),c.quality(x['arms'][mode]['decision'],labels[x['task_id']]))for x in cases]
   summary['methods'][mode]={'valid':sum(x['arms'][mode]['event']['valid']for x in cases),'repaired_exact':sum(x['arms'][mode]['decision']==x['pre_decision']for x in cases),'improved':sum(b>a for a,b in pairs),'harmed':sum(b<a for a,b in pairs),'changed':sum(x['arms'][mode]['decision']!=x['perturbed_decision']for x in cases)}
   if mode=='linked':
    summary['methods'][mode]['revision_basis']=dict(collections.Counter(str(x['arms']['linked']['event'].get('revision_basis'))for x in cases))
    summary['methods'][mode]['decision_effect_choices']=dict(collections.Counter(str(((x['arms']['linked']['event'].get('report')or{}).get('decision_effect')or{}).get('choice'))for x in cases))
  v=summary['methods']['linked'];summary['promotion_gate']=v['valid']==len(cases)and v['repaired_exact']>0 and v['harmed']==0;summary['scope']='perturbation-recovery diagnostic; repair measured against the recorded pre-perturbation decision; not effectiveness proof';create(a.root/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
