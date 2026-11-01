"""Small same-state interface comparison. No labels before all generations seal."""
import argparse,collections,copy,json
from pathlib import Path
from . import core as c
from .engine import initialize
from .runner import frozen_plan,locked
from .runtime import Runtime
from .data import read,labels_for
from experiments.consensus_recommendation.common import ROOT,create,fingerprint
from experiments.paired_deliberation.native_optimization_runtime import audit

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'experiments/results/decision_linked_probe_v1');p.add_argument('--inputs',type=Path,default=ROOT/'experiments/results/consensus_mechanism_paper_v2/data/train.json');p.add_argument('--tasks',type=Path,help='explicit task list (e.g. opportunity-verified tasks: initial wrong and a catalog source adjudicated to support the gold answer); labels must not drive live selection on test partitions');p.add_argument('--run',action='store_true');p.add_argument('--max-cny',type=float,default=.2);a=p.parse_args()
 if not 0<a.max_cny<=.5:raise ValueError('probe hard limit must be 0..0.5 CNY')
 tasks=read(a.tasks)if a.tasks else read(a.inputs)
 if not a.tasks:tasks=sum(([t for t in tasks if t['kind']==kind][:2]for kind in ('qa','rec')),[])
 if not a.tasks and len(tasks)!=4:raise ValueError('two native tasks per kind required')
 if a.tasks and not 2<=len(tasks)<=8:raise ValueError('explicit probe tasks must be 2..8; single-kind opportunity lists are allowed')
 if not a.run:print(json.dumps({'tasks':len(tasks),'max_calls':len(tasks)*5,'provider':'qwen_flash','max_cny':a.max_cny,'network':False}));return
 from dotenv import load_dotenv
 load_dotenv(ROOT/'.env.local',override=False)
 with locked(a.root):
  plan=frozen_plan(a.root,tasks,'repair_probe','qwen_flash',1,('ordinary','linked'),max_cny=a.max_cny);api=Runtime(a.root,plan,'qwen_flash');cases=[]
  for i,t in enumerate(tasks):
   dest=a.root/'cases'/f'case_{i}.json'
   if dest.exists():cases.append(read(dest));continue
   state,reports,events,_=initialize(t,i,api);rows=c.actions(t,state,allow_joint=False)
   if not rows:raise RuntimeError('no actionable claim; probe not qualified')
   row=min(rows,key=lambda r:fingerprint([t['id'],r['id']]));exposed=copy.deepcopy(state);c.expose(t,exposed,row['source_ids']);arms={}
   for mode,phase in [('ordinary','direct'),('linked','review')]:
    task=copy.deepcopy(t);task['metadata']['decision_linked_v1']=mode=='linked'
    after,event=c.invoke(api,f'{i}_{mode}',task,copy.deepcopy(exposed),phase,100,reports,row,reviewer=0)
    arms[mode]={'decision':after['decision'],'event':event}
   case={'task_id':t['id'],'kind':t['kind'],'initial_decision':state['decision'],'action':row,'initial_events':events,'arms':arms};create(dest,case);cases.append(case)
  create(a.root/'generation_complete.json',{'task_ids':[t['id']for t in tasks],'hashes':[fingerprint(x)for x in cases]});labels=labels_for(tasks,a.root/'generation_complete.json');summary={'tasks':len(tasks),'methods':{},'physical':audit(a.root,'qwen_flash')}
  for mode in ['ordinary','linked']:
   pairs=[(c.quality(x['initial_decision'],labels[x['task_id']]),c.quality(x['arms'][mode]['decision'],labels[x['task_id']]))for x in cases]
   summary['methods'][mode]={'valid':sum(x['arms'][mode]['event']['valid']for x in cases),'improved':sum(b>a for a,b in pairs),'harmed':sum(b<a for a,b in pairs),'changed':sum(x['arms'][mode]['decision']!=x['initial_decision']for x in cases)}
   if mode=='linked':summary['methods'][mode]['revision_basis']=dict(collections.Counter(str(x['arms']['linked']['event'].get('revision_basis'))for x in cases))
  v=summary['methods']['linked'];summary['promotion_gate']=v['valid']==len(cases) and v['improved']>0 and v['harmed']<=v['improved'];summary['gate_validity']='improved>0 is only meaningful when tasks carry verified improvement opportunity; on no-opportunity tasks keep is correct behavior and this gate can never pass';summary['scope']='engineering/early signal only; small task counts do not establish effectiveness';create(a.root/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
