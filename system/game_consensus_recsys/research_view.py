"""Read-only replay of globally sealed research outputs; never reads gold or slots."""
import json
from pathlib import Path
from experiments.consensus_recommendation.common import fingerprint
ROOT=Path(__file__).resolve().parents[2]
STAGES={'main':'StrategyQA 与 ReDial','benchmark':'HotpotQA 是非题'}

def inventory(root=None):
 base=Path(root)if root else ROOT/'experiments/results/consensus_mechanism_paper_v2';out=[]
 for stage,title in STAGES.items():
  seal=base/stage/'generation_complete.json';row={'stage':stage,'title':title,'sealed':seal.is_file(),'cases':0}
  if seal.is_file():row['cases']=len(json.loads(seal.read_text())['task_ids'])
  out.append(row)
 return {'stages':out,'scope':'completed research replay, not online evaluation or satisfaction','labels_exposed':False}

def sealed_case(stage,index,root=None):
 if stage not in STAGES or type(index)is not int or index<0:raise ValueError('invalid stage or case index')
 base=(Path(root)if root else ROOT/'experiments/results/consensus_mechanism_paper_v2')/stage
 seal_path=base/'generation_complete.json'
 if not seal_path.is_file():raise RuntimeError('该批次尚未完整封存，暂不开放案例回放')
 seal=json.loads(seal_path.read_text())
 if index>=len(seal['task_ids']):raise IndexError('case not found')
 case=json.loads((base/'cases'/f'case_{index}.json').read_text())
 if fingerprint(case)!=seal['hashes'][index]or case['task_id']!=seal['task_ids'][index]:raise RuntimeError('sealed case integrity mismatch')
 task=json.loads((base/'inputs.json').read_text())[index]
 if task['id']!=case['task_id']:raise RuntimeError('task/case identity mismatch')
 return {'case':case,'task':{k:task[k]for k in ('id','kind','prompt','candidates','source_catalog')},'labels_exposed':False,'scope':'frozen experiment replay; not online invocation of the research mechanism'}
