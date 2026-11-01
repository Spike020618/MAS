"""Label-blind source omission, exact redundancy and unrelated static material."""
import argparse,copy,json
from pathlib import Path
from experiments.consensus_recommendation.common import create,fingerprint,sha
from experiments.prepare_protocol_v3 import tokens

def variants(tasks,originals):
 lookup={t['id']:t for t in originals};out={name:[]for name in ('missing','redundant','distractor')}
 for t in tasks:
  initial=t['metadata']['protocol_v3']['initial_source_ids'];unread=[sid for sid in t['source_catalog']if sid not in initial]
  if len(unread)<3:raise ValueError('not enough baseline unseen sources for stress')
  omit=max(unread,key=lambda sid:(t['source_catalog'][sid].get('retrieval_score_v3',0),sid))
  x=copy.deepcopy(t);del x['source_catalog'][omit];x['metadata']['stress']={'kind':'missing','removed_source_id':omit};out['missing'].append(x)
  x=copy.deepcopy(t);sid=unread[0]
  for i in range(2):
   s=copy.deepcopy(x['source_catalog'][sid]);new='duplicate:'+fingerprint([sid,i])[:24];s['source_id']=new;x['source_catalog'][new]=s
  x['metadata']['stress']={'kind':'redundant','duplicate_count':2};out['redundant'].append(x)
  # Static material only, never another query's conditioned retrieval scores.
  original=lookup[t['id']];pool=[s for s in original['source_catalog'].values()if s.get('kind')==('cached_web_reference'if t['kind']=='qa'else 'knowledge_graph')]
  if t['kind']=='qa':pool=[s for o in originals if o['kind']=='qa'for s in o['source_catalog'].values()if s.get('kind')=='cached_web_reference']
  present={fingerprint(s['text'][:2500])for s in t['source_catalog'].values()};pool=[s for s in pool if fingerprint(s['text'][:2500])not in present]
  if not pool:raise ValueError('no extra static distractor; no query-conditioned fallback')
  q=tokens(t['prompt']);s=copy.deepcopy(min(pool,key=lambda s:(len(q&tokens(s.get('title','')+' '+s['text'])),fingerprint(s['text']))));sid='noise:'+fingerprint([t['id'],s['text']])[:24]
  s.update(source_id=sid,text=s['text'][:2500],scope={**s.get('scope',{}),'task_id':t['id'],'subject':'external_reference'if t['kind']=='qa'else 'catalog'},allowed_uses=['prediction_basis'],current_user_fact_authority=False,retrieval_score_v3=0.)
  x=copy.deepcopy(t);x['source_catalog'][sid]=s;x['metadata']['stress']={'kind':'distractor','added_source_id':sid,'scope':'low lexical overlap static material; not certified irrelevant'};out['distractor'].append(x)
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--original-inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();tasks=json.loads(a.inputs.read_text());commit=json.loads((a.inputs.parent/'split_commitment.json').read_text());out=variants(tasks,json.loads(a.original_inputs.read_text()))
 for name,rows in out.items():create(a.output/name/'tasks.json',rows);create(a.output/name/'split_commitment.json',{**commit,'stress_variant':name,'baseline_inputs_sha256':sha(a.inputs)})
 print(json.dumps({'variants':{k:len(v)for k,v in out.items()},'model_api_calls':0},indent=2))
