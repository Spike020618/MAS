"""Blind 80-question StrategyQA training selection and bounded real retrieval.
Official container is parsed but only qid/question are used; answer/facts never
enter selection, retrieval or serving tasks. Not an untouched test split.
"""
import argparse,json,hashlib,re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from experiments.consensus_recommendation.common import ROOT,create,sha
from .task_adapter import QA_LABELS,adapt_source
OUT=ROOT/'experiments/results/e2_qa80_v1'

def run(output=OUT,fetch_limit=8):
 output=Path(output)
 exclude={t['metadata']['native_id'].rsplit('/',1)[-1]for t in json.loads((ROOT/'experiments/results/evidence_consensus_e1_tasks_v1/tasks.json').read_text())if t['kind']=='qa'}
 raw=json.loads(QA_LABELS.read_text());public=[{'qid':r['qid'],'question':r['question']}for r in raw if r['qid']not in exclude]
 selected=sorted(public,key=lambda r:hashlib.sha256(('e2-qa80-training:'+r['qid']).encode()).hexdigest())[:80]
 if len(selected)!=80:raise ValueError('insufficient native questions')
 create(output/'selection_commitment.json',{'questions':selected,'excluded_e1_qids':sorted(exclude),'selection':'SHA256 qid, independent of answers/facts/outcomes','native_container_sha256':sha(QA_LABELS),'container_read_scope':'only qid/question extracted; gold fields never used','training_only':True})
 # Public material from existing independent banks may be reused only for the
 # SAME original question. Missing sources remain explicit, never fabricated.
 banks={}
 for p in (ROOT/'experiments/results').glob('**/banks/*.json'):
  b=json.loads(p.read_text());qid=str(b.get('question_id','')).rsplit('/',1)[-1]
  if qid:banks[qid]=b.get('sources',[])
 def retrieve(r):
  import requests
  path=output/'retrieval'/(r['qid']+'.json')
  if path.exists():return json.loads(path.read_text())
  try:
   response=requests.get('https://en.wikipedia.org/w/api.php',params={'action':'query','list':'search','srsearch':r['question'],'srlimit':2,'format':'json'},timeout=15)
   data=response.json();hits=data.get('query',{}).get('search',[]);sources=[]
   for h in hits:
    text=re.sub('<[^>]+>','',h.get('snippet',''))
    if text:sources.append({'title':h['title'],'text':text,'url':'https://en.wikipedia.org/?curid='+str(h['pageid']),'kind':'live_search_snippet','retrieval_limit':'search excerpt only, not full-page verification'})
   result={'qid':r['qid'],'sources':sources,'http_status':response.status_code}
  except Exception as e:result={'qid':r['qid'],'sources':[],'error_type':type(e).__name__}
  create(path,result);return result
 missing=[r for r in selected if not banks.get(r['qid'])]
 with ThreadPoolExecutor(max_workers=4)as pool:
  for result in pool.map(retrieve,missing[:fetch_limit]):banks[result['qid']]=result['sources']
 tasks=[];unresolved=[]
 for r in selected:
  taskid='e2:qa:'+r['qid'];catalog={}
  for i,s in enumerate(banks.get(r['qid'],[])):
   if not s.get('text')or not s.get('url'):continue
   sid='web:'+str(i)+':'+hashlib.sha256(s['text'].encode()).hexdigest()[:12]
   catalog[sid]=adapt_source(sid,{**s,'source_id':sid,'kind':s.get('kind','cached_web_reference'),'allowed_use':'reference_material'},taskid,subject='external_reference')
  if not catalog:unresolved.append(r['qid']);continue
  tasks.append({'id':taskid,'kind':'qa','prompt':r['question'],'options':['yes','no'],'candidates':[],'initial':'abstain','source_catalog':catalog,'metadata':{'native_dataset':'StrategyQA','native_id':'StrategyQA-official/'+r['qid'],'query_labels_available':False,'history_scope':'E2 training only; official train reused, not confirmation'}})
 create(output/'tasks.ready.json',tasks)
 summary={'committed':80,'ready':len(tasks),'unresolved_qids':unresolved,'retrieval_attempt_cap':fetch_limit,'complete_80_request_packets':len(tasks)==80,'note':'missing public retrieval is explicit; do not run/claim 80 completed tasks','model_api_calls':0}
 create(output/'summary.json',summary);print(json.dumps(summary,indent=2));return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--fetch-limit',type=int,default=8);a=p.parse_args();run(a.output,a.fetch_limit)
