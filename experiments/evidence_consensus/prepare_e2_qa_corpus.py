"""Retrieve independent cached public documents for committed QA80, no gold.
Cross-question retrieval is allowed: similarity is NOT support certification.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from experiments.consensus_recommendation.common import ROOT,create,sha
from .task_adapter import adapt_source

def imported_sources(path):
 rows=json.loads(Path(path).read_text());out=[]
 for r in rows:
  if not isinstance(r.get('url'),str)or not r['url'].startswith(('https://','http://'))or not isinstance(r.get('text'),str)or not r['text'].strip():raise ValueError('actual URL and text required')
  out.append({k:r[k]for k in ('url','text','title')if k in r})
 return out

def run(output,web_import=None):
 selected=json.loads((ROOT/'experiments/results/e2_qa80_v1/selection_commitment.json').read_text())['questions'];docs={};lineage=[]
 for p in sorted((ROOT/'experiments/results').glob('**/banks/*.json')):
  data=json.loads(p.read_text())
  for s in data.get('sources',[]):
   if not s.get('url')or not s.get('text'):continue
   key=hashlib.sha256((s['url']+'\n'+s['text']).encode()).hexdigest();docs[key]={k:s[k]for k in ('url','title','text')if k in s}
  lineage.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
 if web_import:
  for s in imported_sources(web_import):docs[hashlib.sha256((s['url']+'\n'+s['text']).encode()).hexdigest()]=s
 if not docs:raise ValueError('no actual external reference corpus')
 keys=sorted(docs);corpus=[docs[k]for k in keys];v=TfidfVectorizer(stop_words='english',ngram_range=(1,2),sublinear_tf=True);matrix=v.fit_transform([s.get('title','')+' '+s['text']for s in corpus]);scores=(v.transform([q['question']for q in selected])@matrix.T).toarray();tasks=[];diagnostics=[]
 for q,sim in zip(selected,scores):
  indices=np.argsort(-sim,kind='stable')[:3];tid='e2:qa:'+q['qid'];catalog={}
  for j in indices:
   s=corpus[j];sid='corpus:'+keys[j][:20]
   catalog[sid]=adapt_source(sid,{**s,'source_id':sid,'kind':'cached_web_reference','allowed_use':'reference_material','retrieval_cosine':float(sim[j]),'retrieval_warning':'retrieved relevance only; may be unrelated, never certified support','corpus_document_sha256':keys[j]},tid,subject='external_reference')
  tasks.append({'id':tid,'kind':'qa','prompt':q['question'],'options':['yes','no'],'candidates':[],'initial':'abstain','source_catalog':catalog,'metadata':{'native_dataset':'StrategyQA','native_id':'StrategyQA-official/'+q['qid'],'query_labels_available':False,'history_scope':'E2 training, excludes E1 qids; reused external corpus, no untouched-test claim','retrieval_max_cosine':float(sim.max())}})
  diagnostics.append({'task_id':tid,'max_cosine':float(sim.max()),'source_ids':list(catalog),'zero_overlap':bool(sim.max()==0)})
 output=Path(output);create(output/'tasks.json',tasks);create(output/'retrieval_diagnostics.json',diagnostics);create(output/'corpus_manifest.json',{'documents':len(corpus),'bank_lineage':lineage,'web_import':str(web_import)if web_import else None,'label_fields_used':False})
 summary={'count':len(tasks),'source_documents':len(corpus),'zero_overlap_queries':sum(d['zero_overlap']for d in diagnostics),'median_top_cosine':float(np.median([d['max_cosine']for d in diagnostics])),'complete_serving_packets':True,'external_source_quality':'weak cached corpus retrieval; no assertion evidence sufficient or supportive','labels_read':False,'model_api_calls':0}
 create(output/'summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'experiments/results/e2_qa80_corpus_v2');p.add_argument('--web-import',type=Path);a=p.parse_args();run(a.output,a.web_import)
