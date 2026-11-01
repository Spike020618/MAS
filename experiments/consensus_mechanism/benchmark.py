"""HotpotQA yes/no supplied-context benchmark with original support annotations.

The task subset is defined by the native answer type, never model performance.
All official context paragraphs are retained; supporting facts stay in a vault.
"""
import collections,json
from pathlib import Path
from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha
from .data import read
REVISION='1908d6afbbead072334abe2965f91bd2709910ab'
REL='distractor/validation-00000-of-00001.parquet'
URL=f'https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/{REVISION}/{REL}'

def prepare(output,count=None,fetch=False):
 source=ROOT/'experiments/data/public/hotpotqa/validation.parquet'
 if not source.exists():
  if not fetch:raise ValueError('HotpotQA missing; use --fetch-data')
  import requests
  r=requests.get(URL,timeout=(15,120));r.raise_for_status();source.parent.mkdir(parents=True,exist_ok=True);source.write_bytes(r.content)
 import pyarrow.parquet as pq
 rows=pq.read_table(source).to_pylist();eligible=[r for r in rows if r['answer']in ('yes','no')];eligible.sort(key=lambda r:fingerprint(['hotpot-yesno-v1',r['id']]))
 if count is not None and(count<=0 or count>len(eligible)):raise ValueError('insufficient native yes/no examples; no silent downsizing')
 selected=eligible if count is None else eligible[:count];tasks=[];vault={};coverage=[]
 for row in selected:
  tid='hotpot:qa:'+row['id'];context=row['context'];catalog={};by_title={}
  for title,sentences in zip(context['title'],context['sentences']):
   sid='hotpot:'+fingerprint([tid,title])[:24];text=''.join(sentences)
   if not text.strip():raise ValueError('empty official paragraph')
   catalog[sid]={'source_id':sid,'title':title,'text':text,'kind':'official_benchmark_context','scope':{'task_id':tid,'subject':'reference'},'allowed_uses':['prediction_basis'],'retrieval_score_v3':0.}
   by_title[title]=(sid,sentences)
  supports=[]
  for title,index in zip(row['supporting_facts']['title'],row['supporting_facts']['sent_id']):
   if title not in by_title or not 0<=index<len(by_title[title][1]):raise ValueError('official supporting fact missing from full context')
   sid,sentences=by_title[title];supports.append({'source_id':sid,'sentence_index':index,'text':sentences[index]})
  if len({fingerprint(s['text'])for s in catalog.values()})<2:raise ValueError('insufficient distinct context')
  order=sorted(catalog,key=lambda sid:fingerprint(['initial-material',tid,sid]));ordered={sid:catalog[sid]for sid in order}
  tasks.append({'id':tid,'kind':'qa','prompt':row['question'],'options':['yes','no'],'candidates':[],'initial':'abstain','source_catalog':ordered,'metadata':{'native_id':'HotpotQA/'+row['id'],'native_dataset':'HotpotQA-yes-no-distractor','acquisition_partition':'diagnostic','evaluation_vault':str((output/'evaluation_vault.json').resolve()),'protocol_v3':{'initial_source_ids':[order[i%len(order)]for i in range(3)]},'history_scope':'official public validation, not private or pretraining-unseen','selection':'complete native yes/no subset or predeclared hash prefix; not selected by model outcome'}})
  vault[tid]={'kind':'qa','answer':row['answer'],'supporting_facts':supports};coverage.append(len(supports))
 create(output/'tasks.json',tasks);create(output/'evaluation_vault.json',vault)
 create(output/'manifest.json',{'source_url':URL,'revision':REVISION,'source_sha256':sha(source),'official_split':'validation/distractor','original_rows':len(rows),'eligible_yes_no':len(eligible),'selected_tasks':len(tasks),'label_counts':dict(collections.Counter(v['answer']for v in vault.values())),'all_official_paragraphs_retained':True,'support_sentence_coverage':1.,'context_count_distribution':dict(collections.Counter(len(t['source_catalog'])for t in tasks)),'support_sentences':sum(coverage),'model_api_calls':0,'scope':'yes/no subset with supplied official distractor context; not full HotpotQA leaderboard','gold_support_not_runtime_features':True})
 return tasks
