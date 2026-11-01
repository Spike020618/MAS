"""Task-held-out preparation with shared public background, never future labels."""
import argparse,collections,copy,json,math,re
from pathlib import Path
from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha
from experiments.prepare_disjoint_acquisition import group,partition
STOP=set('the a an is are was were do does did of to in on at and or for with this that it as be by from can would which what who how have has i you movie movies film films seeker recommender'.split())
def tokens(s):return set(re.findall(r'[a-z0-9]{2,}',s.lower()))-STOP
def build(tasks,pool_size=8,corpus_path=None):
 if pool_size<6:raise ValueError('at least six candidate sources required; do not manufacture duplicates')
 docs={fingerprint(s['text']):copy.deepcopy(s)for t in tasks if t['kind']=='qa'for s in t['source_catalog'].values()if s.get('kind')=='cached_web_reference'}
 docs=list(docs.values());df=collections.Counter(w for s in docs for w in tokens(s.get('title','')+' '+s['text']))
 out=[];split={};diagnostics=[]
 corpus=None
 if corpus_path is not None:
  from experiments.core.mechanism.title_anchor_retrieval import TitleAnchorCorpus
  corpus=TitleAnchorCorpus(corpus_path)
 for original in tasks:
  t=copy.deepcopy(original);q=tokens(t['prompt']);p=partition(group(t));split[t['id']]=p
  if t['kind']=='qa':
   ranked=[]
   for s in docs:
    title=tokens(s.get('title',''));body=tokens(s['text']);matched=q&(title|body)
    score=sum(math.log(1+(len(docs)+1)/(df[w]+1))*(3 if w in title else 1)for w in matched)/max(1,len(q))
    if matched:ranked.append((score,copy.deepcopy(s)))
   ranked.sort(key=lambda x:(-x[0],fingerprint(x[1]['text'])))
   if corpus is not None:
    from urllib.parse import quote
    ranked=[]
    for doc in corpus.search(t['prompt'],k=10):
     relevance=max(0.,-doc['rank_score']);score=relevance/(1+relevance)
     ranked.append((score,{'kind':'cached_web_reference','title':doc['title'],'text':doc['text'],'document_id':doc['docid'],'url':'https://en.wikipedia.org/wiki/'+quote(doc['title'].replace(' ','_')),'scope':{'task_id':t['id'],'subject':'external_reference'},'retrieval_engine':'title-anchor + BM25 public corpus','corpus_document_sha256':doc['text_sha256'],'semantic_truth_certified':False}))
  else:
   # Only this native query's cards. Its conditioned retrieval numbers stay here.
   ranked=[];ids=[str(c.get('movie_id',c.get('id')))for c in t['candidates']];order={m:i for i,m in enumerate(ids)}
   for s in t['source_catalog'].values():
    if s.get('kind')not in ('knowledge_graph','training_aggregate','historical_prediction_basis'):continue
    if s.get('scope',{}).get('task_id')!=t['id']:raise ValueError('foreign query conditioned card')
    s=copy.deepcopy(s);m=re.search(r'Movie (\d+)',s['text']);mid=str(s.get('movie_id')or(m.group(1)if m else ''));rank=order.get(mid,len(ids));score=len(q&tokens(s['text']))/max(1,len(q))+1/(rank+2)
    s['title']=next((c.get('title','')for c in t['candidates']if str(c.get('movie_id',c.get('id')))==mid),s.get('title',''))
    s['conditioned_on_native_id']=t['metadata']['native_id'];ranked.append((score,s))
   ranked.sort(key=lambda x:(-x[0],fingerprint(x[1]['text'])))
   # Round-robin evidence kinds prevents top-score cards of one kind occupying all slots.
   buckets={k:[v for v in ranked if v[1].get('kind')==k]for k in sorted({v[1].get('kind')for v in ranked})};ranked=[]
   while any(buckets.values()):
    for b in buckets.values():
     if b:ranked.append(b.pop(0))
  chosen=[];seen=set()
  for score,s in ranked:
   h=fingerprint(s['text'])
   if h in seen:continue
   seen.add(h);chosen.append((score,s))
   if len(chosen)==pool_size:break
  catalog={}
  for score,s in chosen:
   sid='v3:'+fingerprint([t['id'],s['text']])[:24];s['origin_source_id']=s.get('source_id');s['source_id']=sid;s['scope']={**s.get('scope',{}),'task_id':t['id']};s['text']=s['text'][:2500];s['retrieval_score_v3']=score;s['current_user_fact_authority']=False;s['allowed_uses']=['prediction_basis'];catalog[sid]=s
  t['source_catalog']=catalog;t['metadata']['acquisition_partition']=p;t['metadata']['protocol_v3']={'initial_source_ids':list(catalog)[:3],'additional_action_cap':2,'source_selection':'query lexical retrieval; own-query conditioned cards only','shared_background':True}
  diagnostics.append({'task_id':t['id'],'kind':t['kind'],'sources':len(catalog),'unseen_after_initial':max(0,len(catalog)-3),'budget_pressure':len(catalog)>5,'semantic_relevance':'not certified','positive_lexical_matches':sum(v[0]>0 for v in chosen),'unique_source_titles':len({s.get('title','')for s in catalog.values()}),'question_title_matches':sum(bool(q&tokens(s.get('title','')))for s in catalog.values())})
  out.append(t)
 commit={'task_split':split,'protocol':'v3-task-heldout-shared-public-background','scope':'development task-heldout; public background may overlap; NOT document-heldout or untouched confirmation','source_policy':'never transfer query conditioned scores; public QA documents shared','label_access':False}
 if corpus is not None:corpus.close()
 return out,commit,diagnostics
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus',type=Path,default=ROOT/'experiments/results/redesign_v1/retrieval_env/wiki_fts.sqlite');a=p.parse_args()
 if not a.corpus.exists():raise SystemExit('public corpus missing; pass --corpus explicitly, no cached-material fallback')
 out,c,d=build(json.loads(a.inputs.read_text()),corpus_path=a.corpus);c['inputs_sha256']=sha(a.inputs);c['corpus']={'path':str(a.corpus.resolve()),'sha256':sha(a.corpus),'retrieval':'title-anchor + BM25; query only, no answers'}
 create(a.output/'excluded_tasks.json',[x for x in d if not x['budget_pressure']]);create(a.output/'tasks.json',out);create(a.output/'split_commitment.json',c);create(a.output/'retrieval_diagnostics.json',d)
 for name,parts in [('training',{'fit','calibration'}),('diagnostic',{'diagnostic'})]:
  subset=[t for t in out if c['task_split'][t['id']]in parts and len(t['source_catalog'])>5];create(a.output/name/'tasks.json',subset);create(a.output/name/'split_commitment.json',{**c,'task_split':{t['id']:c['task_split'][t['id']]for t in subset}})
 print(json.dumps({'tasks':len(out),'partitions':dict(collections.Counter(c['task_split'].values())),'tasks_with_budget_pressure':sum(x['budget_pressure']for x in d),'insufficient_tasks':[x['task_id']for x in d if not x['budget_pressure']],'model_api_calls':0},indent=2))
 if any(not json.loads((a.output/name/'tasks.json').read_text()) for name in ('training','diagnostic')):raise SystemExit('empty coverage-qualified partition; no fallback')
