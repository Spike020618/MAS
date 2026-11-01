"""Prepare new source-disjoint acquisition inputs; no labels, APIs or old rewrites."""
import argparse,copy,hashlib,json,re
from pathlib import Path
from urllib.parse import urlsplit,urlunsplit
from experiments.consensus_recommendation.common import create,fingerprint,sha
PARTS=('fit','calibration','diagnostic')
def partition(key):
 x=int(hashlib.sha256(('disjoint-v1:'+key).encode()).hexdigest(),16)%10
 return 'fit'if x<6 else 'calibration'if x<8 else 'diagnostic'
def group(t):
 n=t['metadata']['native_id'];return t['kind']+':'+(n.rsplit(':',1)[0]if t['kind']=='rec'else n)
def keys(s,mode):
 out={'text:'+hashlib.sha256(' '.join(s['text'].split()).encode()).hexdigest()}
 if mode=='document':
  for k in ('url','source_url','original_url'):
   if s.get(k):
    u=urlsplit(s[k]);out.add('url:'+urlunsplit((u.scheme.lower(),u.netloc.lower(),u.path.rstrip('/'),u.query,'')))
  for k in ('document_id','source_doc','original_document_id'):
   if s.get(k):out.add('doc:'+str(s[k]))
 return out
def build(tasks,mode='document',topk=3):
 if topk<1:raise ValueError('positive topk required')
 raw=[s for t in tasks for s in t['source_catalog'].values()]
 def transferable(s):
  return s.get('kind') not in ('dialogue','model_forecast') and not s.get('current_user_fact_authority',False) and s.get('scope',{}).get('subject') not in ('current_user','current_context','model_forecast')
 sources=[copy.deepcopy(s)for s in raw if transferable(s)];excluded=len(raw)-len(sources);parent=list(range(len(sources)));owner={}
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 for i,s in enumerate(sources):
  for k in keys(s,mode):
   if k in owner:parent[find(i)]=find(owner[k])
   owner[k]=i
 components={}
 for i,s in enumerate(sources):components.setdefault(find(i),[]).append(s)
 libraries={p:[]for p in PARTS};identity={}
 for ss in components.values():
  ident=fingerprint(sorted(set(k for s in ss for k in keys(s,mode))));p=partition(ident);seen=set()
  for s in ss:
   h=fingerprint(sorted(keys(s,mode)))
   if h not in seen:libraries[p].append(s);seen.add(h)
  identity[ident]=p
 diagnostic={'excluded_query_specific_sources':excluded,'mode':mode,'source_components':len(components),'library_sizes':{p:len(v)for p,v in libraries.items()},'source_partition':identity}
 if any(not libraries[p]for p in PARTS):return None,{**diagnostic,'error':'insufficient independent source components; acquire more documents; no fallback'}
 output=[];splits={}
 for t in tasks:
  p=partition(group(t));splits[t['id']]=p;q=set(re.findall(r'[a-z]{3,}',t['prompt'].lower()))
  # Rank using public title/kind only; no source body or future label relevance.
  ranked=sorted(libraries[p],key=lambda s:(-len(q&set(re.findall(r'[a-z]{3,}',(s.get('title','')+' '+s.get('kind','')).lower()))),fingerprint(sorted(keys(s,mode)))))
  t=copy.deepcopy(t);catalog={}
  for i,s in enumerate(ranked[:topk]):
   s=copy.deepcopy(s);sid='disjoint:'+fingerprint(sorted(keys(s,mode)))[:24];s['origin_source_id']=s.get('source_id');s['origin_scope']=copy.deepcopy(s.get('scope',{}));s['source_id']=sid;s['scope']={**s.get('scope',{}),'task_id':t['id'],'subject':'other_context'};s['current_user_fact_authority']=False;s['allowed_uses']=['prediction_basis'];catalog[sid]=s
  t['source_catalog']=catalog;t['metadata']['acquisition_partition']=p;output.append(t)
 return output,{**diagnostic,'task_split':splits,'scope':'new acquisition required; metadata retrieval does not imply factual support'}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--identity',choices=['document','content'],default='document');p.add_argument('--topk',type=int,default=3);a=p.parse_args();tasks=json.loads(a.inputs.read_text());out,commit=build(tasks,a.identity,a.topk);commit['inputs_sha256']=sha(a.inputs);create(a.output/'split_commitment.json',commit)
 if out is None:raise SystemExit(commit['error'])
 create(a.output/'tasks.json',out);print(json.dumps({'tasks':len(out),'library_sizes':commit['library_sizes'],'model_api_calls':0},indent=2))
if __name__=='__main__':main()
