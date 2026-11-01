"""Frozen-prediction INSPIRED diagnostics; no fitting/model calls/test access."""
import json,hashlib
from pathlib import Path
from experiments.consensus_recommendation.common import ROOT
from experiments.paired_deliberation.native_inspired_benchmark import evaluate
from experiments.paired_deliberation.native_consensus_analysis import cluster_difference

RESULT=ROOT/'experiments/results/native_inspired_strong_v1'

def run():
    r=RESULT;labels=json.loads((r/'valid.labels.json').read_text());inputs=json.loads((r/'valid.inputs.json').read_text())
    sparse=json.loads((r/'sparse/fixed_redial_transfer/valid.predictions.json').read_text());tf=json.loads((ROOT/'experiments/results/native_inspired_benchmark_v1/valid.predictions.json').read_text());out={}
    for kv in ('academic_pure','train_inner_tuned'):
        neural=json.loads((r/'kbrd'/kv/'valid.predictions.json').read_text());global_rr=[];same_rr=[];u10=u100=0
        for row,s,k in zip(labels,sparse,neural):
            assert row['id']==s['id']==k['id'];sm={m:1/(60+i+1) for i,m in enumerate(s['ranking'])};km={m:1/(60+i+1) for i,m in enumerate(k['ranking'])};pool=set(sm)|set(km)
            key=lambda m:(-(sm.get(m,0)+km.get(m,0)),m)
            global_rr.append(sorted(pool,key=key)[:100]);same_rr.append(sorted(s['ranking'],key=key))
            u10+=sum(t in set(s['ranking'][:10])|set(k['ranking'][:10]) for t in row['targets']);u100+=sum(t in pool for t in row['targets'])
        out[kv]={'equal_RRF60_global':evaluate(labels,global_rr),'equal_RRF60_same_Sparse100':evaluate(labels,same_rr),
            'oracle_union10_count':u10,'oracle_union100_count':u100,'same_pool_interpretation':'only original fixed-transfer Sparse100 eligible; global permits two-source union'}
        (r/('fixed_RRF60_'+kv+'.valid.predictions.json')).write_text(json.dumps([{'id':row['id'],'ranking':pred} for row,pred in zip(labels,global_rr)],indent=2)+'\n')
    out['fixed_recipe_minus_TFIDF_cluster_interval']=cluster_difference(labels,[x['ranking'] for x in sparse],[x['ranking'] for x in tf])
    novel=[];ni=[]
    for i,(inp,row) in enumerate(zip(inputs,labels)):
        targets=[t for t in row['targets'] if t not in inp['context_movies']]
        if targets:novel.append({'id':row['id'],'targets':targets});ni.append(i)
    out['novel_targets_fixed_recipe_minus_TFIDF_cluster_interval']=cluster_difference(novel,[sparse[i]['ranking'] for i in ni],[tf[i]['ranking'] for i in ni])
    out['scope']='read-only posthoc development diagnostics; no public test, model calls, or selected blend weights'
    out['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (r/'heterogeneous_baseline_analysis.json').write_text(json.dumps(out,indent=2)+'\n');return out
if __name__=='__main__':print(json.dumps(run(),indent=2))
