"""Read-only validation diagnostics, no fit/test calls or runtime oracle use."""
import json
from pathlib import Path
from experiments.paired_deliberation.native_redial_benchmark import ROOT,evaluate,save

RESULT=ROOT/'experiments/results/native_kbrd_reproduction_v1'

def run():
    labels=json.loads((ROOT/'experiments/results/native_redial_benchmark_v1/valid.labels.json').read_text())
    k=json.loads((RESULT/'train_recall_r1/export/valid.predictions.json').read_text());s=json.loads((ROOT/'experiments/results/native_retrieval_inner_v1/valid.predictions.json').read_text())
    km={r['id']:r['ranking'] for r in k};sm={r['id']:r['ranking'] for r in s};ids=[r['id'] for r in labels]
    ranks={};out={}
    for cid in ids:
        a=km[cid];b=sm[cid];sa={m:1/(60+i+1) for i,m in enumerate(a)};sb={m:1/(60+i+1) for i,m in enumerate(b)}
        union=set(a)|set(b);ranks[cid]=sorted(union,key=lambda m:(-(sa.get(m,0)+sb.get(m,0)),m))[:100]
    old_ids={r['id'] for r in json.loads((ROOT/'experiments/results/native_retrieval_inner_v1/old32.inputs.json').read_text())}
    new_ids={r['id'] for r in json.loads((ROOT/'experiments/results/native_retrieval_inner_v1/new64.inputs.json').read_text())}
    for name,chosen in [('full_valid',set(ids)),('old32',old_ids),('new64',new_ids)]:
        rows=[r for r in labels if r['id'] in chosen];total=sum(len(r['targets']) for r in rows);counts={}
        for cutoff in (10,100):
            count=sum(t in set(km[r['id']][:cutoff])|set(sm[r['id']][:cutoff]) for r in rows for t in r['targets'])
            counts['oracle_union_top'+str(cutoff)+'_hits']=count;counts['oracle_union_top'+str(cutoff)+'_coverage']=count/total
        counts.update(targets=total,kbrd=evaluate(rows,[km[r['id']] for r in rows]),sparse=evaluate(rows,[sm[r['id']] for r in rows]),
            fixed_rrf60=evaluate(rows,[ranks[r['id']] for r in rows]));out[name]=counts
    out['scope']='read-only reused validation diagnostic; union is retrospective upper bound, not runtime prediction; fixed equal RRF60 has no validation tuning but is posthoc exploratory'
    save(RESULT/'complementarity.json',out);save(RESULT/'fixed_rrf60.valid.predictions.json',[{'id':cid,'ranking':ranks[cid]} for cid in ids])
    print(json.dumps(out,indent=2));return out
if __name__=='__main__':run()
