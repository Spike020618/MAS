"""Read-only source audit of sealed new64 predictions; writes derived report only."""
import hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def load(p):return json.loads(p.read_text())
def run():
    r=ROOT/'experiments/results/native_heterogeneous_new64_v1'
    labels_path=ROOT/'experiments/results/native_retrieval_inner_v1/new64.labels.json'
    labels={x['id']:[str(t) for t in x['targets']] for x in load(labels_path)}
    paths=sorted((r/'cases').glob('case_*.json'))
    cases=[load(p) for p in paths]
    assert len(cases)==64 and len({c['id'] for c in cases})==64
    calpath=ROOT/'experiments/results/native_calibrated_transfer_v1/new64.inputs.json'
    cal={x['id']:x for x in load(calpath)}
    methods=sorted(cases[0]['methods'])+['calibrated_count']
    scores={m:[] for m in methods};den=[];groups=[];events=[]
    for c in cases:
        targets=labels[c['id']];den.append(len(targets));groups.append(c['id'].rsplit(':',1)[0])
        for m in methods:
            ranking=([str(i['movie_id']) for i in cal[c['id']]['candidates']] if m=='calibrated_count' else c['methods'][m]['ranking'])
            assert len(set(ranking))==len(ranking)
            scores[m].append([int(t in ranking[:10]) for t in targets])
    total=sum(den);rng=np.random.default_rng(20261028)
    unique=sorted(set(groups));idx={g:[i for i,v in enumerate(groups) if v==g] for g in unique}
    bs=rng.integers(0,len(unique),size=(20000,len(unique)))
    gden=np.array([sum(den[i] for i in idx[g]) for g in unique])
    def compare(a,b):
        gain=loss=0;detail=[];gd=[]
        for i,c in enumerate(cases):
            aa=np.array(scores[a][i]);bb=np.array(scores[b][i]);plus=int(((aa==1)&(bb==0)).sum());minus=int(((aa==0)&(bb==1)).sum())
            gain+=plus;loss+=minus
            if plus or minus:detail.append({'id':c['id'],'targets':labels[c['id']],'new_hits':plus,'lost_hits':minus})
        for g in unique:gd.append(sum(sum(scores[a][i])-sum(scores[b][i]) for i in idx[g]))
        estimates=np.array(gd)[bs].sum(1)/gden[bs].sum(1)
        return {'new_hits':gain,'lost_hits':loss,'net_hits':gain-loss,'recall_difference':(gain-loss)/total,'conversation_cluster_bootstrap95':np.quantile(estimates,[.025,.975]).tolist(),'changed_hit_cases':detail}
    comparisons={a+'_vs_'+b:compare(a,b) for a,b in [('game_challenge','hetero_base'),('central_challenge','hetero_base'),('game_challenge','single4'),('central_challenge','single4'),('game_challenge','central_challenge'),('game_challenge','calibrated_count')]}
    different_rank=[];different_actions=[];different_top10=[]
    for c in cases:
        a=c['methods']['game_challenge'];b=c['methods']['central_challenge']
        if a['ranking']!=b['ranking']:different_rank.append(c['id'])
        if a['ranking'][:10]!=b['ranking'][:10]:different_top10.append(c['id'])
        if a['allocation']['actions']!=b['allocation']['actions']:different_actions.append(c['id'])
    report={'scope':'sealed development valid; no API calls; duplicate target occurrences retained; bootstrap conversations with seed 20261028, 20000 draws; not a pristine confirmation','contexts':len(cases),'conversations':len(unique),'item_targets':total,'methods':{m:{'hits':sum(map(sum,scores[m])),'recall@10':sum(map(sum,scores[m]))/total} for m in methods},'comparisons':comparisons,'game_central_difference':{'full_ranking_cases':different_rank,'top10_order_cases':different_top10,'allocated_action_cases':different_actions},'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[labels_path,calpath]}}
    out=r/'sealed_independent_analysis.json'
    if out.exists():
        assert load(out)==report, 'sealed audit differs: write a new audit version'
    else:
        out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},ensure_ascii=False,indent=2))
if __name__=='__main__':run()
