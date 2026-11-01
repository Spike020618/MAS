"""Full sealed training-side ranking opportunity counts, never truncated candidate counts."""
import collections,json
from pathlib import Path
from experiments.consensus_recommendation.common import ROOT,create,fingerprint
from experiments.consensus_mechanism import core as c
from experiments.consensus_mechanism.data import read,labels_for

def audit(root):
    tasks=read(root/'inputs.json');cases=[read(root/'cases'/f'case_{i}.json')for i in range(len(tasks))]
    if read(root/'generation_complete.json')!={'task_ids':[t['id']for t in tasks],'hashes':[fingerprint(x)for x in cases]}:raise ValueError('training seal mismatch')
    labels=labels_for(tasks,root/'generation_complete.json');rows=[];counts=collections.Counter()
    for t,case in zip(tasks,cases):
        if t['kind']!='rec':continue
        ranking=case['baseline']['decision'];targets=set(labels[t['id']]['targets']);pool=set(c.candidate_ids(t))
        outside=sorted(targets& (pool-set(ranking[:10])))
        quality=c.quality(ranking,labels[t['id']])
        row={'task_id':t['id'],'initial_quality':quality,'pool_size':len(pool),
             'targets_in_pool':sorted(targets&pool),'targets_top10':sorted(targets&set(ranking[:10])),
             'outside_top10_in_pool':outside,'mechanical_recall_headroom':bool(outside),
             'below_first_in_top10':sorted(targets&set(ranking[1:10]))}
        rows.append(row);counts['recommendation_tasks']+=1
        counts['initial_zero']+=quality==0;counts['initial_partial']+=0<quality<1;counts['initial_full']+=quality==1
        counts['mechanical_recall_headroom']+=bool(outside)
        counts['zero_quality_with_headroom']+=bool(outside)and quality==0
        counts['partial_quality_with_headroom']+=bool(outside)and 0<quality<1
        counts['ranking_position_headroom_proxy']+=bool(row['below_first_in_top10'])
    return {'counts':dict(counts),'rows':rows,'seal_verified':True,'model_api_calls':0,
            'scope':'Full training-side mechanical action space; gold-selected opportunities are not an unbiased test or semantic evidence certification.'}

if __name__=='__main__':
    root=ROOT/'experiments/results/consensus_mechanism_paper_v2/train'
    result=audit(root);create(ROOT/'experiments/results/recommendation_headroom_audit_v1/summary.json',result)
    print(json.dumps({k:v for k,v in result.items()if k!='rows'},ensure_ascii=False,indent=2))
