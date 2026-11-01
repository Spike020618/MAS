"""Read-only alignment of learned/sparse baselines with SEALED LLM cohorts.

All cohorts are already-used development labels. No provider construction or
unsealed focused-transfer scoring. Global RRF and shared-pool RRF are separate.
"""
import json
from pathlib import Path
from experiments.paired_deliberation.native_redial_benchmark import ROOT,evaluate,save
from experiments.paired_deliberation.native_open_evidence_analysis import load_sealed
from experiments.paired_deliberation.native_kbrd_scores import PrecomputedKBRD

RESULT=ROOT/'experiments/results/native_kbrd_reproduction_v1'

def run():
    k={r['id']:r['ranking'] for r in json.loads((RESULT/'train_recall_r1/export/valid.predictions.json').read_text())}
    s={r['id']:r['ranking'] for r in json.loads((ROOT/'experiments/results/native_retrieval_inner_v1/valid.predictions.json').read_text())}
    rrf={r['id']:r['ranking'] for r in json.loads((RESULT/'fixed_rrf60.valid.predictions.json').read_text())}
    lookup=PrecomputedKBRD(RESULT/'train_recall_r1/export')
    output={'scope':'read-only posthoc diagnostic on already used development labels; no tuning, no new model calls',
        'ranking_selected_epoch':9,'CE_selected_epoch':5,
        'shared_rrf_definition':'same truncated KBRDtop100 + Sparsetop100 equal RRF60 rule restricted to the actual shared Sparse100 pool',
        'global_vs_shared_limit':'global baselines may propose outside LLM pool; shared versions give exact same candidate permissions',
        'cohorts':{},'deferred_unsealed':['native_open_evidence_focused_transfer_v1']}
    for name in ('native_open_evidence_screen_v2','native_open_evidence_focused_screen_v1','native_open_evidence_transfer_v1'):
        root=ROOT/'experiments/results'/name;plan,inputs,cases=load_sealed(root)
        labels_by_id={r['id']:r for r in json.loads(Path(plan['label_source']).read_text())};labels=[labels_by_id[r['id']] for r in inputs]
        ranks={'KBRD_global':[],'Sparse_shared':[],'fixed_RRF60_global':[],'KBRD_shared':[],'fixed_RRF60_shared':[]};case_info=[]
        for row in inputs:
            cid=row['id'];pool=[int(m['movie_id']) for m in row['candidates']];assert pool==s[cid]
            native=lookup.score_pool(cid,list(map(str,pool)));scores={int(m):logit for m,logit,z,rank in native['rows']}
            shared_k=sorted(pool,key=lambda m:(-scores[m],m));sk={m:1/(60+i+1) for i,m in enumerate(k[cid])}
            ss={m:1/(60+i+1) for i,m in enumerate(pool)};shared_rrf=sorted(pool,key=lambda m:(-(ss[m]+sk.get(m,0)),m))
            for method,pred in [('KBRD_global',k[cid]),('Sparse_shared',pool),('fixed_RRF60_global',rrf[cid]),('KBRD_shared',shared_k),('fixed_RRF60_shared',shared_rrf)]:ranks[method].append(pred)
            case_info.append({'id':cid,'pool':pool,'k_shared':shared_k,'rrf_shared':shared_rrf})
        baseline={method:evaluate(labels,predictions) for method,predictions in ranks.items()}
        llm={};diagnostics={}
        for method in plan['methods']:
            predictions=[list(map(int,c['methods'][method]['ranking'])) for c in cases];llm[method]=evaluate(labels,predictions)
            counts={key:0 for key in ('llm_hits','llm_misses','llm_misses_KBRD_shared_top10','llm_misses_Sparse_top10',
                'llm_misses_either_shared_top10','llm_misses_only_KBRD_shared_top10','llm_misses_only_Sparse_top10',
                'llm_misses_both_shared_top10','llm_misses_global_union100','llm_misses_global_union100_outside_actual_pool',
                'shared_rrf_gained_vs_llm','shared_rrf_lost_vs_llm','global_rrf_gained_vs_llm','global_rrf_lost_vs_llm')}
            details=[]
            for i,(row,pred,info) in enumerate(zip(labels,predictions,case_info)):
                for t in row['targets']:
                    hit=t in pred[:10];kh=t in ranks['KBRD_shared'][i][:10];sh=t in ranks['Sparse_shared'][i][:10]
                    rh=t in ranks['fixed_RRF60_shared'][i][:10];rg=t in ranks['fixed_RRF60_global'][i][:10]
                    counts['llm_hits' if hit else 'llm_misses']+=1
                    if not hit:
                        counts['llm_misses_KBRD_shared_top10']+=kh;counts['llm_misses_Sparse_top10']+=sh
                        counts['llm_misses_either_shared_top10']+=kh or sh;counts['llm_misses_only_KBRD_shared_top10']+=kh and not sh
                        counts['llm_misses_only_Sparse_top10']+=sh and not kh;counts['llm_misses_both_shared_top10']+=kh and sh
                        covered=t in set(k[row['id']])|set(s[row['id']]);counts['llm_misses_global_union100']+=covered
                        counts['llm_misses_global_union100_outside_actual_pool']+=covered and t not in info['pool']
                    counts['shared_rrf_gained_vs_llm']+=rh and not hit;counts['shared_rrf_lost_vs_llm']+=hit and not rh
                    counts['global_rrf_gained_vs_llm']+=rg and not hit;counts['global_rrf_lost_vs_llm']+=hit and not rg
                    details.append({'id':row['id'],'target':t,'llm_hit@10':hit,'KBRD_shared_hit@10':kh,'Sparse_hit@10':sh,
                        'RRF_shared_hit@10':rh,'RRF_global_hit@10':rg,'in_actual_pool':t in info['pool']})
            diagnostics[method]={'counts':counts,'among_llm_misses_shared_union10_recoverable_fraction':counts['llm_misses_either_shared_top10']/max(counts['llm_misses'],1),
                'details':details}
        output['cohorts'][name]={'ids':plan['ids'],'contexts':len(inputs),'targets':sum(len(r['targets']) for r in labels),
            'baseline_metrics':baseline,'llm_metrics':llm,'complementarity_vs_each_llm':diagnostics,
            'generation_seal_verified':True,'candidate_pool_exact_sparse_match':True}
    save(RESULT/'baseline_alignment.json',output)
    print(json.dumps({name:{'contexts':v['contexts'],'targets':v['targets'],'R10':{m:score['metrics']['recall@10'] for m,score in v['baseline_metrics'].items()},'LLM_R10':{m:score['metrics']['recall@10'] for m,score in v['llm_metrics'].items()}} for name,v in output['cohorts'].items()},indent=2))
    return output
if __name__=='__main__':run()
