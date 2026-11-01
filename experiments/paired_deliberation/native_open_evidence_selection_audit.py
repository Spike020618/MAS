"""Post-seal oracle diagnostics of proposal/allocation, never runtime decisions."""
import argparse,collections,json,time
from pathlib import Path
from experiments.consensus_recommendation.common import ROOT,create,sha
from experiments.paired_deliberation.native_open_evidence_analysis import load_sealed,resolve_receipt
from experiments.paired_deliberation.client_v5 import parse_raw


def analyze(root):
    root=Path(root);plan,inputs,cases=load_sealed(root)
    labels={r['id']:r['targets'] for r in json.loads(Path(plan['label_source']).read_text())}
    counts=collections.Counter();per_method={m:collections.Counter() for m in ('central_challenge','game_challenge')};outside={m:collections.Counter() for m in per_method};examples=[]
    for i,(row,case) in enumerate(zip(inputs,cases)):
        gold=collections.Counter(map(str,labels[row['id']]));pool={m['movie_id'] for m in row['candidates']};union={m for call in case['initial'] for m in call['ranking'][:10]};base=case['methods']['independent3']['ranking']
        counts['item_targets']+=sum(gold.values());counts['pool_covered']+=sum(v for t,v in gold.items() if t in pool);counts['initial_union_covered']+=sum(v for t,v in gold.items() if t in union)
        base_hits=sum(gold[t] for t in base[:10]);counts['initial3_top10_hits']+=base_hits
        commit=json.loads((root/'commitments'/('case'+str(i)+'_central.json')).read_text());targets=commit['targets']
        counts['all_proposed_b_target_coverage']+=sum(v for t,v in gold.items() if t in {q['b'] for q in targets})
        improvements=sorted([max(0,gold[t['b']]-gold[t['a']]) for t in targets],reverse=True)
        counts['all_proposed_best_three_query_oracle_hits']+=base_hits+sum(improvements[:3])
        for method in per_method:
            events=case['methods'][method]['events'];selected={e['target']['b'] for e in events};novel={e['target']['b']for e in events if e['target']['novel_to_initial_union']};a=per_method[method]
            a['selected_b_target_coverage']+=sum(v for t,v in gold.items() if t in selected)
            a['selected_novel_b_target_coverage']+=sum(v for t,v in gold.items() if t in novel)
            a['selected_action_oracle_top10_hits']+=base_hits+sum(max(0,gold[e['target']['b']]-gold[e['target']['a']]) for e in events)
            a['observed_final_top10_hits']+=sum(gold[t] for t in case['methods'][method]['ranking'][:10])
            for e in events:
                call=json.loads((root/'logical_calls'/(e['slot']+'.json')).read_text());receipt=resolve_receipt(root,call['physical_slot'],call['request'])
                if receipt['status']!='returned':continue
                try:report=parse_raw(receipt['wrapper']['raw'],call['request'])['report']
                except (ValueError,KeyError,TypeError):continue
                if not isinstance(report,dict):continue
                preferred=report.get('preferred');preferred=str(preferred) if type(preferred)in(str,int) else None
                if preferred is None or preferred in (e['target']['a'],e['target']['b']):continue
                outside[method]['out_of_pair_preferred_responses']+=1
                if preferred in pool:outside[method]['out_of_pair_preferred_in_pool']+=1
                if preferred in gold:
                    outside[method]['out_of_pair_preferred_true_target_responses']+=1
                    if preferred not in union:
                        outside[method]['true_shared_omission_proposed_outside_pair']+=1;examples.append({'method':method,'case':case['id'],'movie_id':preferred,'logical_slot':e['slot'],'source_id':report.get('source_id')})
    result={'analysis_source_sha256':sha(Path(__file__)),'generation_sealed_before_labels':True,'scope':'retrospective labels-only diagnostic; no gold used in generation, not a runnable baseline',
        'coverage':dict(counts),'per_method':{m:dict(v) for m,v in per_method.items()},'rejected_out_of_pair_diagnostic':{m:dict(v) for m,v in outside.items()},'shared_omission_examples':examples,
        'oracle_rule':'disjoint public Top10/outside pairs; retain incumbent or swap challenger; sum positive target-count differences, all-proposed oracle may select best3; semantic/format constraints ignored',
        'interpretation':'if selected_b_target_coverage=0, even perfect pair judgements cannot add true hits; this diagnoses proposal/allocation rather than verifier accuracy'}
    create(root/'selection_audit.json',result);print(json.dumps({'root':str(root),'coverage':result['coverage'],'per_method':result['per_method'],'out_of_pair':result['rejected_out_of_pair_diagnostic']},ensure_ascii=False),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',action='append',required=True);p.add_argument('--watch',action='store_true');a=p.parse_args();remaining=[Path(r) for r in a.root]
    while remaining:
        for root in list(remaining):
            if not(root/'generation_complete.json').exists():continue
            if not(root/'selection_audit.json').exists():analyze(root)
            remaining.remove(root)
        if not a.watch:break
        if remaining:time.sleep(10)
