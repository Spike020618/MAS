"""Offline accounting of available correct votes vs readout loss in frozen suites.

No correctness oracle is deployed, no label-based routing is claimed, and model
bundle choices are observational rather than randomized backend interventions.
"""
from collections import Counter,defaultdict
from pathlib import Path
import json

from experiments.mechanism.attribute_legacy_gain import ROOT,vote,trace_kind
from experiments.mechanism.audit_legacy_conclusion_parsing import conclusion
from experiments.mechanism.run_upgrade_pilot import freeze
from experiments.mechanism.study_information_complementarity import fingerprint

OUT=ROOT/'experiments/results/audit/correct_minority_2026-10-24'


def account(rows,mode):
    grouped=defaultdict(list);losses=[]
    for row in rows:
        rounds=sorted({t['round_id'] for t in row['rounds']})
        for rid in sorted({rounds[0],rounds[-1]}):
            turns=[t for t in row['rounds'] if t['round_id']==rid]
            assert len({t['agent_id'] for t in turns})==len(turns)
            stances=[t['stance'] if mode=='original' else conclusion(t['report']['C'],t['stance'],mode) for t in turns]
            correct=[s==row['ground_truth'] for s in stances]
            selected=vote(stances)
            item={'dataset':row['dataset'],'question_id':str(row['question_id']),
                  'gold':row['ground_truth'],'round':rid,'decision':selected,'correct':selected==row['ground_truth'],
                  'coverage':any(correct),'agents':len(turns),'correct_votes':sum(correct),
                  'correct_routes':dict(Counter(trace_kind(t) for t,ok in zip(turns,correct) if ok)),
                  'wrong_routes':dict(Counter(trace_kind(t) for t,ok in zip(turns,correct) if not ok)),
                  'all_routes':dict(Counter(trace_kind(t) for t in turns))}
            grouped[(row['condition'],row['seed'],rid)].append(item)
            if item['coverage'] and not item['correct']:
                losses.append({**item,'condition':row['condition'],'seed':row['seed'],
                               'reports':[{'agent_id':t['agent_id'],'stance':s,'route':trace_kind(t),'report':t['report']} for t,s in zip(turns,stances)]})
    result=[]
    for (condition,seed,rid),items in sorted(grouped.items()):
        coverage=sum(r['coverage'] for r in items);correct=sum(r['correct'] for r in items)
        loss=sum(r['coverage'] and not r['correct'] for r in items)
        assert coverage-loss==correct
        result.append({'condition':condition,'seed':seed,'round':rid,'n':len(items),'any_correct':coverage,
            'majority_correct':correct,'readout_loss':loss,'all_wrong':len(items)-coverage,
            'correct_vote_distribution':dict(Counter(str(r['correct_votes']) for r in items)),
            'loss_correct_routes':dict(sum((Counter(r['correct_routes']) for r in items if r['coverage'] and not r['correct']),Counter())),
            'loss_all_routes':dict(sum((Counter(r['all_routes']) for r in items if r['coverage'] and not r['correct']),Counter())),
            'by_dataset':{d:{'n':sum(r['dataset']==d for r in items),
                'any_correct':sum(r['coverage'] for r in items if r['dataset']==d),
                'majority_correct':sum(r['correct'] for r in items if r['dataset']==d),
                'readout_loss':sum(r['coverage'] and not r['correct'] for r in items if r['dataset']==d)}
                for d in sorted({r['dataset'] for r in items})}})
    return {'summary':result,'lost_correct_cases':losses}


def main():
    suites={};hashes={}
    for path in sorted((ROOT/'experiments/results/mechanism/frozen').glob('*/question_level.jsonl')):
        if path.parent.name.endswith('.recomputed'):continue
        rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        assert all(r['completed'] and not r.get('error') for r in rows)
        hashes[str(path.relative_to(ROOT))]=fingerprint(path)
        suites[path.parent.name]={mode:account(rows,mode) for mode in ['original','literal_yes_no_only','unresolved_placeholder_abstains']}
    OUT.mkdir(parents=True,exist_ok=True)
    result={'scope':'bounded frozen-report accounting, not new validation or causal gain attribution',
            'new_model_calls':0,'suites':suites,'input_sha256':hashes,
            'limits':['source duplicates and changed execution bundles remain; do not pool suites as independent questions',
                'available correct votes use labels offline; not an implemented selector',
                'conclusion parser sensitivities do not replay past rewards or reputation',
                'low readout accuracy does not prove that existing explanations identify correct minority',
                'no cross-study comparison with new native MC as one estimand']}
    freeze(OUT/'result.json',result)
    print('historical suites audited',len(suites),'API calls 0')


if __name__=='__main__':main()
