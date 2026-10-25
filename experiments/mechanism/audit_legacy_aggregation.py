"""Saved-report voting diagnostics; no selected subset claimed as new policy."""
from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
from math import sqrt

from experiments.mechanism.attribute_legacy_gain import ROOT, OUT, vote, transitions


def phi(left,right):
    if len(left)!=len(right) or not left:
        raise ValueError('paired nonempty indicators required')
    a = sum(x and y for x,y in zip(left,right))
    b = sum(x and not y for x,y in zip(left,right))
    c = sum(not x and y for x,y in zip(left,right))
    d = sum(not x and not y for x,y in zip(left,right))
    denominator = sqrt((a+b)*(c+d)*(a+c)*(b+d))
    return {'both_error':a,'only_left_error':b,'only_right_error':c,'both_correct':d,
            'phi':(a*d-b*c)/denominator if denominator else None}


def analyze(rows):
    n = len(rows)
    results = []
    for rid in [1,3]:
        matrix = [[next(t for t in r['rounds'] if t['round_id']==rid and t['agent_id']==aid)['stance']
                   for aid in range(5)] for r in rows]
        gold = [r['ground_truth'] for r in rows]
        majority = [vote(v)==g for v,g in zip(matrix,gold)]
        counts = [sum(v==g for v in votes) for votes,g in zip(matrix,gold)]
        singles = []
        for aid in range(5):
            correct = [v[aid]==g for v,g in zip(matrix,gold)]
            singles.append({'agent':aid, **transitions(correct,majority)})
        pairs = []
        for a,b in combinations(range(5),2):
            pairs.append({'agents':[a,b],**phi([v[a]!=g for v,g in zip(matrix,gold)],
                                               [v[b]!=g for v,g in zip(matrix,gold)])})
        triples = []
        for ids in combinations(range(5),3):
            outcomes = [vote([v[a] for a in ids])==g for v,g in zip(matrix,gold)]
            triples.append({'agents':list(ids),**transitions(outcomes,majority)})
        expected_single = sum(counts)/5
        results.append({'round':rid,'n':n,'majority_correct':sum(majority),
            'uniform_random_single_expected_correct':expected_single,
            'majority_minus_random_single_expected_correct':sum(majority)-expected_single,
            'correct_vote_count_distribution':dict(sorted(Counter(counts).items())),
            'any_correct_available':sum(c>0 for c in counts),
            'correct_available_but_majority_wrong':sum(c>0 and not ok for c,ok in zip(counts,majority)),
            'all_wrong':sum(c==0 for c in counts),
            'fixed_single_comparisons_all_unselected':singles,
            'fixed_triple_comparisons_all_unselected':triples,
            'pairwise_error_indicators_across_questions':pairs})
    all15 = [vote([t['stance'] for t in r['rounds']])==r['ground_truth'] for r in rows]
    return {'rounds':results,'all15_plurality_correct':sum(all15),
            'n':n,'scope':'same realized reports, no altered prompts/history, not independently executed single/triple/early-stop policies'}


def decomposition(left,right):
    """Accounting identity, NOT identified mediation or causal attribution."""
    if left['n']!=right['n']:
        raise ValueError('matched denominators required')
    n = left['n']
    majority_gain = right['majority_correct']-left['majority_correct']
    individual_gain = right['uniform_random_single_expected_correct']-left['uniform_random_single_expected_correct']
    surplus_gain = right['majority_minus_random_single_expected_correct']-left['majority_minus_random_single_expected_correct']
    if abs(majority_gain-individual_gain-surplus_gain)>1e-9:
        raise ValueError('decomposition identity failed')
    return {'n':n,'majority_net_correct_gain':majority_gain,
        'mean_individual_correct_gain':individual_gain,'voting_surplus_gain':surplus_gain,
        'majority_gain_pp':100*majority_gain/n,'mean_individual_gain_pp':100*individual_gain/n,
        'voting_surplus_gain_pp':100*surplus_gain/n,
        'scope':'algebraic partition of observed difference, NOT percent caused by each mechanism component'}


def main():
    path = ROOT/'experiments/results/mechanism/frozen/formal_v10/question_level.jsonl'
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    groups = {c:[r for r in rows if r['condition']==c] for c in sorted({r['condition'] for r in rows})}
    r = {'no_api_calls':True, 'overall':{c:analyze(rs) for c,rs in groups.items()},
         'by_domain':{d:{c:analyze([r for r in rs if r['dataset']==d]) for c,rs in groups.items()}
                      for d in sorted({r['dataset'] for r in rows})},
         'source_sha256':sha256(path.read_bytes()).hexdigest(),
         'limits':['do not select the highest-scoring persona/triple using these held-out outcomes',
                   'voting vs uniformly random saved single report is an algebraic expectation, not extra API sampling',
                   'error phi across questions mixes common difficulty, persona and backend composition; no causal dependence claim',
                   'oracle any-correct count uses gold offline and is not a realizable routing strategy',
                   'dropping reports/rounds changes future reputation evolution if actually deployed; no rollout saving established']}
    r['static_vs_none_accounting_decomposition'] = [decomposition(a,b) for a,b in zip(
        r['overall']['independent__no_mechanism@0.2']['rounds'],r['overall']['independent__static_aeic@0.2']['rounds'])]
    (OUT/'aggregation.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({c:{'rounds':[{k:v for k,v in rr.items() if k not in ['fixed_single_comparisons_all_unselected','fixed_triple_comparisons_all_unselected','pairwise_error_indicators_across_questions']} for rr in s['rounds']],
                          'all15_correct':s['all15_plurality_correct']} for c,s in r['overall'].items()},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
