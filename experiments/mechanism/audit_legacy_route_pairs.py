"""Matched question/persona/round report changes by observed execution bundle.

Post-hoc accounting, not a randomized backend intervention or off-policy value
estimate. Backend, cap, instruction and self-check travel together historically.
"""
from collections import defaultdict
from hashlib import sha256
import json

from experiments.mechanism.attribute_legacy_gain import ROOT, OUT, trace_kind
from experiments.mechanism.audit_legacy_conclusion_parsing import conclusion


def partition(rows,mode='original',round_id=None):
    index = defaultdict(dict)
    for r in rows:
        c = r['condition'].split('__')[1].split('@')[0]
        if c not in ['no_mechanism','static_aeic']:continue
        for t in r['rounds']:
            if round_id is not None and t['round_id']!=round_id:continue
            key = (r['dataset'],str(r['question_id']),t['round_id'],t['agent_id'])
            if c in index[key]:raise ValueError('duplicate agent slot')
            index[key][c] = (r['ground_truth'],t)
    cells = defaultdict(list)
    for key,pair in index.items():
        if set(pair)!=set(['no_mechanism','static_aeic']):raise ValueError('unpaired agent slot')
        truth,left = pair['no_mechanism']; other_truth,right = pair['static_aeic']
        if truth!=other_truth:raise ValueError('inconsistent labels')
        l,r = left['stance'],right['stance']
        if mode!='original':
            l = conclusion(left['report']['C'],l,mode)
            r = conclusion(right['report']['C'],r,mode)
        category = (left['chosen_action'],right['chosen_action'],trace_kind(left),trace_kind(right))
        cells[category].append((l==truth,r==truth,key))
    summary = []
    for (la,ra,lp,rp),values in sorted(cells.items()):
        b = sum(not l and r for l,r,k in values);c = sum(l and not r for l,r,k in values)
        summary.append({'left_action':la,'right_action':ra,'left_route':lp,'right_route':rp,
                        'same_action_and_route':la==ra and lp==rp,
                        'slots':len(values),'left_correct':sum(l for l,r,k in values),
                        'right_correct':sum(r for l,r,k in values),'wrong_correct':b,'correct_wrong':c,
                        'net_correct_votes':b-c})
    return {'mode':mode,'round':round_id,'paired_slots':len(index),'cells':summary,
            'total_net_correct_votes':sum(x['net_correct_votes'] for x in summary),
            'same_bundle_net_votes':sum(x['net_correct_votes'] for x in summary if x['same_action_and_route']),
            'changed_bundle_net_votes':sum(x['net_correct_votes'] for x in summary if not x['same_action_and_route'])}


def main():
    path = ROOT/'experiments/results/mechanism/frozen/formal_v10/question_level.jsonl'
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    result = {'scope':'observed paired report accounting by actual logged bundle, not causal allocation attribution',
              'no_api_calls':True,'source_sha256':sha256(path.read_bytes()).hexdigest(),
              'partitions':[partition(rows,mode,rid) for mode in ['original','literal_yes_no_only'] for rid in [1,3,None]],
              'limits':['paired question/persona/round, but condition collection time and backend versions not controlled',
                        'changed bundle simultaneously changes model/cap/instruction/self-check',
                        'equal bundle does not mean same stochastic draw or identical report',
                        'selected switches are not a random sample of all possible allocations',
                        'cannot evaluate same-budget random routing without unobserved alternative report executions',
                        'net vote partition is not a decomposition of nonlinear majority decisions']}
    coverage = defaultdict(set)
    for r in rows:
        for t in r['rounds']:
            key = (r['dataset'],str(r['question_id']),t['round_id'],t['agent_id'])
            coverage[key].add((t['chosen_action'],trace_kind(t)))
    low = ('low_effort','zhipu_flash:glm-4-flash')
    high = ('aeic_effort','deepseek:deepseek-chat')
    result['observed_low_high_bundle_coverage_all_three_conditions'] = {
        'logical_question_round_persona_slots':len(coverage),
        'both_low_and_AEIC_observed':sum(low in s and high in s for s in coverage.values()),
        'no_low_observation':sum(low not in s for s in coverage.values()),
        'no_AEIC_observation':sum(high not in s for s in coverage.values()),
        'scope':'observed across separate condition executions; not randomized paired potential outcomes, no routing propensities recorded'}
    (OUT/'route_pairs.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
