"""Recompute explanatory contrasts and document what their controls identify.

No new API calls, no deletion/relabeling, no equivalence from nonsignificance.
"""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import comb
import random

from experiments.mechanism.attribute_legacy_gain import ROOT, OUT, transitions
from experiments.mechanism.audit_legacy_source_groups import normalize, split_question
from experiments.mechanism.extend_legacy_sensitivity import source_screen, outcome
from experiments.mechanism.report_interface import observe_interface
from experiments.mechanism.audit_resource_identification import execution_messages


def paired(left,right):
    if left.keys()!=right.keys() or not left:
        raise ValueError('nonempty exact paired keys required')
    keys = sorted(left)
    l,r = [left[k] for k in keys],[right[k] for k in keys]
    result = transitions(l,r)
    b,c = result['wrong_correct'],result['correct_wrong']
    n = b+c
    result['exact_mcnemar_p_unadjusted_descriptive'] = min(1.,2*sum(comb(n,i) for i in range(min(b,c)+1))/2**n) if n else 1.
    if not n:
        result['stratified_bootstrap_percentile_95_descriptive'] = None
        result['bootstrap_note'] = 'zero observed discordance: do not display degenerate [0,0] as certainty of equivalence'
        result['scope'] = 'post-hoc paired diagnostic, no equivalence test or new confirmatory endpoint'
        return result
    domains = sorted({k[0] for k in keys})
    cells = [[int(right[k])-int(left[k]) for k in keys if k[0]==d] for d in domains]
    rng = random.Random(20261022)
    draws = []
    for _ in range(4000):
        draws.append(sum(sum(cell[rng.randrange(len(cell))] for _ in cell) for cell in cells)/len(keys))
    draws.sort()
    result['stratified_bootstrap_percentile_95_descriptive'] = [draws[99],draws[3899]]
    result['scope'] = 'post-hoc paired diagnostic, no equivalence test or new confirmatory endpoint'
    return result


def interface_fixture_counterexamples():
    # Schema checks are not factual validation; these are constructed fixtures.
    raw = json.dumps({'A':'unsupported','E':[],'I':'invalid inference','C':'Yes'})
    empty = observe_interface(raw,'aeic').to_dict()
    free = observe_interface('Specific evidence: observation. Final answer: Yes.','free_text').to_dict()
    generic = observe_interface(json.dumps({'analysis':'Specific evidence: observation.','answer':'Yes'}),'generic_structure').to_dict()
    return {'constructed_not_measured_model_behavior':True,
            'empty_E_AEIC_parse_success':empty['parse_success'],
            'empty_E_AEIC_boundary_explicit':empty['evidence_boundary_explicit'],
            'free_text_with_evidence_words_boundary_explicit':free['evidence_boundary_explicit'],
            'generic_with_evidence_words_boundary_explicit':generic['evidence_boundary_explicit']}


def main():
    split_path = ROOT/'experiments/mechanism/splits.json'
    splits = json.loads(split_path.read_text())['datasets']
    caches = {d:json.loads((ROOT/sp['source']).read_text())['questions'] for d,sp in splits.items()}
    gold = {d:{str(q['question_id']):q['ground_truth'] for q in qs} for d,qs in caches.items()}
    clean = {d:set(source_screen(caches[d],sp['splits'],'test')['keep']) for d,sp in splits.items()}
    inputs = [split_path]
    dp = ROOT/'experiments/results/mechanism/decomposition/mechanism_decomposition_v1.jsonl'
    ip = ROOT/'experiments/results/mechanism/interface/report_interface_v1.jsonl'
    decomposition, interface = defaultdict(dict),defaultdict(dict)
    for path,groups,field in [(dp,decomposition,'condition'),(ip,interface,'protocol')]:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if not r['completed'] or r.get('error') or r['ground_truth']!=gold[r['dataset']][str(r['question_id'])]:
                raise ValueError('invalid record or inconsistent native-verified label')
            k = (r['dataset'],str(r['question_id']))
            group = r[field].split('__variant@')[-1] if field=='condition' else r[field]
            if k in groups[group]:raise ValueError('duplicate paired key')
            groups[group][k] = r
            if field=='protocol' and observe_interface(r['raw_response'],group).to_dict()!=r['observation']:
                raise ValueError('interface observation does not replay')
        inputs.append(path)
    def screen(groups,clean_only):
        return {g:{k:r for k,r in rows.items() if not clean_only or k[1] in clean[k[0]]} for g,rows in groups.items()}
    results = {}
    for label,clean_only in [('original',False),('source_group_clean',True)]:
        dg,ig = screen(decomposition,clean_only),screen(interface,clean_only)
        d_summary = {}
        for g,rs in dg.items():
            turns = [t for r in rs.values() for t in r['rounds']]
            d_summary[g] = {'n':len(rs),'correct':sum(r['final_decision']==r['ground_truth'] for r in rs.values()),
                'literal_parser_correct':sum(outcome(r,'literal_yes_no_only') for r in rs.values()),
                'tokens':sum(t['token_input']+t['token_output'] for t in turns),
                'calls':sum(len(t['action_trace']) for t in turns),
                'action_counts':dict(Counter(t['chosen_action'] for t in turns)),
                'execution_backend_counts':dict(Counter(c['backend'] for t in turns for c in t['action_trace']
                    if c['stage']!='choose' and not c['stage'].startswith('format_repair')))}
        i_summary = {g:{'n':len(rs),'correct':sum(r['observation']['stance']==r['ground_truth'] for r in rs.values()),
                       'parse_success':sum(r['observation']['parse_success'] for r in rs.values()),
                       'explicit_E_boundary':sum(r['observation']['evidence_boundary_explicit'] for r in rs.values()),
                       'input_tokens':sum(r['token_input'] for r in rs.values()),
                       'output_tokens':sum(r['token_output'] for r in rs.values())} for g,rs in ig.items()}
        dc,ic = [],[]
        for l,r in [('no_contract_fixed','contract_fixed'),('contract_fixed','full_mechanism'),('matched_compute','full_mechanism')]:
            dc.append({'left':l,'right':r,**paired({k:v['final_decision']==v['ground_truth'] for k,v in dg[l].items()},
                                                  {k:v['final_decision']==v['ground_truth'] for k,v in dg[r].items()})})
        for l in ['free_text','generic_structure']:
            ic.append({'left':l,'right':'aeic',**paired({k:v['observation']['stance']==v['ground_truth'] for k,v in ig[l].items()},
                                                      {k:v['observation']['stance']==v['ground_truth'] for k,v in ig['aeic'].items()})})
        results[label] = {'decomposition':d_summary,'interface':i_summary,'decomposition_contrasts':dc,'interface_contrasts':ic}
    low,high = execution_messages('low_effort',fixed=True),execution_messages('aeic_effort',fixed=True)
    prompts = {'fixed_resource_low_vs_AEIC':{'same_cap':low[0]['max_tokens']==high[0]['max_tokens'],
                                          'same_messages':low[0]['messages']==high[0]['messages']}}
    for a in ['low_effort','aeic_effort','verify','defer_imitate']:
        full,matched = execution_messages(a),execution_messages(a,matched=True)
        prompts[a+'_full_vs_matched'] = {'same_calls':len(full)==len(matched),
                                        'same_messages':[x['messages'] for x in full]==[x['messages'] for x in matched]}
    from experiments.mechanism.audit_resource_identification import audit
    schedule_checks = audit()['schedule_checks']
    if schedule_checks.get('agent_slots')!=180 or any(v!=180 for v in schedule_checks.values()):
        raise ValueError('matched schedule did not pass all 180 observed slots')
    inputs += [ROOT/'experiments/results/mechanism/decomposition/matched_compute_schedule_v1.json',
               ROOT/'experiments/mechanism/collect.py',ROOT/'experiments/mechanism/report_interface.py',
               ROOT/'experiments/mechanism/audit_resource_identification.py']
    result = {'no_api_calls':True,'results':results,'current_mocked_prompt_checks':prompts,
        'observed_matched_schedule_checks':schedule_checks,
        'interface_metric_counterexamples':interface_fixture_counterexamples(),
        'source_hashes':{str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in inputs},
        'limits':['current mocked requests are not archived raw requests',
                  'matched schedule controls backend/call count/cap, not all report instructions or realized tokens',
                  'fixed-resource contrast still changes action-specific report instructions',
                  'decomposition uses 3 agents/1 round/DeepSeek-only low-AEIC tiers, not formal 5-agent mixed-model estimand',
                  'interface is single reporting call, no contract/action selector/reputation/ensemble',
                  'common output cap is not equal realized token count; input prompts differ by protocol',
                  'evidence boundary metric is defined by schema, not independent factual evidence adjudication',
                  'p>0.05 does not establish equality or absence of relevant effects']}
    (OUT/'explanatory_controls.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
