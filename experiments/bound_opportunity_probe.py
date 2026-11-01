"""Fixed-state diagnostics, isolated from running paper/probe releases.

Controlled fixtures test execution and opportunity use, not natural benchmark quality.
Each ordinary/linked arm gets exactly the same frozen state and evidence.
"""
import argparse
import collections
import copy
import json
from pathlib import Path
from experiments.consensus_mechanism import core as c, decision_review as linked
from experiments.consensus_mechanism.data import read
from experiments.consensus_mechanism.runner import locked
from experiments.consensus_mechanism.runtime import Runtime
from experiments.consensus_recommendation.common import ROOT, create, fingerprint, sha
from experiments.evidence_consensus.schema import validate_task
from experiments.paired_deliberation.native_optimization_runtime import audit

VERSION = 'bound-opportunity-2-typed-verdict'
BRANCHES = {'empty': [], 'A': ['A'], 'B': ['B'], 'AB': ['A', 'B']}

def revision_basis(refs, assigned):
    cited = {r['source_id'] for r in refs}
    assigned = set(assigned)
    new, prior = cited & assigned, cited - assigned
    return 'joint' if new and prior else 'assigned' if new else 'initial' if prior else 'none'

def validate_linked(task, state, report, action):
    # Deliberately no preliminary strict validation: that would reject initial-only
    # citations before the relaxed linked validator can see them (core.invoke).
    refs, _ = linked.validate(task, state, report, 100, action)
    return refs, revision_basis(refs, action['source_ids'])

def review_request(task, state, reports, action, reviewer=0):
    req = linked.request(task, state, 100, reports, action, reviewer)
    req['messages'][0]['content'] += (
        ' FIELD TYPES: verdict describes the TARGET CLAIM and is ONLY support, refute, or unclear. '
        'An answer option such as not_enough_info is NEVER a verdict. '
        'The native decision is distinct from the claim verdict. '
        'If metadata permits an uncertainty answer, a patch setting that uncertainty answer '
        'is legal even with unclear verdict and no citations; missing evidence warrants uncertainty, not no. '
        'For an evidence-only task, use only the supplied records. '
        'Do not explain the output contract; give a concise evidence rationale.')
    return req

def invoke(api, slot, task, state, reports, action, mode):
    phase = 'direct' if mode == 'ordinary' else 'review'
    req = (c.request(task, state, phase, 100, reports, action, reviewer=0)
           if mode == 'ordinary' else review_request(task, state, reports, action, 0))
    before_hash = fingerprint(state)
    event = {'valid': False, 'accepted': False, 'phase': phase,
             'request_hash': fingerprint(req), 'state_hash': before_hash}
    try:
        record = api.call(slot, req)
        if fingerprint(state) != before_hash:
            raise ValueError('state changed during request')
        event.update(report=record['report'], usage=record['usage'], returned_model=record['model'])
        if mode == 'linked':
            refs, basis = validate_linked(task, state, record['report'], action)
        else:
            refs = c.validate_report(task, state, record['report'], phase, 100, action)
            basis = revision_basis(refs, action['source_ids'])
        after, accepted = c.commit(task, state, record['report'], phase, 100, refs)
        event.update(valid=True, accepted=accepted, revision_basis=basis, citations=refs)
        return after, event
    except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
        event['error'] = str(exc)
        return copy.deepcopy(state), event

def fixtures():
    rows = []
    for kind, initial, answer, texts in [
        ('single', 'no', 'yes', ['Record VX17 is approved.', 'Record WX29 is pending.']),
        ('joint', 'no', 'yes', ['Record VX17 belongs to class Kappa.', 'All records in class Kappa are approved.']),
        ('keep', 'yes', 'yes', ['Record VX17 is approved.', 'Record WX29 is pending.']),
        ('insufficient', 'not_enough_info', 'not_enough_info', ['Record WX29 is approved.', 'Record WX29 belongs to class Kappa.']),
        ('redundant', 'no', 'yes', ['Record VX17 is approved.', 'Record VX17 is approved.'])]:
        tid = 'controlled:' + fingerprint(['bound-record', kind])[:16]
        catalog = {sid: {'source_id': sid, 'text': text, 'title': sid,
                        'kind': 'controlled_record', 'scope': {'task_id': tid}}
                   for sid, text in zip(['A', 'B'], texts)}
        task = {'id': tid, 'kind': 'qa', 'prompt': 'Using ONLY these records, is record VX17 approved? Answer yes, no, or not_enough_info. Absence of a fact is not its negation.',
                'options': ['yes', 'no', 'not_enough_info'], 'candidates': [],
                'initial': initial, 'source_catalog': catalog,
                'metadata': {'evidence_only': True, 'acquisition_partition': 'fit',
                             'uncertainty_answers': ['not_enough_info']}}
        state = c.fresh(task)
        state['claims']['target'] = {'id': 'target', 'text': 'Record VX17 is approved.', 'active': True, 'owner': 0}
        # Correct-maintenance case already has supporting evidence before review.
        if kind == 'keep': c.expose(task, state, ['A'])
        rows.append({'task': task, 'state': state, 'reports': [], 'target_id': 'target',
                     'sources': {'A': ['A'], 'B': ['B']}, 'stratum': kind,
                     'adjudication': {'status': 'constructed', 'author': 'explicit closed-world record rules',
                                      'reason': 'Exact entity and predicate; single, chain, absent, or duplicate records.'},
                     'evaluation': {'kind': 'qa', 'answer': answer}})
    return rows

def validate_binding(row):
    task, state = row['task'], row['state']
    validate_task(task)
    if state['task_id'] != task['id'] or row['target_id'] not in state['claims']:
        raise ValueError('frozen state/target mismatch')
    c.full_decision(task, state['decision'])
    if not state['claims'][row['target_id']].get('active', True):
        raise ValueError('target must be active')
    catalog = task['source_catalog']
    if any(s not in catalog for s in state['visible']): raise ValueError('unknown visible source')
    if row['adjudication']['status'] not in ('constructed', 'supported'):
        raise ValueError('unverified opportunity cannot be promoted')
    if not row['adjudication'].get('reason') or not row['adjudication'].get('author'):
        raise ValueError('record adjudication author and reasoning')
    if row['adjudication']['status'] == 'supported':
        if task['metadata'].get('acquisition_partition') not in ('fit', 'calibration'):
            raise ValueError('label-selected natural tasks must be training-side')
        witnesses = row['adjudication'].get('span_ids', [])
        pool = c.spans(task)
        bound = set(row['sources']['A'] + row['sources']['B'])
        if not witnesses or any(s not in pool or pool[s]['source_id'] not in bound for s in witnesses):
            raise ValueError('support judgment must bind exact action evidence spans')
    for name in ('A', 'B'):
        ids = row['sources'][name]
        if not ids or len(ids) != len(set(ids)) or any(s not in catalog for s in ids):
            raise ValueError('explicit catalog source binding required')
        c.expose(task, copy.deepcopy(state), ids)
    if row['evaluation']['kind'] != task['kind']: raise ValueError('label kind mismatch')
    if row['stratum'] not in ('keep', 'insufficient', 'single', 'joint', 'redundant'):
        raise ValueError('unknown diagnostic stratum')
    initial_quality = c.quality(state['decision'], row['evaluation'])
    if row['stratum'] in ('keep', 'insufficient') and initial_quality != 1:
        raise ValueError('maintenance controls must start correct')
    if row['stratum'] in ('single', 'joint', 'redundant') and initial_quality >= 1:
        raise ValueError('recovery tasks must start below full quality')
    return True

def run_case(row, api, index):
    task, state = row['task'], row['state']
    arms = {}
    for branch, keys in BRANCHES.items():
        ids = list(dict.fromkeys(s for key in keys for s in row['sources'][key]))
        exposed = copy.deepcopy(state)
        c.expose(task, exposed, ids)
        action = {'target_id': row['target_id'], 'source_ids': ids}
        for mode in ('ordinary', 'linked'):
            after, event = invoke(api, f'{index}_{branch}_{mode}', task, copy.deepcopy(exposed),
                                  row['reports'], action, mode)
            arms[branch + ':' + mode] = {'decision': after['decision'], 'event': event,
                                        'exposed_state_hash': fingerprint(exposed), 'action': action}
    return {'task_id': task['id'], 'stratum': row['stratum'], 'initial_decision': state['decision'],
            'bound_state_hash': fingerprint(state), 'arms': arms}

def summarize(rows, cases):
    methods = {}
    for mode in ('ordinary', 'linked'):
        counts = collections.Counter()
        gains, interactions = [], []
        for row, case in zip(rows, cases):
            label = row['evaluation']; before = c.quality(case['initial_decision'], label)
            branch = {k: case['arms'][k + ':' + mode] for k in BRANCHES}
            quality = {k: c.quality(v['decision'], label) for k, v in branch.items()}
            counts['tasks'] += 1; counts['valid'] += sum(v['event']['valid'] for v in branch.values())
            counts['events'] += 4
            counts['improved'] += quality['AB'] > before; counts['harmed'] += quality['AB'] < before
            counts['empty_improved'] += quality['empty'] > before
            if row['stratum'] in ('keep', 'insufficient'):
                counts['maintenance_tasks'] += 1
                counts['maintained'] += quality['AB'] == before
                counts['false_revision'] += branch['AB']['decision'] != case['initial_decision']
            else:
                counts['recovery_tasks'] += 1
                counts['recovered'] += quality['AB'] > before
            if all(v['event']['valid'] for v in branch.values()):
                ga, gb, gab = [quality[k] - quality['empty'] for k in ('A', 'B', 'AB')]
                gains.append((ga, gb, gab)); interactions.append(gab - ga - gb)
                counts['action_difference_tasks'] += len({ga, gb, gab}) > 1
                counts['joint_only_tasks'] += gab > 0 and ga <= 0 and gb <= 0
                counts['positive_evidence_tasks'] += max(ga, gb, gab) > 0
        methods[mode] = {**dict(counts), 'measured_gains': gains, 'interactions': interactions,
                         'recovery_rate': counts['recovered'] / max(1, counts['recovery_tasks']),
                         'maintenance_rate': counts['maintained'] / max(1, counts['maintenance_tasks'])}
    v = methods['linked']
    gate = {'complete_contracts': v['valid'] == v['events'],
            'has_maintenance_controls': v.get('maintenance_tasks', 0) >= 2,
            'maintains_controls': v.get('false_revision', 0) == 0,
            'uses_new_evidence': v.get('positive_evidence_tasks', 0) > 0,
            'recovers_some': v.get('recovered', 0) > 0,
            'joint_signal': v.get('joint_only_tasks', 0) > 0}
    return {'tasks': len(cases), 'methods': methods, 'checks': gate,
            'promotion_gate': all(gate.values()),
            'scope': 'Fixed-state diagnostic only. Constructed records do not establish natural QA, recommendation, or game advantage.'}

def budget_eligibility(summary, calls, candidate_actions):
    if calls < 1: return {'eligible': False, 'reason': 'zero budget is a control only'}
    v = summary['methods']['linked']
    checks = {'contracts': v['valid'] == v['events'],
              'positive_actions': v.get('positive_evidence_tasks', 0) > 0,
              'different_values': v.get('action_difference_tasks', 0) > 0,
              'scarcity': candidate_actions > calls}
    return {'eligible': all(checks.values()), 'checks': checks,
            'scope': 'Necessary diagnostic checks only; no learned selector or game gain inferred.'}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'run', 'report', 'budget_check'])
    p.add_argument('--root', type=Path, default=ROOT/'experiments/results/bound_opportunity_v3')
    p.add_argument('--manifest', type=Path)
    p.add_argument('--run', action='store_true')
    p.add_argument('--max-cny', type=float, default=.2)
    p.add_argument('--calls', type=int, default=2)
    p.add_argument('--candidate-actions', type=int, default=3)
    a = p.parse_args()
    if not 0 < a.max_cny <= .5: p.error('hard cap must be 0..0.5 CNY')
    if a.stage == 'budget_check':
        print(json.dumps(budget_eligibility(read(a.root/'summary.json'), a.calls, a.candidate_actions), ensure_ascii=False, indent=2)); return
    rows = read(a.manifest) if a.manifest else fixtures()
    if len({r['task']['id'] for r in rows}) != len(rows) or not rows: raise ValueError('nonempty unique tasks required')
    for row in rows: validate_binding(row)
    source_files = [Path(__file__)] + list((ROOT/'experiments/consensus_mechanism').glob('*.py')) + list((ROOT/'experiments/paired_deliberation').glob('*.py')) + [ROOT/'experiments/consensus_recommendation/common.py', ROOT/'experiments/evidence_consensus/schema.py']
    plan = {'version': VERSION, 'manifest_hash': fingerprint(rows), 'provider': 'qwen_flash',
            'physical_cap': len(rows)*8, 'max_cny': a.max_cny, 'reuse_root': str(ROOT/'experiments/results/bound_opportunity_v2'),
            'source_sha256': {str(f.relative_to(ROOT)): sha(f) for f in sorted(set(source_files))},
            'scope': 'training-side controlled diagnostic, independent of paper main and probe v3'}
    with locked(a.root):
        create(a.root/'manifest.private.json', rows); create(a.root/'plan.json', plan)
        for name in plan['source_sha256']:
            dest = a.root/'source_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and sha(dest) != plan['source_sha256'][name]: raise ValueError('source snapshot drift')
            dest.write_bytes((ROOT/name).read_bytes())
        if a.stage == 'prepare' or (a.stage == 'run' and not a.run):
            print(json.dumps({'tasks': len(rows), 'max_calls': len(rows)*8, 'network': False, 'root': str(a.root)})); return
        if a.stage == 'run':
            from dotenv import load_dotenv
            load_dotenv(ROOT/'.env.local', override=False)
            api = Runtime(a.root, plan, 'qwen_flash')
            for i, row in enumerate(rows):
                path = a.root/'cases'/f'case_{i}.json'
                if not path.exists(): create(path, run_case(row, api, i))
            cases = [read(a.root/'cases'/f'case_{i}.json') for i in range(len(rows))]
            create(a.root/'generation_complete.json', {'manifest_hash': fingerprint(rows), 'hashes': [fingerprint(x) for x in cases]})
        cases = [read(a.root/'cases'/f'case_{i}.json') for i in range(len(rows))]
        seal = read(a.root/'generation_complete.json')
        if seal != {'manifest_hash': fingerprint(rows), 'hashes': [fingerprint(x) for x in cases]}: raise ValueError('seal mismatch')
        result = summarize(rows, cases); result['physical'] = audit(a.root, 'qwen_flash')
        create(a.root/'summary.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
