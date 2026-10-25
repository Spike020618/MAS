"""Bounded same-model sampling-action gate, then optional frozen test."""
import argparse
from collections import Counter
from hashlib import sha256
from itertools import product, combinations
import json
from pathlib import Path
from random import Random
import sys
import time
from urllib.parse import urlparse

from experiments.core.mechanism.bandit_policy_learning import IndependentActors, empirical_deviation
from experiments.mechanism import run_full_context_challenge as base
from experiments.mechanism import study_information_complementarity as source
from experiments.mechanism.run_premise_review import parse_row
from experiments.mechanism.run_upgrade_pilot import CallJournal, freeze, lock, read_rows, append

ROOT = base.ROOT
OUT = ROOT / 'experiments/results/resource_allocation/sampling_coordination_v1'
DESIGN = ROOT / 'docs/research/采样行动协调_锁定方案_2026-10-25.md'
TEMPERATURES = (.2, .8, 1.3)
PROFILES = tuple(product(range(3), repeat=3))


def request(case, role, action):
    if set(case) != {'id', 'question', 'choices', 'source_id', 'split'}:
        raise ValueError('public case schema, no labels')
    if role not in range(3) or action not in (*range(3), 'thinking'):
        raise ValueError('registered action required')
    value = {'backend': 'deepseek', 'model': 'deepseek-flash', 'max_tokens': 1000,
             'thinking': {'type': 'disabled'}, 'temperature': TEMPERATURES[action] if action != 'thinking' else .2,
             'top_p': 1, 'response_format': {'type': 'json_object'}, 'messages': [
                 {'role': 'system', 'content': source.INSTRUCTION},
                 {'role': 'user', 'content': json.dumps({k: case[k] for k in ('question', 'choices')}, sort_keys=True)}]}
    if action == 'thinking':
        value['thinking']['type'] = 'enabled'; value['max_tokens'] = 4096; del value['temperature']
    if len(base.encoded(value)) > 150000: raise ValueError('request too large')
    return value


def outcome(slots, profile):
    answers = [slots[f'{i}/{a}']['answer'] for i, a in enumerate(profile)]
    counts = Counter(a for a in answers if a not in (None, 'abstain', 'invalid'))
    winners = [a for a, n in counts.items() if n >= 2]
    return winners[0] if winners else 'abstain'


def payoff(slots, label, profile):
    correct = int(outcome(slots, profile) == label)
    costs = [slots[f'{i}/{a}']['tokens'] / 50000 for i, a in enumerate(profile)]
    return [correct - c for c in costs]


def action_gate(outputs, labels):
    varying = []; coupled = []
    for cid, slots in outputs.items():
        correct = {p: int(outcome(slots, p) == labels[cid]) for p in PROFILES}
        if len(set(correct.values())) > 1: varying.append(cid)
        interactions = False
        for i, j in combinations(range(3), 2):
            other = next(k for k in range(3) if k not in (i, j))
            for fixed in range(3):
                for a, b in product((1, 2), repeat=2):
                    p = [0, 0, 0]; p[other] = fixed
                    def c(x, y):
                        q = p[:]; q[i] = x; q[j] = y
                        return correct[tuple(q)]
                    if c(a, b) - c(a, 0) - c(0, b) + c(0, 0) != 0: interactions = True
        if interactions: coupled.append(cid)
    return {'varying_correctness_ids': varying, 'coupled_correctness_ids': coupled,
            'passed': len(varying) >= 2 and bool(coupled),
            'meaning': 'observed action variability only, not efficacy or necessity', 'automatic_large_run': False}


def prepare():
    if (OUT / 'protocol.json').exists(): verify(); return
    OUT.mkdir(parents=True, exist_ok=True)
    # Temporary pyarrow runtime only; never edit the original source preparer.
    sys.path.insert(0, '/private/tmp/mas-production-deps')
    prior = source.OUT
    try:
        source.OUT = OUT / 'source_data'; source.prepare()
    finally: source.OUT = prior
    public = base.read(OUT / 'source_data/public_cases.json')
    labels = {v['id']: v['label'] for v in base.read(OUT / 'source_data/private_labels.json')}
    for split, original in (('train', 'calibration'), ('test', 'evaluation')):
        chosen = []
        for domain in ('strategyqa', 'mmlu', 'truthfulqa'):
            ranked = sorted((c for c in public if c['split'] == original and c['id'].startswith(domain + ':')),
                            key=lambda c: sha256(('sampling-v1/' + c['id']).encode()).hexdigest())
            chosen += ranked[:4]
        chosen.sort(key=lambda c: sha256(c['id'].encode()).hexdigest())
        if len(chosen) != 12: raise ValueError('expected12 per split')
        freeze(OUT / (split + '_public.json'), chosen)
        freeze(OUT / (split + '_labels.json'), {c['id']: labels[c['id']] for c in chosen})
    freeze(OUT / 'protocol.json', {'version': 'sampling-coordination-v1', 'train': 12, 'test': 12,
        'call_limit': 240, 'usd_proxy_cap': 5, 'temperatures': TEMPERATURES, 'top_p': 1,
        'model': 'deepseek-flash', 'fingerprint': base.FP, 'max_tokens': 1000, 'thinking_cap': 4096,
        'learning': '3 independent REINFORCE actors, lr.05 seed7,3000 cached training updates',
        'train_gate': '>=2 varying-correctness questions and nonzero pair interaction',
        'no_automatic_expansion': True, 'human_authorization': '做',
        'design_sha256': base.h(DESIGN),
        'code_sha256': {str(p.relative_to(ROOT)): base.h(p) for p in (Path(__file__),
            ROOT / 'experiments/core/mechanism/bandit_policy_learning.py',
            ROOT / 'experiments/tests/test_sampling_coordination_screen.py', ROOT / 'experiments/mechanism/run_upgrade_pilot.py',
            Path(source.__file__), ROOT / 'experiments/mechanism/run_premise_review.py')},
        'files': {name: base.h(OUT / name) for name in ('train_public.json', 'test_public.json', 'train_labels.json', 'test_labels.json')}})


def verify():
    p = base.read(OUT / 'protocol.json')
    if base.h(DESIGN) != p['design_sha256']: raise ValueError('design changed')
    for rel, digest in p['code_sha256'].items():
        if base.h(ROOT / rel) != digest: raise ValueError('code changed:' + rel)
    for name, digest in p['files'].items():
        if base.h(OUT / name) != digest: raise ValueError('input changed:' + name)
    return p


def collect(case, split, transport, spend, policy=None):
    folder = OUT / split / sha256(case['id'].encode()).hexdigest()
    with lock(folder):
        journal = CallJournal(folder, transport, 10)
        keys = [f'{i}/{a}' for i in range(3) for a in range(3)]
        if policy is not None:
            rng = Random(int(sha256(('frozen/' + case['id']).encode()).hexdigest(), 16))
            selected = [rng.choices(range(3), weights=p, k=1)[0] for p in policy['probabilities']]
            freeze(folder / 'selected_before_calls.json', {'profile': selected, 'policy_sha256': base.h(OUT / 'frozen_policy.json')})
            first = [f'{i}/{a}' for i, a in enumerate(selected)]
            keys = first + [k for k in keys if k not in first]
        keys += ['thinking']; slots = {}
        for key in keys:
            i, a = (0, 'thinking') if key == 'thinking' else map(int, key.split('/'))
            req = request(case, i, a)
            proxy_tokens = len(base.encoded(req)) + 4096
            reserve = (proxy_tokens * .3 + req['max_tokens'] * 1.2) / 1e6
            cached = any(r['key'] == key for r in read_rows(folder / 'responses.jsonl'))
            if not cached and spend['used'] + reserve > spend['cap']: raise RuntimeError('budget exhausted')
            raw = journal.call(key, req)
            if raw['response']['http_status'] != 200: raise RuntimeError('HTTP failure, no retry')
            body = json.loads(raw['response']['body']); usage = body.get('usage', {})
            if (body.get('model'), body.get('system_fingerprint')) != ('deepseek-flash', base.FP): raise RuntimeError('model changed')
            if any(type(usage.get(k)) is not int or usage[k] < 0 for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')): raise RuntimeError('unknown usage')
            if usage['total_tokens'] != usage['prompt_tokens'] + usage['completion_tokens']: raise RuntimeError('usage inconsistent')
            if not cached: spend['used'] += (usage['prompt_tokens'] * .3 + usage['completion_tokens'] * 1.2) / 1e6
            if usage['prompt_tokens'] > proxy_tokens or usage['completion_tokens'] > req['max_tokens']: raise RuntimeError('token reservation exceeded')
            slots[key] = parse_row(raw, case['choices'])
        freeze(folder / 'sealed_outputs.json', slots)
        return slots


def train_policy(outputs, labels):
    actor = IndependentActors([3, 3, 3], learning_rate=.05, seed=7)
    rng = Random(7); ids = sorted(outputs)
    for _ in range(3000):
        cid = rng.choice(ids); actions = actor.sample()
        actor.update(actions, payoff(outputs[cid], labels[cid], actions))
    policy = actor.freeze()
    table = {p: [sum(payoff(outputs[c], labels[c], p)[i] for c in ids) / len(ids) for i in range(3)] for p in PROFILES}
    fixed = max(PROFILES, key=lambda p: (sum(int(outcome(outputs[c], p) == labels[c]) -
        sum(outputs[c][f'{i}/{a}']['tokens'] for i, a in enumerate(p)) / 50000 for c in ids), tuple(-a for a in p)))
    single = max((f'{i}/{a}' for i in range(3) for a in range(3)), key=lambda k:
                 (sum(outputs[c][k]['answer'] == labels[c] for c in ids), -sum(outputs[c][k]['tokens'] for c in ids), k))
    policy.update(train_empirical_deviation=empirical_deviation(table, policy['probabilities']),
                  centralized_erm=list(fixed), train_selected_single=single)
    return policy


def run():
    verify()
    if time.strftime('%Y-%m-%d', time.gmtime()) != base.PRICE['checked_date']: raise RuntimeError('recheck rates')
    from experiments.core.llm import LLM_BACKENDS
    if urlparse(LLM_BACKENDS['deepseek']['base_url']).hostname != 'api.deepseek.com': raise ValueError('unexpected endpoint')
    from experiments.mechanism.study_explicit_thinking_control import transport
    with lock(OUT):
        raw = [r for p in OUT.rglob('responses.jsonl') for r in read_rows(p)]
        for p in OUT.rglob('attempts.jsonl'):
            if {r['key'] for r in read_rows(p)} - {r['key'] for r in read_rows(p.with_name('responses.jsonl'))}: raise RuntimeError('unresolved attempt, no retry')
        spent = sum((json.loads(r['response']['body'])['usage']['prompt_tokens'] * .3 +
                     json.loads(r['response']['body'])['usage']['completion_tokens'] * 1.2) / 1e6 for r in raw)
        spend = {'used': spent, 'cap': 5}; start = time.perf_counter()
        try:
            train = {}
            for c in base.read(OUT / 'train_public.json'):
                train[c['id']] = collect(c, 'train', transport, spend)
                print('sealed train', len(train), '/12', flush=True)
            freeze(OUT / 'train_sealed_outputs.json', train)
            labels = base.read(OUT / 'train_labels.json'); gate = action_gate(train, labels)
            freeze(OUT / 'action_gate.json', gate)
            print(json.dumps(gate, ensure_ascii=False), flush=True)
            if not gate['passed']:
                freeze(OUT / 'screen_status.json', {'status': 'train_action_gate_failed_test_not_started', 'automatic_expansion': False})
                return
            policy = train_policy(train, labels); freeze(OUT / 'frozen_policy.json', policy)
            test = {}
            for c in base.read(OUT / 'test_public.json'):
                test[c['id']] = collect(c, 'test', transport, spend, policy)
                print('sealed test', len(test), '/12', flush=True)
            freeze(OUT / 'test_sealed_outputs.json', test)
            freeze(OUT / 'test_readouts.json', {cid: {'game': outcome(s, base.read(OUT / 'test' / sha256(cid.encode()).hexdigest() / 'selected_before_calls.json')['profile']),
                'centralized_erm': outcome(s, policy['centralized_erm']), 'single': s[policy['train_selected_single']]['answer'],
                'thinking': s['thinking']['answer']} for cid, s in test.items()})
            labels = base.read(OUT / 'test_labels.json'); decisions = base.read(OUT / 'test_readouts.json')
            metrics = {m: sum(v[m] == labels[cid] for cid, v in decisions.items()) for m in next(iter(decisions.values()))}
            allfixed = {''.join(map(str, p)): sum(outcome(s, p) == labels[cid] for cid, s in test.items()) for p in PROFILES}
            freeze(OUT / 'test_result.json', {'n': 12, 'correct': metrics, 'all_fixed_profiles': allfixed,
                'uniform_random_expected_correct': sum(allfixed.values()) / 27,
                'scope': 'one cached output/action, exploratory; no BNE or general efficacy claim'})
            freeze(OUT / 'screen_status.json', {'status': 'frozen_test_completed_no_expansion', 'automatic_expansion': False})
            print(json.dumps(metrics), flush=True)
        finally:
            append(OUT / 'invocations.jsonl', {'elapsed_seconds': time.perf_counter() - start, 'known_cost_proxy_usd': spend['used']})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('prepare', 'verify', 'run'))
    {'prepare': prepare, 'verify': verify, 'run': run}[p.parse_args().stage]()
