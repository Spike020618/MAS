"""Offline data, message-order, replay and historical-integrity checks."""
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from experiments.mechanism import run_sampling_coordination_screen as r
from experiments.mechanism.complete_legacy_gain_audit import frozen_hashes


def main():
    before = frozen_hashes(); r.verify()
    source_cases = {c['id']: c for c in r.base.read(r.OUT / 'source_data/public_cases.json')}
    source_labels = {c['id']: c['label'] for c in r.base.read(r.OUT / 'source_data/private_labels.json')}
    provenance = r.base.read(r.OUT / 'source_data/provenance.json')
    for name, digest in provenance['files'].items(): assert r.base.h(r.OUT / 'source_data' / name) == digest, name
    train = r.base.read(r.OUT / 'train_public.json'); test = r.base.read(r.OUT / 'test_public.json')
    assert len(train) == len(test) == 12
    assert not {c['id'] for c in train} & {c['id'] for c in test}
    assert not {c['question'].strip() for c in train} & {c['question'].strip() for c in test}
    for split, cases in (('train', train), ('test', test)):
        labels = r.base.read(r.OUT / (split + '_labels.json'))
        for c in cases:
            assert c == source_cases[c['id']] and labels[c['id']] == source_labels[c['id']]
            assert set(c) == {'id', 'question', 'choices', 'source_id', 'split'}
            for a in range(3):
                req = r.request(c, 0, a)
                assert set(json.loads(req['messages'][1]['content'])) == {'question', 'choices'}
    final = (r.OUT / 'screen_status.json').exists()
    names = ('test_sampling_coordination_screen', 'test_bandit_policy_learning', 'test_action_contribution_frontier',
        'test_option_binding_check', 'test_full_context_challenge', 'test_targeted_challenge_preflight',
        'test_targeted_challenge', 'test_document_verification', 'test_equilibrium_ranking_reference',
        'test_equilibrium_reference_pipeline', 'test_learned_production_game', 'test_learned_production_runner')
    with patch('socket.socket.connect', side_effect=AssertionError('offline')), patch('socket.getaddrinfo', side_effect=AssertionError('offline')):
        stream = io.StringIO(); tests = unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromNames(
            ['experiments.tests.' + n for n in names]))
        assert tests.wasSuccessful(), stream.getvalue()
        if final:
            def blocked(req): raise AssertionError('no model calls on replay')
            gate = r.base.read(r.OUT / 'action_gate.json')
            tr = r.base.read(r.OUT / 'train_sealed_outputs.json')
            assert r.action_gate(tr, r.base.read(r.OUT / 'train_labels.json')) == gate
            for c in train:
                assert r.collect(c, 'train', blocked, {'used': 0, 'cap': 5}) == tr[c['id']]
            if not gate['passed']:
                assert not list((r.OUT / 'test').rglob('attempts.jsonl'))
                assert not (r.OUT / 'frozen_policy.json').exists()
            else:
                policy = r.base.read(r.OUT / 'frozen_policy.json')
                assert r.train_policy(tr, r.base.read(r.OUT / 'train_labels.json')) == policy
                output = r.base.read(r.OUT / 'test_sealed_outputs.json')
                for c in test:
                    assert r.collect(c, 'test', blocked, {'used': 0, 'cap': 5}, policy) == output[c['id']]
                    folder = r.OUT / 'test' / r.sha256(c['id'].encode()).hexdigest()
                    raws = r.read_rows(folder / 'responses.jsonl')
                    p = r.base.read(folder / 'selected_before_calls.json')['profile']
                    assert [x['key'] for x in raws[:3]] == [f'{i}/{a}' for i, a in enumerate(p)]
            attempts = [x for p in r.OUT.rglob('attempts.jsonl') for x in r.read_rows(p)]
            raws = [x for p in r.OUT.rglob('responses.jsonl') for x in r.read_rows(p)]
            assert len(attempts) == len(raws) == (240 if gate['passed'] else 120)
    assert before == frozen_hashes()
    paths = [p for p in r.OUT.rglob('*') if p.is_file() and p.name not in ('run.lock', 'preparation_acceptance.json', 'final_acceptance.json')]
    paths += [Path(__file__)]
    accepted = {'status': 'screen_completed' if final else 'screen_prepared', 'tests': tests.testsRun,
        'verification_new_calls': 0, 'historical_frozen_files': len(before),
        'sha256': {str(p.relative_to(r.ROOT)): r.base.h(p) for p in paths}}
    r.freeze(r.OUT / ('final_acceptance.json' if final else 'preparation_acceptance.json'), accepted)
    print(json.dumps({k: v for k, v in accepted.items() if k != 'sha256'}, ensure_ascii=False))


if __name__ == '__main__': main()
