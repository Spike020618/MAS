"""Offline preparation/final acceptance. No new model requests."""
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile

from experiments.mechanism import run_option_binding_check as run
from experiments.mechanism.complete_legacy_gain_audit import frozen_hashes


def main():
    original = frozen_hashes(); protocol = run.verify()
    selection = run.base.read(run.OUT / 'selection.json')
    cases = run.base.read(run.OUT / 'public_cases.json')
    labels = run.base.read(run.OUT / 'private_labels.json')
    assert len(cases) == len({c['article_id'] for c in cases}) == len({c['article_sha256'] for c in cases}) == 12
    assert not {c['article_id'] for c in cases} & set(selection['excluded_article_ids'])
    assert not {c['article_sha256'] for c in cases} & set(selection['excluded_article_hashes'])
    assert run.base.h(run.base.ARCHIVE) == selection['archive_sha256']
    for rel, value in {**selection['journal_sha256'], **selection['prior_public_cases_sha256']}.items():
        assert run.base.h(run.ROOT / rel) == value, rel
    with zipfile.ZipFile(run.base.ARCHIVE) as z:
        source = [json.loads(line) for line in z.read('QuALITY.v1.0.1.htmlstripped.dev').decode().splitlines() if line.strip()]
    for c in cases:
        assert set(c) == run.PUBLIC
        matches = [(r, q) for r in source if str(r['article_id']) == c['article_id']
                   for q in r['questions'] if q['question'] == c['question']]
        assert any(r['article'] == c['article'] and dict(zip('ABCD', q['options'])) == c['choices']
                   and 'ABCD'[int(q['gold_label']) - 1] == labels[c['id']] for r, q in matches)
    modules = ('test_option_binding_check', 'test_full_context_challenge', 'test_targeted_challenge_preflight',
               'test_targeted_challenge', 'test_document_verification', 'test_equilibrium_ranking_reference',
               'test_equilibrium_reference_pipeline', 'test_learned_production_game', 'test_learned_production_runner')
    final = (run.OUT / 'result.json').exists()
    with patch('socket.socket.connect', side_effect=AssertionError('offline')), \
         patch('socket.getaddrinfo', side_effect=AssertionError('offline')):
        stream = io.StringIO()
        tests = unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromNames(
            ['experiments.tests.' + name for name in modules]))
        assert tests.wasSuccessful(), stream.getvalue()
        if final:
            def blocked(req): raise AssertionError('no new calls during replay')
            outputs = run.base.read(run.OUT / 'sealed_outputs.json')
            for c in cases:
                assert run.collect(c, run.OUT, blocked, {'used': 0, 'cap': 5}) == outputs[c['id']]
                folder = run.OUT / c['id'].replace(':', '_')
                raw = run.read_rows(folder / 'responses.jsonl')
                assert len(raw) == len(run.read_rows(folder / 'attempts.jsonl')) == 7
                idx = {r['key']: r for r in raw}
                for arm in run.ARMS:
                    data = json.loads(idx[arm + '/final']['request']['messages'][1]['content'])
                    assert data['review_report'] == outputs[c['id']]['reports'][arm]['substantive']
                    assert data['initial'] == {k: outputs[c['id']]['initial'][k] for k in ('answer', 'reasoning')}
                    assert set(data) == {'article', 'question', 'choices', 'initial', 'review_report'}
                    assert data['article'] == c['article']
                assert len({idx[a + '/final']['request']['messages'][0]['content'] for a in run.ARMS}) == 1
            with patch('builtins.print'): run.analyze()
    assert frozen_hashes() == original
    paths = [p for p in run.OUT.rglob('*') if p.is_file() and p.name not in
             ('run.lock', 'preparation_acceptance.json', 'final_acceptance.json')]
    paths += [Path(__file__)]
    acceptance = {'status': 'completed_exploratory_screen' if final else 'ready_to_execute_frozen_screen',
        'tests': tests.testsRun, 'verification_new_calls': 0, 'original_frozen_files': len(original),
        'groups': 12, 'planned_calls': 84, 'scientific_success_not_assumed': True,
        'sha256': {str(p.relative_to(run.ROOT)): run.base.h(p) for p in paths}}
    run.freeze(run.OUT / ('final_acceptance.json' if final else 'preparation_acceptance.json'), acceptance)
    print(json.dumps({k: v for k, v in acceptance.items() if k != 'sha256'}, ensure_ascii=False))


if __name__ == '__main__': main()
