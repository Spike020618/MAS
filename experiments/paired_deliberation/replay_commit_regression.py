"""Replay recorded observations through commit-v4 without inventing new proposals."""
import argparse
import copy
import json
from pathlib import Path

from experiments.consensus_recommendation.common import create, fingerprint, sha
from experiments.paired_deliberation.controlled_effect import verify_plan
from experiments.paired_deliberation.evidence_commit_state import EvidenceCommitState


def run(root, output):
    root, output = Path(root), Path(output)
    plan = verify_plan(root)
    inputs = {r['id']: r for r in json.loads((root / 'inputs.json').read_text())}
    labels = json.loads((root / 'labels.json').read_text())
    if fingerprint(labels) != plan['labels_hash']: raise ValueError('labels changed')
    rows = []
    for label in labels:
        saved_path = root / 'cases' / (label['id'] + '.json')
        saved = json.loads(saved_path.read_text())
        if saved['integrity'] != fingerprint(saved['result']): raise ValueError('case changed')
        for method in ('verified_fixed', 'verified_adaptive'):
            original = saved['result']['methods'][method]
            if original['status'] != 'completed': raise ValueError('source incomplete')
            result = original['result']
            item = inputs[label['id']]
            state = EvidenceCommitState(item['shared']['initial'], item['evidence'])
            for step in result['schedule']:
                target = step['target']
                record = result['verification_cache'][fingerprint(state.claims[target]['claim'])]
                class Recorded:
                    def verify(self, claim): return copy.deepcopy(record)
                state.observe(state.claims[target]['owner'], target, fingerprint(state.claims[target]), Recorded())
            rejection_replays = []
            for rejection in result['rejected']:
                if 'proposal' not in rejection: continue
                target = rejection['target']
                before = copy.deepcopy(state.claims)
                class Never:
                    def verify(self, claim): raise AssertionError('stale proposal must not request new verification')
                try:
                    # Original proposal, INCLUDING its original stale hash. No
                    # silent rebinding or claim of an online v4 repair outcome.
                    state.propose(state.claims[target]['owner'], target, rejection['proposal'], Never())
                except (ValueError, KeyError, TypeError) as exc:
                    rejection_replays.append({'target': target, 'error': str(exc),
                                              'observation_preserved': state.claims == before})
                else: raise AssertionError('old proposal unexpectedly accepted')
            target = label['target']
            checked = label['target_hash'] in result['verification_cache']
            wanted = {'supported': 'supported', 'refuted': 'retracted', 'insufficient': 'unresolved'}[label['expected_verdict']]
            rows.append({'id': label['id'], 'method': method, 'target_checked': checked,
                         'target_status': state.claims[target]['status'],
                         'state_matches_label': state.claims[target]['status'] == wanted if checked else None,
                         'original_rejections_replayed': rejection_replays, 'source_case_sha256': sha(saved_path)})
    summary = {}
    for method in ('verified_fixed', 'verified_adaptive'):
        selected = [r for r in rows if r['method'] == method]
        rejected = [j for r in selected for j in r['original_rejections_replayed']]
        summary[method] = {'checked': sum(r['target_checked'] for r in selected),
                           'state_matches_label': sum(r['state_matches_label'] is True for r in selected),
                           'original_rejections_replayed': len(rejected),
                           'observations_preserved': sum(r['observation_preserved'] for r in rejected)}
    result = {'kind': 'recorded_observations_commit_regression', 'source_plan_hash': plan['plan_hash'],
              'summary': summary, 'rows': rows, 'new_api_calls': 0,
              'source_hashes': {p: sha(Path(__file__).parent / p) for p in
                                ('commit_engine.py', 'evidence_commit_state.py', 'replay_commit_regression.py')},
              'limits': ['fixed historical query schedule and judgement outputs',
                         'historical repairs retain original hashes and are not rebound',
                         'no model responses or recommendation ranks generated; semantic errors remain']}
    create(output, result)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.output), indent=2))
