"""Zero-call preparation of 48 object/polarity challenge cases, not independent labels."""
import argparse
import itertools
from pathlib import Path

from experiments.consensus_recommendation.common import create, fingerprint, sha
from experiments.paired_deliberation.atomic_evidence import render
from experiments.paired_deliberation.atomic_relations_v4 import relation_request


def cases():
    rows = []
    for predicate, polarity, obj, mode, scope in itertools.product(
            ('wants', 'likes'), (True, False), ('Mystery', 'Comedy'),
            ('matching', 'opposite', 'conflict'), ('preference_state', 'utterance')):
        other = 'Comedy' if obj == 'Mystery' else 'Mystery'
        verb = {'wants': 'want', 'likes': 'like'}[predicate]
        def sentence(target, positive):
            return 'I ' + (verb if positive else 'do not ' + verb) + ' ' + target + ' tonight.'
        matching, opposite = sentence(obj, polarity), sentence(obj, not polarity)
        # A distractor always expresses the opposite polarity about another object.
        text = (' '.join([matching, opposite]) if mode == 'conflict' else
                matching if mode == 'matching' else opposite) + ' ' + sentence(other, not polarity)
        sources = {'message:1': {'role': 'seeker', 'text': text}}
        atom = {'scope': scope, 'subject': 'seeker', 'predicate': predicate, 'object': obj,
                'polarity': polarity, 'time': 'current', 'reason': 'Constructed original meaning.'}
        claim = {'claim_id': 'c1', 'text': ('The seeker ' if scope == 'preference_state' else 'The seeker said they ')
                 + ((predicate if polarity else 'does not ' + verb) if scope == 'preference_state' else
                    (verb if polarity else 'do not ' + verb)) + ' ' + obj + ' tonight.',
                 'source_ids': ['message:1'], 'candidate_ids': [], 'stance': 'support',
                 'claim_type': 'explicit_preference', 'depends_on': []}
        relation = {'matching': 'supports', 'opposite': 'refutes', 'conflict': 'both'}[mode]
        verdict = ('supported' if mode != 'opposite' else 'insufficient') if scope == 'utterance' else {
            'matching': 'supported', 'opposite': 'refuted', 'conflict': 'insufficient'}[mode]
        row = {'id': 'scope_' + fingerprint([predicate, polarity, obj, mode, scope])[:12],
               'claim': claim, 'sources': sources, 'expected_atom': atom,
               'expected_relations': {'relations': [{'source_id': 'message:1', 'relation': relation,
                   'support_quote': matching if mode != 'opposite' else '',
                   'refute_quote': opposite if mode != 'matching' else ''}],
                   'reason': 'Constructed target-object propositions; other object is irrelevant.', 'usage': {}},
               'expected_verdict': verdict, 'mode': mode}
        rows.append(row)
    return rows


def prepare(root):
    root = Path(root)
    rows = cases()
    # Requests contain fixed atom and sources only; labels stored separately.
    requests = [{'id': r['id'], 'request': relation_request(r['expected_atom'], r['sources'])} for r in rows]
    base = Path(__file__).parent
    plan = {'version': 'commit_v4_calibration_draft', 'cases': len(rows),
            'labels_hash': fingerprint(rows), 'requests_hash': fingerprint(requests),
            'source_hashes': {name: sha(base / name) for name in
                ('atomic_relations_v4.py', 'commit_engine.py', 'evidence_commit_state.py', 'prepare_commit_calibration.py')},
            'new_api_calls': 0, 'online_release': False, 'automatic_pilot_allowed': False,
            'scope': 'object/polarity relation challenge with fixed canonical atom; parser not evaluated here',
            'limitations': ['constructed known-template diagnostic, not independent natural-language accuracy',
                           'historical development failures require separate regression',
                           'a passing relation-only batch cannot release end-to-end dispatch'],
            'required_gates': ['all source schemas and quotations valid', 'all 48 relations and verdicts correct',
                               'zero false support/refutation; conflict state unresolved',
                               'separate original-claim parser calibration and runtime engineering acceptance']}
    plan['plan_hash'] = fingerprint(plan)
    create(root / 'labels.json', rows)
    create(root / 'requests.json', requests)
    create(root / 'plan.json', plan)
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    print(prepare(parser.parse_args().root)['plan_hash'])
