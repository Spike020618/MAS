import copy
import json
import unittest
import tempfile
from pathlib import Path

from experiments.consensus_recommendation.common import create, fingerprint
from experiments.paired_deliberation.rapid_negotiation import API, aggregate, local_update, select_target, validate_pair, physical_audit
from experiments.paired_deliberation.durable_provider import DurableProvider
from experiments.paired_deliberation.transport_d2 import parse_d2_raw
from experiments.paired_deliberation.requests_v5 import RequestsV5


class RapidNegotiationTests(unittest.TestCase):
    def test_authentic_quote_does_not_claim_semantic_truth(self):
        req = RequestsV5.envelope('soft_pair_review', {'pair': ['1', '2'],
            'available_sources': {'message:1': {'kind': 'dialogue', 'text': 'I want Comedy.'}}}, 'JSON')
        answer = {'preferred': '2', 'source_id': 'message:1', 'quote': 'I want Comedy.', 'reason': 'A soft forecast.'}
        result = validate_pair(answer, req)
        self.assertTrue(result['quote_authentic'])
        self.assertFalse(result['entailment_verified'])
        self.assertFalse(result['hard_constraint_created'])
        for key, value in [('preferred', '3'), ('source_id', 'history:unseen'), ('quote', 'I hate Comedy.')]:
            bad = {**answer, key: value}
            with self.assertRaises(ValueError): validate_pair(bad, req)

    def test_only_two_scores_of_one_owner_change_and_stale_state_rejected(self):
        vectors = [{'1': .7, '2': .2, '3': .1}, {'1': .1, '2': .8, '3': .1}]
        before = copy.deepcopy(vectors); target = {'a': '1', 'b': '2', 'owner': 0}
        after = local_update(vectors, target, '2', fingerprint(vectors))
        self.assertEqual(vectors, before)
        self.assertEqual(after[1], before[1]); self.assertEqual(after[0]['3'], .1)
        self.assertEqual(after[0]['2'], .7); self.assertEqual(sum(after[0].values()), sum(before[0].values()))
        with self.assertRaisesRegex(ValueError, 'stale'): local_update(after, target, '1', fingerprint(before))

    def test_boundary_disagreement_selected_without_labels_or_repeat(self):
        pool = [str(i) for i in range(10)]
        a = {m: (10-i)/55 for i, m in enumerate(pool)}
        b = copy.deepcopy(a); b['4'], b['5'] = b['5'], b['4']
        target = select_target([a, a, b], pool, set())
        self.assertEqual({target['a'], target['b']}, {'4', '5'})
        self.assertIsNone(select_target([a, a, b], pool, {frozenset(('4', '5'))}))
        self.assertIsNone(select_target([a, a, a], pool, set()))
        self.assertEqual(aggregate([a, a], pool)[0], pool)

    def test_known_failed_request_is_replayed_without_new_network_or_zero_token_imputation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); old = root / 'old'; current = root / 'current'
            req = RequestsV5.envelope('ranking_v6', {'schema': {'ranking': []}}, 'JSON')
            raw = {'status_code': 400, 'body': '{"error":{"code":"provider_refusal"}}', 'elapsed_ms': 1}
            create(old / 'sample.raw.json', {'request': req, 'raw': raw, 'integrity': fingerprint(raw)})
            def forbidden(request): self.fail('known failed request was sent again')
            wrapper = DurableProvider(current / 'glm_flash/slots', {'plan_hash': 'test'}, forbidden,
                                      parse_d2_raw, True, reuse_initial=old, reuse_all=True)
            api = API.__new__(API); api.root = current; api.providers = {'glm_flash': wrapper}
            with self.assertRaises(RuntimeError): api.complete('glm_flash', 'sample', req, {})
            physical, receipts = physical_audit(current)
            self.assertEqual(physical['glm_flash']['attempts'], 0)
            self.assertEqual(physical['glm_flash']['reused_raw_receipts_not_new_requests'], 1)
            self.assertIsNone(receipts['glm_flash:sample'])


if __name__ == '__main__': unittest.main()
