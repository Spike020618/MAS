import copy
import json
import unittest

from experiments.paired_deliberation.training_evidence import TrainingEvidence
from experiments.paired_deliberation.training_evidence_view import attach, for_method, render
from experiments.paired_deliberation.contracts_v6 import METHODS
from experiments.paired_deliberation.requests_v5 import RequestsV5


class TrainingEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.movies = {str(i): {"movie_id": str(i), "title": "Film %d" % i, "year": 2000,
            "genres": ["Adventure" if i % 2 == 0 else "Comedy"]} for i in range(1, 7)}
        self.train = [{"case_id": "t1", "query": "seeker likes space adventure", "context_movie_ids": ["1"]},
                      {"case_id": "t2", "query": "seeker likes space adventure", "context_movie_ids": ["1"]},
                      {"case_id": "t3", "query": "seeker likes comedy", "context_movie_ids": ["4"]}]
        self.labels = [{"case_id": "t1", "relevant_movie_ids": ["2"]}, {"case_id": "t2", "relevant_movie_ids": ["3"]},
                       {"case_id": "t3", "relevant_movie_ids": ["2"]}]
        self.backend = TrainingEvidence(self.train, self.labels, self.movies)
        self.inp = {"case_id": "q", "query": "space adventure unique_never_trained_word", "context_movie_ids": ["1"]}
        self.pool = ["6", "4", "2", "5", "3"]

    def test_frozen_pool_is_not_reranked_or_target_injected_and_contributions_sum(self):
        before = copy.deepcopy((self.inp, self.pool, self.train, self.labels))
        trace = self.backend.explain(self.inp, self.pool, visible_neighbors=1)
        self.assertEqual([r["movie_id"] for r in trace["candidate_signals"]], self.pool)
        for row in trace["candidate_signals"]:
            direct = sum(n["similarity"] for n in trace["neighbor_contributions"] if row["movie_id"] in n["historical_targets_in_pool"])
            self.assertAlmostEqual(row["context_transfer_score"], direct)
            self.assertAlmostEqual(row["visible_context_support_mass"] + row["omitted_context_support_mass"], direct)
        self.assertTrue(any(row["omitted_context_support_mass"] > 0 for row in trace["candidate_signals"]))
        self.assertEqual((self.inp, self.pool, self.train, self.labels), before)
        self.assertNotIn("unique_never_trained_word", self.backend.retriever.vectorizer.vocabulary_)

    def test_current_labels_and_training_case_queries_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "evaluation labels"):
            self.backend.explain({**self.inp, "relevant_movie_ids": ["2"]}, self.pool)
        with self.assertRaisesRegex(ValueError, "overlaps training"):
            self.backend.explain({**self.inp, "case_id": "t1"}, self.pool)

    def test_all_five_methods_receive_equal_independent_copies(self):
        view = render(self.backend.explain(self.inp, self.pool))
        supplied = [for_method(view, method) for method in METHODS]
        self.assertTrue(all(value == supplied[0] for value in supplied))
        supplied[0]["candidate_rows"][0][1] = 999
        self.assertNotEqual(supplied[0], supplied[1])

    def test_prediction_evidence_is_separate_and_cannot_enter_first_party_verifier(self):
        view = render(self.backend.explain(self.inp, self.pool))
        raw = {"evidence": {"observations": [{"source_id": "message:0", "text": "I want Adventure."}]}, "schema": {"ranking": []}}
        request = RequestsV5.envelope("ranking_v6", raw, "Return JSON.")
        new = attach(request, view, "independent")
        payload = json.loads(new["messages"][1]["content"])
        self.assertEqual(payload["evidence"], raw["evidence"])
        self.assertIn("training_prediction_evidence", payload)
        self.assertFalse(view["current_case_labels_available"])
        for kind in ("atomic_interpretation", "atomic_source_relations", "verification"):
            with self.assertRaisesRegex(ValueError, "first-party"):
                attach({**request, "kind": kind}, view, "verified_adaptive")


if __name__ == "__main__":
    unittest.main()
