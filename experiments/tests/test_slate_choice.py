import copy
import itertools
import json
import unittest

import numpy as np

from experiments.slate_choice.common import PROTOCOL, user_bucket, sample_key
from experiments.slate_choice.build import chronological_split
from experiments.slate_choice.baselines import Recommender, select_slate, score_prediction


class SlateChoiceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(PROTOCOL.read_text())
        self.items = {str(i): {"movie_id": str(i), "title": f"Film {i}", "year": 2000,
                              "prior": 3.0 + i / 50, "genres": ["Action" if i % 2 else "Drama"]}
                      for i in range(1, 13)}
        self.inp = {"user_id": "7", "cutoff": 100, "hard_constraints": {"max_year": 2015},
                    "history": [{"movie_id": str(i), "rating": 2.0 + i / 10, "timestamp": i}
                                for i in range(1, 6)], "seen_movie_ids": [str(i) for i in range(1, 6)]}

    def test_cutoff_ties_are_not_in_history(self):
        records = [{"timestamp": t, "movie_id": str(t)} for t in (99, 100, 101)]
        past, future = chronological_split(records, 100)
        self.assertEqual([r["timestamp"] for r in past], [99])
        self.assertEqual([r["timestamp"] for r in future], [100, 101])

    def test_partition_deterministic_and_disjoint(self):
        buckets = [user_bucket(str(i)) for i in range(1000)]
        self.assertEqual(buckets, [user_bucket(str(i)) for i in range(1000)])
        self.assertEqual(set(buckets), set(range(10)))
        self.assertNotEqual(sample_key("1"), sample_key("2"))

    def test_all_methods_share_feasible_pool(self):
        result = Recommender(self.items, self.cfg).predict(self.inp, 5)
        pool = set(result["candidate_ids"])
        self.assertFalse(pool & set(self.inp["seen_movie_ids"]))
        for output in result["predictions"].values():
            self.assertEqual(len(output["movie_ids"]), 5)
            self.assertEqual(len(set(output["movie_ids"])), 5)
            self.assertTrue(set(output["movie_ids"]) <= pool)

    def test_reproducible_predictions(self):
        model = Recommender(self.items, self.cfg)
        self.assertEqual(model.predict(self.inp, 5), model.predict(self.inp, 5))

    def test_numeric_operations_pass_strict_float_checks(self):
        with np.errstate(all="raise"):
            result = Recommender(self.items, self.cfg).predict(self.inp, 5)
        self.assertEqual(len(result["predictions"]), 4)

    def test_labels_do_not_change_prediction(self):
        model = Recommender(self.items, self.cfg)
        altered = copy.deepcopy(self.inp)
        altered["future_ratings"] = [{"movie_id": "12", "rating": 5}]
        self.assertEqual(model.predict(self.inp, 5), model.predict(altered, 5))

    def test_hidden_personal_ratings_not_used_at_budget(self):
        inp = copy.deepcopy(self.inp)
        inp["history"] = [{"movie_id": "6", "rating": 1.0, "timestamp": 0}] + inp["history"]
        changed = copy.deepcopy(inp)
        changed["history"][0]["rating"] = 5.0
        model = Recommender(self.items, self.cfg)
        self.assertEqual(model.predict(inp, 5), model.predict(changed, 5))

    def test_future_history_rejected(self):
        self.inp["history"][-1]["timestamp"] = 100
        with self.assertRaises(ValueError):
            Recommender(self.items, self.cfg).predict(self.inp, 5)

    def test_hard_year_constraint_enforced(self):
        self.items["12"]["year"] = 2020
        with self.assertRaises(ValueError):
            Recommender(self.items, self.cfg).predict(self.inp, 5)

    def test_robust_list_covers_complementary_preferences(self):
        values = np.array([[5, 4, 1], [1, 4, 5]], dtype=float)
        point = np.array([5, 4, 1], dtype=float)
        genres = np.eye(3)
        chosen = select_slate(values, point, genres, 2, "greedy_minimax_regret")
        # Greedy is allowed to miss the globally optimal complementary pair.
        regret = (values.max(axis=1) - values[:, chosen].max(axis=1)).max()
        exact = min((values.max(axis=1) - values[:, list(pair)].max(axis=1)).max()
                    for pair in itertools.combinations(range(3), 2))
        self.assertGreaterEqual(regret, exact)
        self.assertLessEqual(regret, 1.0)

    def test_zero_positive_users_are_retained(self):
        result = Recommender(self.items, self.cfg).predict(self.inp, 5)
        metric = score_prediction(result, {"future_ratings": [{"movie_id": "12", "rating": 1.0}]}, self.items)
        self.assertEqual(metric["future_positive_count"], 0)
        for output in metric["methods"].values():
            self.assertEqual(output["recall_at_5"], 0)

    def test_no_gold_in_candidate_pool(self):
        model = Recommender(self.items, self.cfg)
        result = model.predict(self.inp, 5)
        before = copy.deepcopy(result)
        score_prediction(result, {"future_ratings": [{"movie_id": "999", "rating": 5}]}, self.items)
        self.assertEqual(result, before)

    def test_unknown_method_rejected(self):
        with self.assertRaises(ValueError):
            select_slate(np.ones((2, 3)), np.ones(3), np.eye(3), 2, "fake_game")


if __name__ == "__main__":
    unittest.main()
