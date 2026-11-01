import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.slate_choice.controls import exact_smr, capture_values, CFRecommender
from experiments.slate_choice.baselines import Recommender, select_slate
from experiments.tests.test_slate_choice import SlateChoiceTests


class ExactControlTests(unittest.TestCase):
    def test_matches_exhaustive_small_games(self):
        rng = np.random.default_rng(77)
        for _ in range(12):
            values = rng.uniform(0.5, 5, size=(4, 7))
            point = values.mean(axis=0)
            initial = select_slate(values, point, np.eye(7), 2, "greedy_minimax_regret")
            result = exact_smr(values, point, 2, initial)
            target = min(float((values.max(axis=1) - values[:, list(s)].max(axis=1)).max())
                         for s in itertools.combinations(range(7), 2))
            self.assertTrue(result["global_optimum_certified_to_tolerance"])
            self.assertAlmostEqual(result["regret"], target, places=7)

    def test_corrects_greedy_complementarity_failure(self):
        values = np.array([[5., 4., 1.], [1., 4., 5.]])
        result = exact_smr(values, np.array([5., 4., 1.]), 2, [1, 0])
        self.assertEqual(result["initial_regret"], 1)
        self.assertEqual(result["regret"], 0)
        self.assertEqual(set(result["chosen"]), {0, 2})

    def test_timeout_does_not_claim_infeasibility(self):
        values = np.array([[5., 4., 1.], [1., 4., 5.]])
        result = exact_smr(values, np.array([5., 4., 1.]), 2, [1, 0], wall_budget=0)
        self.assertFalse(result["global_optimum_certified_to_tolerance"])
        self.assertEqual(result["lower_bound"], 0)
        self.assertEqual(result["upper_bound"], 1)

    def test_cardinality_invalid(self):
        with self.assertRaises(ValueError):
            exact_smr(np.ones((2, 3)), np.ones(3), 2, [0, 0])

    def test_capture_does_not_change_decisions(self):
        fixture = SlateChoiceTests()
        fixture.setUp()
        model = Recommender(fixture.items, fixture.cfg)
        result, arrays = capture_values(model, fixture.inp, 5)
        self.assertEqual(result, model.predict(fixture.inp, 5))
        self.assertEqual(arrays["values"].shape[1], len(result["candidate_ids"]))

    def test_cf_alignment_checked(self):
        fixture = SlateChoiceTests()
        fixture.setUp()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            np.savez_compressed(path, ids=np.array(["wrong"]), features=np.ones((1, 3)))
            with self.assertRaises(ValueError):
                CFRecommender(fixture.items, fixture.cfg, path)


if __name__ == "__main__":
    unittest.main()
