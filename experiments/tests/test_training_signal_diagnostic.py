import unittest

from experiments.paired_deliberation.evaluate_training_signals import metrics, rankings


class SignalDiagnosticTests(unittest.TestCase):
    def test_absent_signal_preserves_prior_instead_of_creating_a_new_order(self):
        trace = {"candidate_signals": [{"movie_id": m, "context_transfer_score": 0.0, "association_score": 0.0}
                                        for m in ["9", "2", "7", "1", "6", "4"]]}
        result = rankings(trace)
        self.assertEqual(result["context_score"], result["frozen_retrieval"])
        self.assertEqual(result["association_score"], result["frozen_retrieval"])

    def test_unannotated_is_unknown_but_out_of_pool_target_is_a_miss(self):
        ranking = ["1", "2", "3", "4", "5"]
        self.assertIsNone(metrics(ranking, []))
        self.assertEqual(metrics(ranking, ["outside"]), {"Hit@5": 0, "Recall@5": 0.0, "NDCG@5": 0.0})


if __name__ == "__main__":
    unittest.main()
