import copy
import tempfile
import unittest
from pathlib import Path

from experiments.recommendation_v2.prepare import ROOT, prefix_input, select_case, validate_pair, save
from experiments.recommendation_v2.audit import LexicalIndex, cached
from experiments.scripts.data.build_redial_benchmark import read_dialogues


class RecommendationPrefixTests(unittest.TestCase):
    def setUp(self):
        self.dialogue = {"conversationId": 1, "initiatorWorkerId": 10,
                         "movieMentions": {"1": "Film A (2015)", "2": "Film B (2016)"},
                         "messages": [{"senderWorkerId": 10, "text": "I like action"},
                                      {"senderWorkerId": 20, "text": "Try @1"},
                                      {"senderWorkerId": 10, "text": "@1 was great, but prefer @2"}],
                         "initiatorQuestions": {"1": {"liked": 1, "seen": 1, "suggested": 0},
                                                "2": {"liked": 1, "seen": 0, "suggested": 1}}}
        self.mapping = {"1": "101", "2": "102"}

    def test_future_liking_not_exposed(self):
        inp, label = select_case(self.dialogue, self.mapping)
        self.assertEqual(inp["cutoff_exclusive"], 2)
        self.assertEqual(inp["context_movie_ids"], ["101"])
        self.assertEqual(inp["liked_movie_ids"], [])
        self.assertEqual(inp["excluded_seen_movie_ids"], [])
        self.assertEqual(label["relevant_movie_ids"], ["102"])

    def test_fixed_prefix_invariant_to_annotations(self):
        before = prefix_input(self.dialogue, 2, self.mapping)
        altered = copy.deepcopy(self.dialogue)
        altered["initiatorQuestions"] = {"SECRET": "FUTURE LABEL"}
        altered["respondentQuestions"] = {"SECRET": True}
        self.assertEqual(prefix_input(altered, 2, self.mapping), before)

    def test_fixed_prefix_invariant_to_future_messages(self):
        before = prefix_input(self.dialogue, 2, self.mapping)
        altered = copy.deepcopy(self.dialogue)
        altered["messages"][2]["text"] = "SECRET @2 is awful"
        altered["messages"].append({"senderWorkerId": 10, "text": "SECRET"})
        self.assertEqual(prefix_input(altered, 2, self.mapping), before)

    def test_implicit_liking_not_misclassified_as_absence(self):
        d = copy.deepcopy(self.dialogue)
        d["messages"][1]["text"] = "Have you seen @1?"
        d["messages"].insert(2, {"senderWorkerId": 10, "text": "Yea I have, great movie"})
        inp = prefix_input(d, 3, self.mapping)
        self.assertIn("great movie", inp["query"])
        self.assertEqual(inp["mention_events"][0]["sentiment"], "unknown")
        self.assertEqual(inp["preference_status"], "not_extracted")

    def test_targets_not_in_serving_input(self):
        inp, label = select_case(self.dialogue, self.mapping)
        self.assertNotIn("relevant_movie_ids", inp)
        self.assertNotIn("102", inp["context_movie_ids"])
        validate_pair(inp, label, {"101", "102"})

    def test_validation_rejects_unproven_profile(self):
        inp, label = select_case(self.dialogue, self.mapping)
        inp["liked_movie_ids"] = ["101"]
        with self.assertRaises(ValueError):
            validate_pair(inp, label, {"101", "102"})

    def test_validation_rejects_unknown_movie(self):
        inp, label = select_case(self.dialogue, self.mapping)
        with self.assertRaises(ValueError):
            validate_pair(inp, label, {"101"})

    def test_invalid_cutoff(self):
        with self.assertRaises(ValueError):
            prefix_input(self.dialogue, 9, self.mapping)

    def test_outputs_never_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            save(path, {"x": 1})
            save(path, {"x": 1})
            with self.assertRaises(FileExistsError):
                save(path, {"x": 2})

    def test_retrieval_does_not_use_hidden_labels(self):
        movies = {"1": {"title": "Space Adventure", "genres": ["Sci-Fi"]},
                  "2": {"title": "Romantic Story", "genres": ["Romance"]}}
        index = LexicalIndex(movies)
        inp = {"query": "space", "hard_constraints": {}}
        self.assertEqual(index.search(inp), ["1"])
        inp["relevant_movie_ids"] = ["2"]
        self.assertEqual(index.search(inp), ["1"])

    def test_empty_match_does_not_use_popularity(self):
        index = LexicalIndex({"1": {"title": "Space", "genres": []}})
        self.assertEqual(index.search({"query": "unknownword"}), [])

    def test_hard_constraints_fail_closed_in_diagnostic(self):
        index = LexicalIndex({"1": {"title": "Space", "genres": []}})
        with self.assertRaises(ValueError):
            index.search({"query": "space", "hard_constraints": {"min_year": 2015}})

    def test_checkpoint_refuses_changed_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            save(path, {"dependencies": {"source": "abc"}, "result": 1})
            self.assertEqual(cached(path, {"source": "abc"})["result"], 1)
            with self.assertRaises(ValueError):
                cached(path, {"source": "changed"})

    def test_failed_checkpoint_never_counts_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            save(path, {"dependencies": {}, "pass": False})
            with self.assertRaises(ValueError):
                cached(path, {})

    def test_checkpoint_checks_retrieval_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            save(path, {"dependencies": {}, "artifacts": {"pool.json": "not-the-hash"}})
            save(path.parent / "pool.json", {"x": 1})
            with self.assertRaises(ValueError):
                cached(path, {})


class ActualReDialRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "experiments/data/raw/redial/test.jsonl"
        if not path.exists():
            raise unittest.SkipTest("raw ReDial not installed")
        cls.examples = {str(d["conversationId"]): d for d in read_dialogues(path)
                        if str(d["conversationId"]) in {"22499", "22924", "21008"}}

    def test_22499_future_black_panther_like_is_not_input(self):
        inp = prefix_input(self.examples["22499"], 5, {"204974": "122906"})
        self.assertEqual(inp["liked_movie_ids"], [])
        self.assertEqual(inp["excluded_seen_movie_ids"], [])
        self.assertIn("122906", inp["context_movie_ids"])
        self.assertNotIn("Woow", inp["query"])

    def test_22924_future_silence_feedback_is_not_input(self):
        inp = prefix_input(self.examples["22924"], 5, {"203424": "593", "96140": "27822"})
        self.assertEqual(inp["liked_movie_ids"], [])
        self.assertEqual(inp["excluded_seen_movie_ids"], [])
        self.assertNotIn("pretty creepy", inp["query"])

    def test_21008_valid_implicit_feedback_is_retained(self):
        inp = prefix_input(self.examples["21008"], 7, {"204974": "122906"})
        self.assertIn("Yea I have great movie", inp["query"])
        self.assertEqual(inp["preference_status"], "not_extracted")


if __name__ == "__main__":
    unittest.main()
