import unittest

from game_consensus_recsys.catalog import InMemoryCatalog
from game_consensus_recsys.consensus import ConsensusEngine
from game_consensus_recsys.domain import SearchRequest
from game_consensus_recsys.workflow import ConsensusWorkflow
from game_consensus_recsys.planner import ConsensusPlanner


class RequestTests(unittest.TestCase):
    def test_rejects_invalid_year_range(self):
        with self.assertRaises(ValueError):
            ConsensusWorkflow().run(SearchRequest("科幻", min_year=2020, max_year=2010))


class WorkflowTests(unittest.TestCase):
    def test_returns_grounded_top_k_with_auditable_trace(self):
        response = ConsensusWorkflow().run(SearchRequest(
            "2010年后的科幻剧情电影", top_k=3,
            preferred_genres=["科幻", "剧情"], min_year=2010,
        ))
        self.assertEqual(len(response.decisions), 3)
        self.assertTrue(all(item.candidate.year >= 2010 for item in response.decisions))
        self.assertTrue(all(item.accepted_evidence for item in response.decisions))
        self.assertEqual(len(response.trace.reports), 3 * len(InMemoryCatalog().search(
            SearchRequest("2010年后的科幻剧情电影", preferred_genres=["科幻", "剧情"], min_year=2010)
        )))
        self.assertEqual(response.trace.events[-1].event, "ranking_completed")
        self.assertEqual(response.trace.events[0].event, "plan_selected")

    def test_excluded_genre_is_never_recommended(self):
        response = ConsensusWorkflow().run(SearchRequest(
            "科幻电影", top_k=8, excluded_genres=["动作"]
        ))
        self.assertTrue(response.decisions)
        self.assertTrue(all("动作" not in item.candidate.genres for item in response.decisions))

    def test_reputation_is_longitudinal(self):
        engine = ConsensusEngine()
        workflow = ConsensusWorkflow(engine=engine)
        request = SearchRequest("科幻电影", top_k=2)
        first = workflow.run(request)
        second = workflow.run(request)
        self.assertEqual(first.trace.reputations_before, {})
        self.assertNotEqual(first.trace.reputations_after, {})
        self.assertEqual(second.trace.reputations_before, first.trace.reputations_after)


class PlannerTests(unittest.TestCase):
    def test_simple_request_uses_targeted_review(self):
        plan = ConsensusPlanner().plan(SearchRequest("科幻电影"))
        self.assertEqual(plan.topology, "targeted_review")
        self.assertEqual(plan.agent_roles, ["retrieval", "verifier"])

    def test_complex_request_escalates_without_posthoc_features(self):
        plan = ConsensusPlanner().plan(SearchRequest(
            "推荐2010年以后适合多人共同观看但不要动作类型的科幻剧情电影",
            preferred_genres=["科幻", "剧情"], excluded_genres=["动作"],
            min_year=2010, top_k=10,
        ))
        self.assertEqual(plan.risk_level, "high")
        self.assertEqual(plan.topology, "parallel_review_with_escalation")
        self.assertGreaterEqual(plan.verification_budget, 3)


if __name__ == "__main__":
    unittest.main()
