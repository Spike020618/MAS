import unittest

from system.game_consensus_recsys.catalog import InMemoryCatalog
from system.game_consensus_recsys.consensus import ConsensusEngine
from system.game_consensus_recsys.domain import (
    CandidateReport, CandidateSource, Evidence, MovieCandidate, SearchRequest, Verdict,
)
from system.game_consensus_recsys.workflow import ConsensusWorkflow
from system.game_consensus_recsys.planner import ConsensusPlanner


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

    def test_excluded_movie_id_is_never_recommended(self):
        catalog = InMemoryCatalog()
        first = catalog.search(SearchRequest("科幻电影"))[0]
        response = ConsensusWorkflow(catalog=catalog).run(SearchRequest(
            "科幻电影", top_k=8, excluded_movie_ids=[first.movie_id],
        ))
        self.assertTrue(response.decisions)
        self.assertNotIn(first.movie_id, [item.candidate.movie_id for item in response.decisions])

    def test_required_genres_are_hard_constraints_and_top_k_is_not_padded(self):
        response = ConsensusWorkflow().run(SearchRequest(
            "适合亲子观看的温暖科幻动画", top_k=3,
            required_genres=["科幻", "动画"], excluded_genres=["灾难", "动作"],
        ))
        self.assertEqual([item.candidate.title for item in response.decisions], ["机器人总动员"])
        self.assertTrue(all({"科幻", "动画"}.issubset(item.candidate.genres)
                            for item in response.decisions))

    def test_reputation_is_longitudinal(self):
        engine = ConsensusEngine()
        workflow = ConsensusWorkflow(engine=engine)
        request = SearchRequest("科幻电影", top_k=2)
        first = workflow.run(request)
        second = workflow.run(request)
        self.assertEqual(first.trace.reputations_before, {})
        self.assertNotEqual(first.trace.reputations_after, {})
        self.assertEqual(second.trace.reputations_before, first.trace.reputations_after)

    def test_shortlist_has_uniform_agent_coverage(self):
        response = ConsensusWorkflow().run(SearchRequest(
            "科幻电影", top_k=3, preferred_genres=["科幻"],
        ))
        coverage_event = next(event for event in response.trace.events
                              if event.event == "coverage_checked")
        self.assertTrue(coverage_event.payload["uniform"])
        expected = set(coverage_event.payload["required_agents"])
        self.assertTrue(expected)
        self.assertTrue(all(set(value) == expected
                            for value in coverage_event.payload["coverage"].values()))


class ConsensusTests(unittest.TestCase):
    @staticmethod
    def candidate(movie_id, retrieval_score):
        return MovieCandidate(
            movie_id, f"movie-{movie_id}", 2020, ["科幻"],
            [CandidateSource.LEXICAL], retrieval_score,
        )

    @staticmethod
    def report(candidate_id, agent_id="preference-1", role="preference",
               verdict=Verdict.SUPPORT, utility=0.8):
        return CandidateReport(
            agent_id, role, candidate_id, "assumption",
            [Evidence("claim", "test", f"test:{candidate_id}", True, 1.0)],
            "inference", verdict, utility, 0.8,
        )

    def test_candidate_order_does_not_change_scores(self):
        candidates = [self.candidate("a", 0.9), self.candidate("b", 0.6)]
        reports = [self.report("a", utility=0.7), self.report("b", utility=0.9)]
        forward = ConsensusEngine().decide(candidates, reports, 2)
        reverse = ConsensusEngine().decide(list(reversed(candidates)),
                                           list(reversed(reports)), 2)
        self.assertEqual(
            {item.candidate.movie_id: item.score for item in forward},
            {item.candidate.movie_id: item.score for item in reverse},
        )

    def test_verifier_opposition_is_a_hard_gate(self):
        candidate = self.candidate("blocked", 1.0)
        reports = [
            self.report("blocked", utility=1.0),
            self.report("blocked", "verifier-1", "verifier", Verdict.OPPOSE, 0.0),
        ]
        self.assertEqual(ConsensusEngine().decide([candidate], reports, 1), [])

    def test_score_exposes_recomputable_components(self):
        candidates = [self.candidate("a", 0.9), self.candidate("b", 0.6)]
        decisions = ConsensusEngine().decide(
            candidates, [self.report("a"), self.report("b", utility=0.4)], 2,
        )
        self.assertTrue(all({"retrieval", "agent_utility", "evidence_quality",
                             "opposition_penalty"} == set(item.score_components)
                            for item in decisions))


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
