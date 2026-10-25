import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from system.game_consensus_recsys.catalog import InMemoryCatalog
from system.game_consensus_recsys.deepseek import DeepSeekClient, DeepSeekConfig, DeepSeekError
from system.game_consensus_recsys.deepseek_agents import DeepSeekReviewAgent
from system.game_consensus_recsys.domain import SearchRequest, Verdict
from system.game_consensus_recsys.movie_data import HybridMovieCatalog
from system.game_consensus_recsys.workflow import ConsensusWorkflow


class DeepSeekClientTests(unittest.TestCase):
    def test_json_client_uses_configured_model_and_hides_key_from_result(self):
        seen = {}

        def transport(request, timeout):
            seen["authorization"] = request.headers["Authorization"]
            seen["payload"] = json.loads(request.data.decode())
            return json.dumps({
                "model": "deepseek-v4-flash",
                "choices": [{"message": {"content": '{"reports":[]}'}}],
                "usage": {"total_tokens": 12},
            }).encode()

        client = DeepSeekClient(DeepSeekConfig("secret", max_retries=0), transport)
        result = client.complete_json("json system", "{}")
        self.assertEqual(seen["payload"]["model"], "deepseek-v4-flash")
        self.assertEqual(seen["payload"]["response_format"], {"type": "json_object"})
        self.assertNotIn("secret", json.dumps(result))

    def test_environment_without_key_is_offline(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(DeepSeekConfig.from_environment(load_project_env=False))
            with patch("system.game_consensus_recsys.deepseek._load_project_env"):
                workflow = ConsensusWorkflow.from_environment()
            self.assertEqual(len(workflow.agents), 3)


class DeepSeekAgentTests(unittest.TestCase):
    def test_one_client_can_back_a_logical_agent(self):
        catalog = InMemoryCatalog()
        request = SearchRequest("科幻电影", preferred_genres=["科幻"])
        candidates = catalog.search(request)[:2]

        def transport(_request, _timeout):
            reports = [{
                "candidate_id": item.movie_id,
                "assumption": "只使用输入",
                "evidence_claims": [f"{item.title}包含科幻类型"],
                "inference": "匹配显式偏好",
                "verdict": "support",
                "utility": 0.8,
                "confidence": 0.7,
            } for item in candidates]
            return json.dumps({
                "choices": [{"message": {"content": json.dumps({"reports": reports}, ensure_ascii=False)}}],
                "usage": {"total_tokens": 100},
            }, ensure_ascii=False).encode()

        client = DeepSeekClient(DeepSeekConfig("secret", max_retries=0), transport)
        agent = DeepSeekReviewAgent(client, "preference", "ds-pref")
        result = agent.review(request, candidates)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(item.verdict == Verdict.SUPPORT for item in result))
        self.assertTrue(all(item.token_cost == 100 for item in result))

    def test_api_failure_degrades_to_deterministic_agents(self):
        class BrokenAgent:
            agent_id = "broken"
            role = "critic"
            def review(self, request, candidates):
                raise DeepSeekError("timeout")

        catalog = InMemoryCatalog()
        workflow = ConsensusWorkflow(catalog=catalog, agents=[
            *ConsensusWorkflow(catalog=catalog).agents, BrokenAgent()
        ])
        response = workflow.run(SearchRequest("科幻电影", top_k=2))
        self.assertEqual(len(response.decisions), 2)
        degraded = [event for event in response.trace.events if event.event == "agent_degraded"]
        self.assertEqual(degraded[0].payload["fallback"], "deterministic_agents_continue")


class HybridRetrievalTests(unittest.TestCase):
    def test_train_association_retrieves_and_hard_exclusion_blocks_movie(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            rows = [
                {"movie_id": "a", "title": "Anchor", "year": 2000, "genres": ["Drama"],
                 "rating_count": 10, "rating_mean": 4.0},
                {"movie_id": "b", "title": "Neighbor", "year": 2001, "genres": ["Drama"],
                 "rating_count": 8, "rating_mean": 3.8},
            ]
            catalog_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            associations = root / "associations.json"
            associations.write_text(json.dumps({
                "associations": {"a": [{"movie_id": "b", "count": 2, "score": 1.0}]}
            }))
            catalog = HybridMovieCatalog(catalog_path, associations)
            found = catalog.search(SearchRequest("unmatched", liked_movie_ids=["a"]))
            self.assertEqual(found[0].movie_id, "b")
            blocked = catalog.search(SearchRequest(
                "unmatched", liked_movie_ids=["a"], excluded_movie_ids=["b"]
            ))
            self.assertNotIn("b", [item.movie_id for item in blocked])


if __name__ == "__main__":
    unittest.main()
