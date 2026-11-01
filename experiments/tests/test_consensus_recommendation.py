from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.consensus_recommendation.common import create, fingerprint, sha, verify_artifacts
from experiments.consensus_recommendation.data import make_input, make_label, worker_split, Retriever
from experiments.consensus_recommendation.protocol import bundle, validate, parse, collect_parse, disputes, prompt, config
from experiments.consensus_recommendation.runner import Budget, Collector
from experiments.consensus_recommendation.game import solve
from experiments.consensus_recommendation.evaluate import metrics, paired, power


def fixture():
    movies = {str(i):{"movie_id":str(i),"title":f"Adventure {i}","year":2000+i,"genres":["Adventure"]}
              for i in range(1,9)}
    inp = {"case_id":"redial:x","observations":[{"source_id":"message:0","message_index":0,"role":"seeker","text":"I want an adventure film"}],
           "context_movie_ids":[],"query":"I want an adventure film"}
    candidates={"case_id":inp["case_id"],"ranked_movie_ids":list(movies)}
    evidence=bundle(inp,candidates,movies)
    report={"ranking":["1","2","3","4","5"],"claims":[{"claim_id":"c1","text":"Adventure requested",
                "source_ids":["message:0"],"candidate_ids":["1"],"stance":"support"}],"updates":[]}
    return movies,inp,candidates,evidence,report


class ConsensusTests(unittest.TestCase):
    def test_game_mixed_and_pure_not_conflated(self):
        result=solve([[1.,0.],[0.,1.]])
        self.assertAlmostEqual(result["value"],.5)
        self.assertAlmostEqual(result["deterministic_minimax_loss"],1.)
        self.assertLess(result["duality_gap"],1e-8)
        self.assertLess(result["recommender_deviation_gain"],1e-8)
        self.assertLess(result["opponent_deviation_gain"],1e-8)

    def test_game_invalid_payoffs_rejected(self):
        for a in ([],[1,2],[[float("nan")]]):
            with self.assertRaises(ValueError): solve(a)

    def test_metrics_and_pairing(self):
        self.assertEqual(metrics(["x","y"],["x"])["hit"],1)
        self.assertEqual(metrics([],["x"])["hit"],0)
        self.assertAlmostEqual(metrics(["x","y"],["x"])["ndcg"],1.)
        stats=paired([1,0,1],[0,1,1],["a","b","a"])
        self.assertEqual(stats["corrected"],1)
        self.assertEqual(stats["harmed"],1)
        self.assertEqual(stats["seeker_clusters"],2)
        self.assertGreater(power(.15,.03),1000)

    def test_immutable_artifact_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"x.json"
            create(p,{"a":1}); create(p,{"a":1})
            with self.assertRaises(ValueError): create(p,{"a":2})
            with self.assertRaises(ValueError): verify_artifacts(Path(tmp),{"x.json":"bad"})

    def test_label_blind_cutoff_and_input(self):
        d={"conversationId":1,"initiatorWorkerId":1,"respondentWorkerId":2,
           "movieMentions":{"10":"Adventure 1 (2001)","11":"Adventure 2 (2002)"},
           "messages":[{"senderWorkerId":1,"text":"I liked @10 but want adventure"},
                       {"senderWorkerId":2,"text":"try @11"},{"senderWorkerId":1,"text":"later"}],
           "initiatorQuestions":{"11":{"suggested":1,"liked":1}}}
        exact={("adventure 1",2001):["1"],("adventure 2",2002):["2"]}
        titles={"adventure 1":["1"],"adventure 2":["2"]}
        inp=make_input(d,exact,titles)
        changed=copy.deepcopy(d)
        changed["initiatorQuestions"]={"10":{"liked":1}}
        changed["messages"][2]["text"]="Future preferences completely different"
        self.assertEqual(inp,make_input(changed,exact,titles))
        self.assertEqual(inp["cutoff_exclusive"],1)
        self.assertEqual(make_label(d,inp,exact,titles)["relevant_movie_ids"],["2"])

    def test_worker_split_fixed_and_no_role_dependence(self):
        for i in range(100):
            self.assertEqual(worker_split(i),worker_split(str(i)))
            self.assertIn(worker_split(i),("train","dev","reserved_test"))

    def test_bundle_does_not_serialize_extra_labels(self):
        movies,inp,cands,evidence,_=fixture()
        inp["relevant_movie_ids"]=["7"]; inp["future_dialogue"]="secret"
        self.assertEqual(bundle(inp,cands,movies),evidence)
        self.assertNotIn("future_dialogue",json.dumps(prompt(evidence,"test")))

    def test_same_parser_for_every_arm(self):
        *_,evidence,report=fixture()
        self.assertEqual(parse("```json\n"+json.dumps(report)+"\n```",evidence),report)
        for raw in ("not json",json.dumps({**report,"extra":"x"})):
            with self.assertRaises((ValueError,json.JSONDecodeError)): parse(raw,evidence)
        self.assertEqual(parse(json.dumps({**report,"type":"review"}),evidence),report)

    def test_columnar_evidence_is_lossless_and_targets_qualified(self):
        *_,ev,r=fixture()
        payload=json.loads(prompt(ev,"review",{"current_request":r})[1]["content"])
        rendered=payload["evidence"]
        restored=[dict(zip(rendered["candidate_columns"],row)) for row in rendered["candidate_rows"]]
        self.assertEqual(restored,[{k:v for k,v in m.items() if k!="source_id"}
                                   for m in ev["candidates_in_shared_retrieval_order"]])
        self.assertEqual(payload["allowed_update_target_ids"],["current_request:c1"])

    def test_unknown_candidate_and_duplicates(self):
        *_,ev,r=fixture()
        for ranking in (["1"]*5,["1","2","3","4","99"]):
            with self.assertRaises(ValueError): validate({**r,"ranking":ranking},ev)

    def test_fabricated_sources_rejected(self):
        *_,ev,r=fixture()
        for refs in ([],["message:100"],["movie:99"]):
            changed=copy.deepcopy(r); changed["claims"][0]["source_ids"]=refs
            with self.assertRaises(ValueError): validate(changed,ev)

    def test_bad_claim_does_not_manufacture_accuracy_loss(self):
        *_,ev,r=fixture()
        r["claims"][0]["source_ids"]=["movie:999"]
        parsed,issues=collect_parse(json.dumps(r),ev)
        self.assertEqual(parsed["ranking"],r["ranking"])
        self.assertEqual(parsed["claims"][0]["source_ids"],["movie:999"])
        self.assertTrue(issues)
        r["ranking"]=["1"]*5
        with self.assertRaises(ValueError): collect_parse(json.dumps(r),ev)

    def test_updates_need_real_prior_and_source(self):
        *_,ev,r=fixture()
        changed=copy.deepcopy(r)
        changed["updates"]=[{"target_claim_id":"current_request:c1","decision":"unresolved","reason":"not enough evidence","source_ids":["message:0"]}]
        validate(changed,ev,{"current_request":r})
        with self.assertRaises(ValueError): validate(changed,ev)
        self.assertEqual(r["updates"],[])

    def test_no_forced_claim_or_disagreement(self):
        *_,ev,r=fixture()
        r["claims"]=[]
        validate(r,ev)
        self.assertEqual(disputes({k:copy.deepcopy(r) for k in ("current_request","history_applicability","movie_fit")}),[])

    def test_dispute_order_is_bounded_and_label_free(self):
        *_,ev,r=fixture()
        reports={k:copy.deepcopy(r) for k in ("current_request","history_applicability","movie_fit")}
        reports["movie_fit"]["claims"][0]["stance"]="oppose"
        chosen=disputes(reports)
        self.assertEqual(len(chosen),2)
        self.assertEqual(chosen[0]["target_claim_id"],"movie_fit:c1")

    def test_budget_refuses_before_request(self):
        b=Budget()
        with self.assertRaises(ValueError): b.request([{"content":"x"*config()["path_input_upper_token_budget"]}])
        with self.assertRaises(ValueError): b.charge(None)

    def test_retrieval_uses_training_labels_not_query_labels(self):
        movies,inp,*_=fixture()
        train=[{**inp,"case_id":f"t{i}"} for i in range(3)]
        labels=[{"case_id":f"t{i}","relevant_movie_ids":[str(i+1)]} for i in range(3)]
        retriever=Retriever(train,labels,movies)
        a=retriever.search(inp)
        self.assertEqual(a,retriever.search({**inp,"relevant_movie_ids":["8"]}))

    def test_cached_slot_integrity_and_no_network(self):
        *_,ev,r=fixture()
        settings={"model":"fixed","url":"https://api.deepseek.com/v1/chat/completions"}
        class Response:
            status_code=200
            def json(self):
                return {"model":"fixed-version","usage":{"prompt_tokens":100,"completion_tokens":100},
                        "choices":[{"finish_reason":"stop","message":{"content":json.dumps(r)}}]}
        with tempfile.TemporaryDirectory() as tmp, patch("requests.post",return_value=Response()) as post:
            collector=Collector(Path(tmp),settings,"plan")
            messages=prompt(ev,"test")
            a,_=collector.slot("x","slot",messages,ev,Budget())
            b,_=collector.slot("x","slot",messages,ev,Budget())
            self.assertEqual(a,b); self.assertEqual(post.call_count,1)
            path=Path(tmp)/"slots/x/slot.json"
            row=json.loads(path.read_text()); row["record"]["model_returned"]="tampered"
            path.write_text(json.dumps(row))
            with self.assertRaises(ValueError): collector.slot("x","slot",messages,ev,Budget())

    def test_unknown_billing_intent_is_not_retried(self):
        *_,ev,_=fixture()
        with tempfile.TemporaryDirectory() as tmp, patch("requests.post") as post:
            create(Path(tmp)/"slots/x/s.intent.json",{"started":True})
            collector=Collector(Path(tmp),{"model":"fixed","url":"https://api.deepseek.com/v1/chat/completions"},"plan")
            with self.assertRaises(RuntimeError): collector.slot("x","s",prompt(ev,"test"),ev,Budget())
            post.assert_not_called()


if __name__=="__main__": unittest.main()
