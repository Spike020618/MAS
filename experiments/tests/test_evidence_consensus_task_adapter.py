import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch

from experiments.evidence_consensus.task_adapter import (
    adapt_source, source_index, qa_tasks, rec_tasks, validate_generation_seal,
    load_evaluation_labels, evaluate_decision,
)


class UnifiedTaskAdapterTests(unittest.TestCase):
    def test_source_scope_and_discovery_never_claim_truth_or_reveal_body(self):
        old={'text':'Other user liked movie','kind':'historical_prediction_basis',
             'allowed_use':'prediction_basis','origin_case_id':'other:1'}
        card=adapt_source('h:1',old,'e1:rec:current:1')
        self.assertEqual(card['scope']['subject'],'other_user')
        self.assertEqual(card['scope']['task_id'],'e1:rec:current:1')
        self.assertEqual(card['origin_case_id'],'other:1')
        self.assertFalse(card['semantic_truth_certified'])
        self.assertNotIn('scope',old)
        index=source_index({'source_catalog':{'h:1':card}})
        self.assertNotIn('text',index[0]);self.assertNotIn('Other user liked movie',json.dumps(index))

    def test_incomplete_seal_blocks_label_access(self):
        tasks=[{'id':'qa:1','kind':'qa','metadata':{'native_id':'StrategyQA-official/q'}}]
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'seal.json';p.write_text(json.dumps({'task_ids':[]}))
            with patch('experiments.evidence_consensus.task_adapter.read',return_value={'task_ids':[]})as reader:
                with self.assertRaisesRegex(ValueError,'all-task'):
                    load_evaluation_labels(tasks,generation_complete=p,qa_path=Path(d)/'gold.json')
                self.assertEqual(reader.call_count,1)

    def test_post_seal_native_boolean_and_repeated_rec_targets(self):
        qa={'id':'e1:qa:x','kind':'qa','metadata':{'native_id':'StrategyQA-official/x'}}
        rec={'id':'e1:rec:c:1','kind':'rec','candidates':[{'movie_id':'7'},{'movie_id':'8'}],
             'metadata':{'native_id':'c:1'}}
        with tempfile.TemporaryDirectory()as d:
            d=Path(d);seal=d/'seal.json';seal.write_text(json.dumps({'task_ids':[qa['id'],rec['id']]}))
            q=d/'qa.json';q.write_text(json.dumps([{'qid':'x','answer':True}]))
            r=d/'rec.json';r.write_text(json.dumps([{'conv_id':'c','dialog':[
                {'role':'Seeker','text':['hello'],'movies':[]},
                {'role':'Recommender','text':['x'],'movies':['movie7','movie7','movie8']},
            ]}]))
            e=d/'entities.json';e.write_text(json.dumps({'movie7':7,'movie8':8}))
            gold=load_evaluation_labels([qa,rec],generation_complete=seal,qa_path=q,rec_path=r,entity_path=e)
            self.assertEqual(gold[qa['id']]['answer'],'yes')
            self.assertEqual(gold[rec['id']]['targets'],['7','7','8'])
            self.assertEqual(evaluate_decision(rec,['7'],gold[rec['id']])['hits_at_10'],2)
            self.assertFalse(evaluate_decision(qa,'abstain',gold[qa['id']])['correct'])

    def test_real_prepared24_are_native_scoped_and_answer_free(self):
        qa,skipped=qa_tasks(12);rec=rec_tasks(12)
        self.assertEqual(len(qa+rec),24)
        for t in qa:
            self.assertEqual(t['options'],['yes','no'])
            self.assertEqual(t['initial'],'abstain')
            self.assertNotIn('answer',t);self.assertNotIn('gold',t)
            self.assertTrue(any(c.get('url','').startswith('https://')for c in t['source_catalog'].values()))
        for t in rec:
            self.assertGreaterEqual(len(t['candidates']),100)
            self.assertEqual(len(t['initial']),len(t['candidates']))
        for t in qa+rec:
            self.assertFalse(t['metadata']['query_labels_available'])
            for c in t['source_catalog'].values():
                self.assertEqual(c['scope']['task_id'],t['id'])
                self.assertTrue(c['text']);self.assertFalse(c['semantic_truth_certified'])

    def test_duplicate_seal_ids_cannot_authorize_partial_evaluation(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'s.json';p.write_text(json.dumps({'task_ids':['a','a']}))
            with self.assertRaises(ValueError):validate_generation_seal([{'id':'a'},{'id':'b'}],p)

if __name__=='__main__':unittest.main()
