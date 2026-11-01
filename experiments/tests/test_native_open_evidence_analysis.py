import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.consensus_recommendation.common import fingerprint,ROOT
from experiments.paired_deliberation.native_open_evidence_analysis import load_sealed,paired,physical,resolve_receipt,generation_audit

class OpenAnalysisTests(unittest.TestCase):
    def test_missing_complete_marker_prevents_all_label_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Path,'read_text',side_effect=AssertionError('must stop before any data load')):
                with self.assertRaisesRegex(RuntimeError,'labels forbidden'):load_sealed(Path(directory))
    def test_bootstrap_clusters_multiple_prefixes_of_one_dialogue(self):
        labels=[{'id':'a:2','targets':[1,1]},{'id':'a:4','targets':[2]}]
        result=paired(labels,[['1'],['2']],[[],[]],replicates=20)
        self.assertEqual(result['dialogue_clusters'],1);self.assertEqual(result['targets'],3)
        self.assertEqual(result['gained_hits@10'],3);self.assertEqual(result['lost_hits@10'],0)
        self.assertEqual(result['recall@10_cluster95_interval'],[1,1])
    def test_cost_counts_invalid_paid_outputs_and_separates_429_unknown_and_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);slots=root/'slots';slots.mkdir()
            def put(name,status,body,reused=False):
                raw={'status_code':status,'body':json.dumps(body)};entry={'raw':raw,'integrity':fingerprint(raw)}
                if reused:entry['reused_from']='parent/slots/old.raw.json'
                (slots/(name+'.raw.json')).write_text(json.dumps(entry))
            put('bad_format',200,{'usage':{'prompt_tokens':100,'completion_tokens':10},'output':'not a valid ranking'})
            put('rate_limit',429,{'error':'limited'});put('cached',200,{'usage':{'prompt_tokens':900,'completion_tokens':90}},True)
            (slots/'timeout.intent.json').write_text('{}');result=physical(root)
            self.assertEqual(result['known_usage_including_invalid'],{'prompt_tokens':100,'completion_tokens':10})
            self.assertEqual(result['new_physical_attempts'],3);self.assertEqual(result['reused_responses'],1)
            self.assertEqual(result['http_status'],{'200':1,'429':1});self.assertEqual(len(result['new_unknown_intents']),1)
            self.assertAlmostEqual(result['configured_known_cny'],.00003)
    def test_parent_unknown_is_resolved_without_constructing_a_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'new';parent=Path(directory)/'old';(root/'slots').mkdir(parents=True);(parent/'slots').mkdir(parents=True)
            (root/'plan.json').write_text(json.dumps({'reuse_root':str(parent)}));request={'labels_available':False}
            (parent/'slots/a.intent.json').write_text(json.dumps({'request':request}))
            result=resolve_receipt(root,'a',request);self.assertEqual(result['status'],'unknown_completion');self.assertEqual(result['path'],str(parent/'slots/a.intent.json'))
    def test_frozen_source_audit_does_not_load_labels(self):
        root=ROOT/'experiments/results/native_open_evidence_screen_v2'
        if not(root/'generation_complete.json').exists():self.skipTest('sealed paid fixture not staged')
        plan,inputs,cases=load_sealed(root);original=Path.read_text
        def deny_labels(path,*args,**kwargs):
            if 'labels' in path.name:raise AssertionError('label access during source replay')
            return original(path,*args,**kwargs)
        with patch.object(Path,'read_text',deny_labels):result=generation_audit(root,plan,inputs,cases)
        self.assertGreater(result['valid_events_replayed'],0);self.assertFalse(result['source_identity_is_not_entailment'] is False)

if __name__=='__main__':unittest.main()
