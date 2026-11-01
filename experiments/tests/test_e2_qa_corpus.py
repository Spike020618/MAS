import json,tempfile,unittest
from pathlib import Path
from experiments.consensus_recommendation.common import ROOT
from experiments.evidence_consensus.prepare_e2_qa_corpus import imported_sources
from experiments.evidence_consensus.schema import validate_task

class E2QACorpusTests(unittest.TestCase):
 def test_import_rejects_missing_real_material(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'web.json';p.write_text('[{"title":"invented","text":"x"}]')
   with self.assertRaises(ValueError):imported_sources(p)
 def test_import_does_not_carry_answer_fields(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'web.json';p.write_text(json.dumps([{'url':'https://example.org/a','text':'material','answer':True}]))
   self.assertNotIn('answer',imported_sources(p)[0])
 def test_real_eighty_packets_native_scoped_and_exclude_e1(self):
  tasks=json.loads((ROOT/'experiments/results/e2_qa80_corpus_v2/tasks.json').read_text());old=json.loads((ROOT/'experiments/results/evidence_consensus_e1_tasks_v1/tasks.json').read_text());exclude={t['metadata']['native_id']for t in old if t['kind']=='qa'}
  self.assertEqual(len(tasks),80);self.assertEqual(len({t['id']for t in tasks}),80)
  for t in tasks:
   validate_task(t);self.assertNotIn(t['metadata']['native_id'],exclude);self.assertEqual(t['options'],['yes','no']);self.assertNotIn('answer',t)
   for s in t['source_catalog'].values():self.assertFalse(s['semantic_truth_certified']);self.assertTrue(s['url'].startswith('http'))
