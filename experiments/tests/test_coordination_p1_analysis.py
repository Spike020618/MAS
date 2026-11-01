import unittest
from experiments.coordination_game_v1.analysis import analyze
from experiments.coordination_game_v1.schema import digest
class AnalysisTests(unittest.TestCase):
 def data(self):
  rows=[]
  for t in range(12):
   for rep in range(2):
    for kind,n in [('empty',1),('single',9),('same',3),('cross',3)]:
     for i in range(n):rows.append({'task_id':str(t),'repeat':rep,'branch':kind+str(i),'kind':kind,'answer':'yes'if kind!='empty'else'no','valid':True})
  return rows
 def test_partial_no_gold(self):self.assertFalse(analyze(self.data()[:5])['effects_available'])
 def test_seal(self):
  rows=self.data();seal={'records_hash':digest(rows),'expected_records':384,'budget_violations':0,'leakage_events':0}
  r=analyze(rows,{str(i):'yes'for i in range(12)},seal);self.assertTrue(r['engineering_gate']);self.assertEqual(r['solver_gain'],'not_measured');self.assertEqual(r['paired_vs_empty']['single']['n_questions'],12)
 def test_modified_rejected(self):
  with self.assertRaises(ValueError):analyze(self.data(),{}, {'records_hash':'wrong'})
