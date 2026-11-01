import tempfile,unittest
from pathlib import Path
from experiments.evidence_consensus.action_calibration import extract,features,quality

class ActionCalibrationTests(unittest.TestCase):
 def test_features_have_no_outcome_input(self):
  self.assertEqual(features('rec',{'score':.2,'source_characters':100,'reward':999},2,3),[1.,.2,100.,2.,3.])
 def test_repeated_native_targets_preserved(self):
  self.assertEqual(quality(['1','2'],{'kind':'rec','targets':[1,1,3]}),2)
 def test_unsealed_run_cannot_read_labels(self):
  with tempfile.TemporaryDirectory()as d:
   with self.assertRaises(FileNotFoundError):extract(Path(d))
