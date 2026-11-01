import unittest
from experiments.consensus_mechanism.perturb_probe import perturb

class PerturbTests(unittest.TestCase):
 def test_qa_flip_is_deterministic_and_label_free(self):
  task={'id':'t1','kind':'qa','options':['yes','no']}
  self.assertEqual(perturb(task,'yes')[0],'no');self.assertEqual(perturb(task,'no')[0],'yes')
  self.assertIn('flipped',perturb(task,'yes')[1])
 def test_qa_single_option_left_unchanged(self):
  task={'id':'t2','kind':'qa','options':['yes']}
  decision,note=perturb(task,'yes');self.assertEqual(decision,'yes');self.assertEqual(note,'unperturbed_no_other_option')
 def test_rec_head_swap_shape_and_determinism(self):
  task={'id':'t3','kind':'rec','candidates':[{'movie_id':str(m)}for m in range(20)]}
  decision=[str(m)for m in range(12)]
  corrupted,note=perturb(task,decision)
  self.assertEqual(len(corrupted),len(decision));self.assertNotEqual(corrupted[0],decision[0])
  self.assertEqual(corrupted[0],decision[10]);self.assertEqual(corrupted[10],decision[0]);self.assertEqual(corrupted[1:10],decision[1:10]);self.assertEqual(corrupted[11:],decision[11:])
  self.assertIn('swapped head',note)
  again,note2=perturb(task,decision);self.assertEqual(corrupted,again);self.assertEqual(note,note2)
 def test_rec_ranking_too_short_left_unchanged(self):
  task={'id':'t4','kind':'rec','candidates':[{'movie_id':str(m)}for m in range(5)]}
  decision=['1']
  out,note=perturb(task,decision);self.assertEqual(out,decision);self.assertEqual(note,'unperturbed_ranking_too_short')

if __name__=='__main__':unittest.main()
