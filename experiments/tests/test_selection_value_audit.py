"""Independent regression examples for the audit's official Hotpot answer rules."""
import unittest
from experiments.audits.selection_value_audit import native_score
class NativeAnswerAuditTests(unittest.TestCase):
 def test_multiset_overlap(self):self.assertAlmostEqual(native_score('cat cat','cat'),2/3)
 def test_yes_no_exact_special_answers(self):
  self.assertEqual(native_score('yes','yes'),1)
  self.assertEqual(native_score('yes','no'),0)
  self.assertEqual(native_score('noanswer','answer'),0)
  self.assertEqual(native_score('yes','yes indeed'),0)
 def test_special_word_inside_long_answer_is_not_special_answer(self):
  self.assertAlmostEqual(native_score('no one','one'),2/3)
  self.assertAlmostEqual(native_score('yes man','man'),2/3)
 def test_articles_and_punctuation(self):
  self.assertEqual(native_score('The Eiffel Tower!','Eiffel Tower'),1)
  self.assertEqual(native_score('A CAT, an owl.','cat owl'),1)
if __name__=='__main__':unittest.main()
