import json
import unittest
from experiments.mechanism.audit_probability_compatibility import compatible_decode


class ProbabilityCompatibilityTest(unittest.TestCase):
    def test_literal_strings_only(self):
        content='{"answer":"no","probabilities":{"yes":"0.2","no":"0.8"}}'
        r=compatible_decode(content,['yes','no'])
        self.assertEqual(r['probabilities'],{'yes':.2,'no':.8})
        self.assertEqual(r['answer'],'no')

    def test_no_normalization_or_guessing(self):
        for distribution in [{'yes':'0.6','no':'0.6'},{'yes':'20%','no':'80%'},
                             {'yes':'0.9','no':'0.1'},{'yes':'NaN','no':'1.0'}]:
            content=json.dumps({'answer':'no','probabilities':distribution})
            self.assertIsNone(compatible_decode(content,['yes','no'])['probabilities'])

    def test_duplicate_keys_not_erased(self):
        content='{"answer":"yes","answer":"no","probabilities":{"yes":"0.2","no":"0.8"}}'
        self.assertIsNone(compatible_decode(content,['yes','no'])['answer'])
