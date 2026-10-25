import unittest
from experiments.mechanism.audit_complementarity_extraction import answer


class ExtractionTests(unittest.TestCase):
    def slot(self,raw,finish='stop'):
        return {'valid':False,'answer':'abstain','http_status':200,'finish_reason':finish,'raw':raw}

    def test_explicit_answer_not_reasoning(self):
        self.assertEqual(answer(self.slot('{"answer":"yes","reasoning":"no invalid'),['yes','no']),'yes')

    def test_no_recovery_of_truncation(self):
        self.assertEqual(answer(self.slot('{"answer":"yes"','length'),['yes','no']),'abstain')

    def test_no_scan_later_answer_or_outside_choices(self):
        self.assertEqual(answer(self.slot('Reasoning yes. {"answer":"yes"}'),['yes','no']),'abstain')
        self.assertEqual(answer(self.slot('{"answer":"maybe"'),['yes','no']),'abstain')


if __name__=='__main__':unittest.main()
