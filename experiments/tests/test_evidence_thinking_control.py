import unittest
from experiments.mechanism import control_evidence_thinking as c


class ThinkingTests(unittest.TestCase):
    def test_only_compute_mode_changes(self):
        req=c.p.request({'id':'x','question':'Test?','native_documents':[]},'native')
        changed=c.thinking(req)
        self.assertEqual(changed['messages'],req['messages']);self.assertEqual(changed['model'],req['model'])
        self.assertEqual(changed['thinking'],{'type':'enabled'});self.assertNotIn('temperature',changed)
        self.assertEqual(changed['max_tokens'],4096);self.assertEqual(req['thinking'],{'type':'disabled'})


if __name__=='__main__':unittest.main()
