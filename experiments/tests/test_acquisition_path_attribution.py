import json
import unittest
from experiments.mechanism import attribute_acquisition_paths as a


class PathTests(unittest.TestCase):
    def test_complete_menu_preserves_upstream_dependency(self):
        requests=a.p.base.read(a.OUT/'requests.json')
        self.assertEqual(len(requests),16)
        for arms in requests.values():
            self.assertEqual(set(arms),set(map(str,range(8))))
            self.assertEqual(arms['4'],arms['5'])
            self.assertEqual(arms['6'],arms['7'])

    def test_all_raw_payloads_remain_evidence_only(self):
        for arms in a.p.base.read(a.OUT/'requests.json').values():
            for req in arms.values():
                value=json.loads(req['messages'][1]['content'])
                self.assertEqual(set(value),{'question','documents'})
                self.assertEqual(req['model'],'deepseek-flash')
                for d in value['documents']:
                    self.assertEqual(set(d),{'docid','title','text','text_sha256'})


if __name__=='__main__':unittest.main()
