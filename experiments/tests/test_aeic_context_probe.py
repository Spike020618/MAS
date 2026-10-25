import json
import unittest
from experiments.mechanism.probe_aeic_context import prepare,request,ARMS


class ContextProbeTest(unittest.TestCase):
    def test_selection_not_outcome_based(self):
        items,_=prepare()
        self.assertEqual(len(items),12)
        self.assertEqual(sum(i['group']=='valid_disagreement' for i in items),7)
        self.assertTrue(all('label' not in i['case'] for i in items))

    def test_only_context_treatment_changes(self):
        item=prepare()[0][0];reqs=[request(item,0,a) for a in ARMS]
        for req in reqs:
            self.assertEqual(req['messages'][0],reqs[0]['messages'][0])
            for field in ['model','backend','temperature','max_tokens','response_format']:
                self.assertEqual(req[field],reqs[0][field])
        blind,c,full=[json.loads(r['messages'][1]['content']) for r in reqs]
        self.assertNotIn('peer_reports',blind)
        for a,b in zip(c['peer_reports'],full['peer_reports']):
            self.assertNotIn('available_AEI',a)
            self.assertEqual(a['C'],b['C'])

    def test_missing_not_fabricated(self):
        items,_=prepare()
        for item in items:
            for r in item['reports']:
                self.assertEqual(set(r['available_AEI'])|set(r['unavailable_AEI']),{'A','E','I'})
                self.assertFalse(set(r['available_AEI'])&set(r['unavailable_AEI']))


if __name__=='__main__':unittest.main()
