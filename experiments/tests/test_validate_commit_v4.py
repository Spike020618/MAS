import copy
import unittest

from experiments.paired_deliberation.validate_commit_v4 import infer
from experiments.paired_deliberation.prepare_commit_calibration import cases
from experiments.tests.test_design_v6 import evidence, claim, report


class ValidationV4Tests(unittest.TestCase):
    def row(self):
        e=evidence();e['observations']=[{'source_id':'message:1','role':'seeker','text':'I want Drama.'}]
        c=claim(text='The seeker wants Drama.',kind='explicit_preference');c['source_ids']=['message:1']
        return {'id':'offline_fixture','evidence':e,'reports':{'current_request':report([c,claim('sentinel')])}}

    def test_joint_inference_preserves_sentinel_and_uses_two_calls(self):
        class Client:
            def __init__(self):self.calls=[]
            def complete(self,slot,request):
                self.calls.append(request)
                if request['kind']=='atomic_interpretation':
                    obj={'scope':'preference_state','subject':'seeker','predicate':'wants','object':'Drama',
                         'polarity':True,'time':'current','reason':'Original statement.'}
                else:
                    obj={'relations':[{'source_id':'message:1','relation':'supports','support_quote':'I want Drama.',
                                       'refute_quote':''}],'reason':'Matching preference.'}
                return {'report':obj,'usage':{}}
        client=Client();result=infer(self.row(),client)
        self.assertEqual(result['target_status'],'supported');self.assertTrue(result['sentinel_preserved'])
        self.assertEqual(len(client.calls),2)
        self.assertNotIn('expected_atom',str(client.calls))

    def test_invalid_parser_is_safe_unresolved_not_false_support(self):
        class Bad:
            def complete(self,slot,request):return {'report':{},'usage':{}}
        result=infer(self.row(),Bad())
        self.assertEqual(result['target_status'],'unresolved')
        self.assertTrue(result['verification']['verification_failed'])
        self.assertTrue(result['sentinel_preserved'])


if __name__=='__main__':unittest.main()
