import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.paired_deliberation.calibration_d2 import build
from experiments.paired_deliberation.preflight_d2 import prepare,verify,require_engineering_release,VERIFIER
from experiments.paired_deliberation.validate_d2 import infer,wilson
from experiments.paired_deliberation.transport_d2 import TransportD2,parse_d2_raw


class ReleaseTests(unittest.TestCase):
    def test_input_labels_separated_and_control_counts_match(self):
        inputs,labels=build();self.assertEqual(len(inputs),96);self.assertEqual(len({r['id'] for r in inputs}),96)
        self.assertEqual(sum(r['purpose']=='utterance' and r['group']=='validation' for r in labels),32)
        self.assertEqual(sum(r['purpose']=='current_preference' and r['group']=='validation' for r in labels),32)
        self.assertEqual(sum(r['group']=='regression' for r in labels),4)
        self.assertTrue(all(set(r)=={'id','evidence','training_view','claim'} for r in inputs))
        byid={r['id']:r for r in inputs}
        class Never:
            def complete(self,*args):raise AssertionError('authority/catalog/basis controls must not call a model')
        for label in labels:
            if label['purpose'] in {'authority','catalog_fact','prediction_basis'}:
                result=infer(byid[label['id']],Never(),'offline-control')
                self.assertEqual(result['authority_rejected'],label['expected_rejected'])
                if not result['authority_rejected']:self.assertEqual(result['verification']['verdict'],label['expected_verdict'])

    def test_all_96_harness_paths_run_before_any_network_including_five_candidate_regressions(self):
        inputs,_=build()
        class Abstain:
            def complete(self,slot,request):
                return {'report':{'scope':'unsupported','subject':None,'predicate':None,'object':None,
                    'polarity':None,'time':None,'reason':'Offline harness abstention fixture.'},'usage':{}}
        for row in inputs:
            result=infer(row,Abstain(),'offline-harness')
            self.assertTrue(result['sentinel_preserved'])

    def test_source_input_and_labels_cannot_change_after_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            plan=prepare(directory);self.assertEqual(verify(directory)['plan_hash'],plan['plan_hash'])
            path=Path(directory,'inputs.json');data=json.loads(path.read_text());data[0]['claim']['text']='changed';path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'inputs/labels changed'):verify(directory)

    def test_incomplete_or_failed_release_never_authorizes_engineering(self):
        with tempfile.TemporaryDirectory() as directory:
            plan=prepare(directory)
            with self.assertRaisesRegex(RuntimeError,'calibration incomplete'):require_engineering_release(directory)
            Path(directory,'release_score.json').write_text(json.dumps({'version':'d2_purpose_release_v1','plan_hash':plan['plan_hash'],'pass':False,'gates':{'soft':False}}))
            with self.assertRaisesRegex(RuntimeError,'purpose gate failed'):require_engineering_release(directory)

    def test_physical_cap_and_disabled_thinking_are_frozen(self):
        request={'labels_available':False,'kind':'atomic_interpretation','messages':[{'role':'user','content':'Return JSON.'}],'max_tokens':32}
        class Response:status_code=200;text='{}'
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{'DEEPSEEK_API_KEY':'fixture-secret'}):
            transport=TransportD2('deepseek',VERIFIER,directory,1,1.0)
            transport.preflight(request)
            with patch('requests.post',return_value=Response()) as post:transport(request)
            self.assertEqual(post.call_args.kwargs['json']['thinking'],{'type':'disabled'})
            self.assertFalse(post.call_args.kwargs['allow_redirects'])
            Path(directory,'held.intent.json').write_text(json.dumps({'request':request}))
            with self.assertRaisesRegex(RuntimeError,'cap reached'):transport.preflight(request)
        with self.assertRaisesRegex(RuntimeError,'batch halted'):parse_d2_raw({'status_code':401},request)
        lo,hi=wilson(32,32);self.assertLess(lo,1);self.assertEqual(hi,1)


if __name__=='__main__':unittest.main()

class JSONWireTests(unittest.TestCase):
    def test_json_declaration_is_idempotent_and_reservation_precedes_wire(self):
        from experiments.paired_deliberation.json_contract_d2 import json_request
        from experiments.paired_deliberation.budget_wire_d2 import ChannelClient
        from experiments.paired_deliberation.budget_d2 import JointBudget
        from experiments.paired_deliberation.budget_v6 import AttemptClient
        from experiments.tests.test_d2_budget_coordination import LIMITS,PROFILES
        request={'kind':'utterance_witness_d2','labels_available':False,'messages':[{'role':'system','content':'Return exactly witnesses.'},{'role':'user','content':'{}'}],'max_tokens':32}
        fixed=json_request(request)
        self.assertEqual(fixed,json_request(fixed));self.assertNotEqual(fixed,request)
        class P:
            def complete(self,slot,actual):
                self.request=actual
                return {'report':{},'model':'offline-judge','usage':{'prompt_tokens':1,'completion_tokens':1},'elapsed_ms':1}
        p=P();ledger=JointBudget(LIMITS,PROFILES)
        ChannelClient(ledger,'verifier',p).complete('witness',request)
        self.assertEqual(p.request,fixed)
        self.assertEqual(ledger.records[0]['prompt_upper'],AttemptClient.upper(fixed))
