from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.mechanism import study_two_backend_complementarity as study
from experiments.mechanism.complementarity_readout_gate import vote4


class TwoBackendTests(unittest.TestCase):
    def fixture(self):
        c={'id':'fixture:0','question':'Q','choices':['yes','no'],'split':'evaluation'}
        s={b+'/'+str(i):{'answer':'yes' if b=='zhipu_flash' else 'no','valid':True,'tokens':10,'wall_seconds':1.}
           for b in study.BACKENDS for i in range(3)}
        return c,s

    def test_coverage_is_not_accuracy(self):
        c,s=self.fixture();r=study.summarize([c],{c['id']:s},{c['id']:'yes'})
        self.assertEqual(r['metrics']['DDG']['correct'],0)
        self.assertEqual(r['metrics']['DDG']['any_correct'],1)
        self.assertEqual(r['metrics']['DDG']['readout_loss'],1)
        self.assertEqual(r['metrics']['DGG']['correct'],1)
        self.assertEqual(r['diagnostics']['GLM_supplies_missing_DS_answer'],1)
        self.assertEqual(r['metrics']['DDD']['calls'],3)
        self.assertEqual(r['metrics']['DDG']['calls'],3)

    def test_missing_cost_not_filled_zero(self):
        c,s=self.fixture();s['deepseek/0']['tokens']=None
        r=study.summarize([c],{c['id']:s},{c['id']:'yes'})
        self.assertIsNone(r['metrics']['DDG']['tokens'])
        self.assertEqual(r['metrics']['GGG']['tokens'],30)

    def test_vote4_ties_abstain(self):
        self.assertEqual(vote4(['yes','no','yes','no']),'abstain')
        self.assertEqual(vote4(['yes','yes','no','abstain']),'yes')

    def test_invalid_output_cannot_supply_correct_information(self):
        c,s=self.fixture()
        for slot in s.values():slot.update(answer='abstain',valid=False)
        r=study.summarize([c],{c['id']:s},{c['id']:'yes'})
        self.assertEqual(r['metrics']['DDG']['invalid'],3)
        self.assertEqual(r['metrics']['DDG']['correct'],0)
        self.assertEqual(r['diagnostics']['any_of6_correct'],0)

    def test_full_execution_replay_no_second_api_calls(self):
        from experiments.core.llm import LLM_BACKENDS
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            cases=[{'id':'fixture:'+str(i),'question':'Q'+str(i),'choices':['yes','no'],
                    'split':'calibration' if i<18 else 'evaluation'} for i in range(36)]
            (root/'public_cases.json').write_text(json.dumps(cases))
            (root/'private_labels.json').write_text(json.dumps([{'id':c['id'],'label':'yes'} for c in cases]))
            seen=[]
            def transport(req):
                self.assertNotIn('gold',str(req));self.assertNotIn('label',str(req))
                self.assertNotIn('ali_qwen',str(req))
                seen.append(req)
                content='{"ok":true}' if '"ok"' in req['messages'][-1]['content'] else '{"answer":"yes","reasoning":"fixture"}'
                return {'http_status':200,'body':json.dumps({'model':'fixture-model',
                    'choices':[{'finish_reason':'stop','message':{'content':content}}],
                    'usage':{'prompt_tokens':10,'completion_tokens':20,'total_tokens':30}})}
            cfg={b:{'api_key':'fixture-not-secret','base_url':'https://invalid.example','model':'fixture-model'} for b in study.BACKENDS}
            with patch.object(study,'OUT',root),patch.object(study,'prepare'), \
                 patch.object(study.design.p,'transport',side_effect=transport),patch.dict(LLM_BACKENDS,cfg),redirect_stdout(io.StringIO()):
                study.collect();self.assertEqual(len(seen),218)
                study.collect();self.assertEqual(len(seen),218)
            result=json.loads((root/'result.json').read_text())
            self.assertEqual(result['results']['evaluation']['metrics']['DDG']['correct'],18)
            self.assertEqual(result['calibration_selection']['homogeneous'],'DDD')


if __name__=='__main__':unittest.main()
