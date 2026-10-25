import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from experiments.mechanism import run_reliable_readout as runner


class ReliableRunnerTests(unittest.TestCase):
    def test_complete_fixture_and_idempotent_resume(self):
        from experiments.core.llm import LLM_BACKENDS
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'out';data=root/'data';out.mkdir();data.mkdir()
            cases=[{'id':'fixture:'+str(i),'question':'Q'+str(i),'choices':['yes','no'],'split':'evaluation'} for i in range(18)]
            (out/'public_cases.json').write_text(json.dumps(cases))
            (data/'private_labels.json').write_text(json.dumps([{'id':c['id'],'label':'yes'} for c in cases]))
            prior_cases=json.loads((runner.prior.OUT/'public_cases.json').read_text())
            model={'probabilities':{'deepseek':.7,'zhipu_flash':.5}}
            cal={'model':model,'source_sha256':{},'calibration_ids':[c['id'] for c in prior_cases if c['split']=='calibration'],
                 'code_sha256':runner.design.fingerprint(Path(runner.calibrate.__code__.co_filename))}
            (out/'calibration.json').write_text(json.dumps(cal))
            seen=[]
            def transport(req):
                self.assertNotIn('gold',str(req));self.assertNotIn('ali_qwen',str(req));seen.append(req)
                return {'http_status':200,'body':json.dumps({'model':'glm-4-flash' if req['backend']=='zhipu_flash' else 'deepseek-flash',
                    'choices':[{'finish_reason':'stop','message':{'content':'{"answer":"yes","reasoning":"fixture"}'}}],
                    'usage':{'prompt_tokens':10,'completion_tokens':20,'total_tokens':30}})}
            with patch.object(runner,'OUT',out),patch.object(runner,'DATA',data),patch.object(runner,'transport',side_effect=transport),redirect_stdout(io.StringIO()):
                runner.run();self.assertEqual(len(seen),90)
                runner.run();self.assertEqual(len(seen),90)
            result=json.loads((out/'result.json').read_text())
            self.assertEqual(result['metrics']['weighted_DDG']['correct'],18)
            self.assertEqual(result['metrics']['weighted_DDG']['tokens'],18*3*30)
            self.assertEqual(result['metrics']['single_thinking']['calls'],18)


if __name__=='__main__':unittest.main()
