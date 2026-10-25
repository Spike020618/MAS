import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.mechanism import study_explicit_thinking_control as study
from experiments.mechanism import confirm_thinking_control as confirm


class ThinkingControlTests(unittest.TestCase):
    def test_modes_same_model_information_and_cap(self):
        c={'question':'Q','choices':['yes','no'],'gold':'never-forward'}
        a,b=[study.request(c,m) for m in study.MODES]
        self.assertEqual(a['model'],b['model'])
        self.assertEqual(a['messages'],b['messages'])
        self.assertEqual(a['max_tokens'],4096)
        self.assertNotIn('never-forward',str(a))
        self.assertNotIn('temperature',a)
        self.assertNotIn('temperature',b)
        self.assertEqual(b['thinking'],{'type':'enabled'})
        self.assertEqual(b['reasoning_effort'],'high')

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):study.request({'question':'Q','choices':['yes','no']},'invented')

    def test_full_collection_resume_and_confirmation(self):
        from experiments.core.llm import LLM_BACKENDS
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'data';out=root/'original';con=root/'confirmation'
            data.mkdir()
            cases=[{'id':'fixture:'+str(i),'question':'Q'+str(i),'choices':['yes','no'],
                    'split':'calibration' if i<18 else 'evaluation'} for i in range(36)]
            (data/'public_cases.json').write_text(json.dumps(cases))
            (data/'private_labels.json').write_text(json.dumps([{'id':c['id'],'label':'yes'} for c in cases]))
            seen=[]
            def transport(req):
                seen.append(req)
                enabled=req['thinking']['type']=='enabled'
                body={'model':'deepseek-flash','choices':[{'finish_reason':'stop','message':{
                    'content':json.dumps({'answer':'yes' if enabled else 'no','reasoning':'fixture'}),
                    'reasoning_content':'fixture' if enabled else ''}}],
                    'usage':{'prompt_tokens':10,'completion_tokens':20,'total_tokens':30}}
                return {'http_status':200,'body':json.dumps(body)}
            cfg={'api_key':'fixture-not-secret','base_url':'https://invalid.example','model':'deepseek-chat'}
            with patch.object(study.parent,'OUT',data),patch.object(study,'OUT',out), \
                 patch.object(study,'transport',side_effect=transport),patch.dict(LLM_BACKENDS,{'deepseek':cfg}),redirect_stdout(io.StringIO()):
                study.collect();study.collect()
                self.assertEqual(len(seen),72)
                with patch.object(confirm,'DATA',data),patch.object(confirm,'OUT',con):
                    confirm.run();confirm.run()
                self.assertEqual(len(seen),144)
            r=json.loads((out/'result.json').read_text())
            self.assertEqual(r['results']['all']['enabled_vs_disabled'],{'corrections':36,'harms':0})
            c=json.loads((con/'result.json').read_text())
            self.assertEqual(c['corrections'],36)
            self.assertEqual(c['modes']['enabled']['reasoning_content_present'],36)


if __name__=='__main__':unittest.main()
