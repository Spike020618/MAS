import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.mechanism import run_learned_production as r


class RunnerTests(unittest.TestCase):
    def test_selected_upgrade_is_committed_and_executed_before_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary)
            policy={'contexts':{'global':{'game':{'profile':[0,1]},'centralized_erm':[1,0]}}}
            (out/'frozen_policy.json').write_text(json.dumps(policy))
            identity={'model_expected':'deepseek-flash',
                      'fingerprint_expected':'e72d3c7f663d3e1383b00b3884cb4501'}
            seen=[]
            def transport(req):
                seen.append(req['thinking']['type'])
                if req['thinking']['type']=='enabled':
                    self.assertEqual(len(list((out/'test').rglob('decision.json'))),1)
                return {'http_status':200,'body':json.dumps({'model':identity['model_expected'],
                    'system_fingerprint':identity['fingerprint_expected'],
                    'choices':[{'finish_reason':'stop','message':{'content':'{"answer":"yes","reasoning":"fixture"}'}}],
                    'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}})}
            case={'id':'test:fixture','question':'fixture','choices':['yes','no']}
            with patch.object(r,'OUT',out),patch.object(r.api,'transport',side_effect=transport),patch('builtins.print'):
                r.collect_case(case,'test',policy,identity)
            from experiments.mechanism.run_upgrade_pilot import read_rows
            journal=next((out/'test').rglob('attempts.jsonl'))
            self.assertEqual([row['key'] for row in read_rows(journal)],['fast0','fast1','fast2','think2','think1'])

    def test_train_freeze_test_label_guard_and_zero_call_resume(self):
        from experiments.core.llm import LLM_BACKENDS
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary)
            for split in ('train','test'):
                cases=[{'id':split+':'+str(i),'question':'fixture '+str(i),'choices':['yes','no']} for i in range(12)]
                (out/(split+'_public.json')).write_text(json.dumps(cases))
                (out/(split+'_labels.json')).write_text(json.dumps({c['id']:'yes' for c in cases}))
            calls=[]
            def transport(req):
                calls.append(req)
                return {'http_status':200,'body':json.dumps({'model':'deepseek-flash',
                    'system_fingerprint':'e72d3c7f663d3e1383b00b3884cb4501',
                    'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'answer':'yes','reasoning':'fixture'})}}],
                    'usage':{'prompt_tokens':10,'completion_tokens':10,'total_tokens':20}})}
            original_read=Path.read_text
            def guarded_read(path,*args,**kwargs):
                if path==out/'test_labels.json' and not (out/'test_sealed_readouts.json').exists():
                    raise AssertionError('test labels opened before sealed readouts')
                return original_read(path,*args,**kwargs)
            cfg={**LLM_BACKENDS['deepseek'],'api_key':'fixture-not-a-real-key'}
            with (patch.object(r,'OUT',out),patch.dict(LLM_BACKENDS,{'deepseek':cfg}),
                  patch.object(r.api,'transport',side_effect=transport),patch('builtins.print')):
                r.execute('train')
                policy=(out/'frozen_policy.json').read_bytes()
                with patch.object(Path,'read_text',guarded_read):r.execute('test')
                self.assertEqual(policy,(out/'frozen_policy.json').read_bytes())
                self.assertEqual(len(calls),120)
                with patch.object(r.api,'transport',side_effect=AssertionError('replay must not call')):
                    r.execute('train');r.execute('test')
                self.assertEqual(len(calls),120)
            result=json.loads((out/'result.json').read_text())
            self.assertEqual(result['physical_calls'],120)
            self.assertEqual(result['metrics']['game']['correct'],12)
            self.assertEqual(result['metrics']['game']['upgrades'],0)
            self.assertEqual(result['metrics']['centralized_erm']['correct'],12)
            self.assertEqual(result['metrics']['fixed11']['calls'],60)
            self.assertEqual(result['metrics']['fixed00']['calls'],36)


if __name__=='__main__':unittest.main()
