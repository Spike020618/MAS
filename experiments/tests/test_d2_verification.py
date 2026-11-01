import copy
import json
import unittest

from experiments.consensus_recommendation.common import fingerprint
from experiments.paired_deliberation.state_d2 import StateD2
from experiments.paired_deliberation.verification_d2 import VerificationD2,reduce_utterance
from experiments.paired_deliberation.output_adapter_d2 import D2Client,adapt
from experiments.paired_deliberation.requests_v5 import RequestsV5
from experiments.tests.test_d2_contract_state import registry,claim,report
from experiments.tests.test_design_v6 import evidence


class ScriptClient:
    def __init__(self,atom,judgement):self.atom,self.judgement,self.calls=atom,judgement,[]
    def complete(self,slot,request):
        self.calls.append(copy.deepcopy(request))
        obj=self.atom if request['kind']=='atomic_interpretation' else self.judgement
        return {'report':copy.deepcopy(obj),'usage':{'prompt_tokens':10,'completion_tokens':5}}


def atom(c):return {**c['scope'],'reason':'Offline fixture original meaning.'}


class PurposeTests(unittest.TestCase):
    def test_expression_existence_does_not_require_exhaustive_opposition(self):
        e=evidence();e['observations'][0]['text']='I want Drama. I do not want Drama.'
        r=registry(e);c=claim(r,kind='utterance',text='The seeker said they want Drama.')
        sid=r.resolve('message:1');client=ScriptClient(atom(c),{'witnesses':[{'source_id':sid,'quote':'I want Drama.'}],'reason':'Exact expression.'})
        router=VerificationD2(r,client,service_id='offline-fixture');s=StateD2({'current_request':report(r,[c])},r)
        event=s.observe('central','current_request:c1',fingerprint(s.claims['current_request:c1']),router)
        self.assertEqual(event['verification']['verdict'],'supported')
        self.assertFalse(s.payload()['hard_exclusions'])
        self.assertEqual(client.calls[-1]['kind'],'utterance_witness_d2')
        for req in client.calls:
            p=json.loads(req['messages'][1]['content'])
            self.assertNotIn('training_view',p)
            self.assertFalse(any('train_case:' in key for key in p.get('sources',{})))
        parser=json.loads(client.calls[0]['messages'][1]['content'])
        self.assertEqual(set(parser),{'original_claim','schema'})
        before=len(client.calls)
        s.observe('central','current_request:c1',fingerprint(s.claims['current_request:c1']),router)
        self.assertEqual(len(client.calls),before)

    def test_current_state_still_requires_conflict_handling(self):
        e=evidence();e['observations'][0]['text']='I want Drama. I do not want Drama.'
        r=registry(e);c=claim(r,kind='current_preference',text='The seeker wants Drama.')
        judge={'relations':[{'source_id':r.resolve('message:1'),'relation':'both',
                            'support_quote':'I want Drama.','refute_quote':'I do not want Drama.'}],'reason':'Same-time opposing content.'}
        check=VerificationD2(r,ScriptClient(atom(c),judge),service_id='offline-fixture').verify(c)
        self.assertEqual(check['verdict'],'insufficient');self.assertTrue(check['semantic_record']['evidence_conflict'])

    def test_absent_or_other_speaker_witness_never_refutes_existence(self):
        e=evidence();e['observations'][0]['role']='recommender';r=registry(e)
        c=claim(r,kind='utterance',text='The seeker said they want Drama.')
        client=ScriptClient(atom(c),{'witnesses':[],'reason':'No established expression.'})
        self.assertEqual(VerificationD2(r,client,service_id='fixture').verify(c)['verdict'],'insufficient')
        client=ScriptClient(atom(c),{'witnesses':[{'source_id':r.resolve('message:1'),'quote':e['observations'][0]['text']}], 'reason':'Incorrect speaker.'})
        check=VerificationD2(r,client,service_id='fixture').verify(c)
        self.assertTrue(check['verification_failed']);self.assertEqual(check['verdict'],'insufficient')

    def test_only_narrow_explicit_command_can_activate_hard_exclusion(self):
        for text,hard in [('Do not recommend Drama.',True),('I do not want Drama.',False)]:
            e=evidence();e['observations'][0]['text']=text;r=registry(e)
            c=claim(r,kind='current_preference',text='The seeker does not want Drama.')
            c['scope']['polarity']=False;c['constraint_mode']='hard'
            judge={'relations':[{'source_id':r.resolve('message:1'),'relation':'supports','support_quote':text,'refute_quote':''}], 'reason':'Negative current request.'}
            client=ScriptClient(atom(c),judge);router=VerificationD2(r,client,service_id='fixture')
            s=StateD2({'current_request':report(r,[c])},r)
            s.observe('central','current_request:c1',fingerprint(s.claims['current_request:c1']),router)
            self.assertEqual(bool(s.payload()['hard_exclusions']),hard)
            if hard:self.assertEqual(len(s.payload()['hard_exclusions'][0]['candidate_ids']),6)

    def test_model_cannot_make_recommender_or_unresolved_multi_turn_context_a_hard_fact(self):
        e=evidence();e['observations'][0].update(role='recommender',text='I want Drama.')
        r=registry(e);c=claim(r,kind='current_preference',text='The seeker wants Drama.')
        judge={'relations':[{'source_id':r.resolve('message:1'),'relation':'supports',
                            'support_quote':'I want Drama.','refute_quote':''}],'reason':'Wrong speaker.'}
        check=VerificationD2(r,ScriptClient(atom(c),judge),service_id='fixture').verify(c)
        self.assertTrue(check['verification_failed']);self.assertEqual(check['verdict'],'insufficient')
        e=evidence();e['observations'][0]['text']='Do not recommend Drama.'
        e['observations'].append({'source_id':'message:2','role':'seeker','text':'Actually, recommend Drama.'})
        r=registry(e);c=claim(r,kind='current_preference',text='The seeker does not want Drama.')
        c['scope']['polarity']=False;c['constraint_mode']='hard'
        judge={'relations':[{'source_id':r.resolve('message:1'),'relation':'supports','support_quote':'Do not recommend Drama.','refute_quote':''},
                            {'source_id':r.resolve('message:2'),'relation':'irrelevant','support_quote':'','refute_quote':''}],
               'reason':'An untrusted model misses the later opposing request.'}
        check=VerificationD2(r,ScriptClient(atom(c),judge),service_id='fixture').verify(c)
        self.assertIsNone(check['hard_exclusion_witness'])

    def test_model_scope_disagreement_and_bad_schema_are_separate(self):
        r=registry();c=claim(r,kind='utterance',text='The seeker said they want Drama.')
        wrong=atom(c);wrong['object']='Comedy'
        check=VerificationD2(r,ScriptClient(wrong,{}),service_id='fixture').verify(c)
        self.assertEqual(check['verdict'],'insufficient');self.assertTrue(check['scope_disagreement'])
        self.assertFalse(check['verification_failed'])
        check=VerificationD2(r,ScriptClient({},{}),service_id='fixture').verify(c)
        self.assertTrue(check['verification_failed'])

    def test_wrong_object_quote_remains_a_semantic_proxy_risk(self):
        # Quotation and role validation do not establish semantic entailment.
        r=registry();c=claim(r,kind='utterance',text='The seeker said they want Drama.')
        sources=r.dialogue();sid=r.resolve('message:1')
        result=reduce_utterance(atom(c),sources,{'witnesses':[{'source_id':sid,'quote':sources[sid]['text']}],'reason':'Untrusted judgement.'})
        self.assertEqual(result['verdict'],'supported')


class AdaptationTests(unittest.TestCase):
    def test_schema_unwrap_and_exact_template_echo_are_traced(self):
        request=RequestsV5.envelope('test_d2',{'schema':{'value':''}},'Return JSON.')
        for obj in ({'schema':{'value':'original'}},{'value':'original','schema':{'value':''}}):
            original={'report':obj,'usage':{}}
            result,event=adapt(original,request)
            self.assertEqual(result['report'],{'value':'original'})
            self.assertTrue(event['operations']);self.assertEqual(original['report'],obj)
    def test_report_rank_ids_are_lossless_but_out_of_pool_is_not_fixed(self):
        e=evidence();request=RequestsV5.envelope('initial_d2',{'evidence':e,'schema':{'ranking':[],'claims':[],'updates':[]}},'Return JSON.')
        record={'report':{'answer':{'ranking':[1,2,3,4,999],'claims':[],'updates':[]}}}
        result,event=adapt(record,request)
        self.assertEqual(result['report']['ranking'],['1','2','3','4',999])
        self.assertFalse(result['report']['claims'])


if __name__=='__main__':unittest.main()
