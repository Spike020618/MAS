import copy,json,tempfile,unittest
from pathlib import Path
from experiments.evidence_consensus.runner import task_case,common_state,request,linked_patch
from experiments.evidence_consensus.proposal import proposals,allocate


def task():
    return {'id':'dev:qa:1','kind':'qa','prompt':'Is the reference premise supported?','options':['yes','no'],'candidates':[],'initial':'abstain','source_catalog':{str(i):{'source_id':str(i),'kind':'external_evidence','title':'reference premise','text':'Evidence '+str(i),'scope':{'task_id':'dev:qa:1'},'allowed_uses':['prediction_basis']}for i in range(4)}}

class FakeAPI:
    profile={'input_cny_per_million':.15,'output_cny_per_million':1.5}
    def __init__(self,root):self.root=root;self.requests=[]
    def call(self,slot,req):
        self.requests.append(copy.deepcopy(req));p=json.loads(req['messages'][1]['content']);e=copy.deepcopy(p['schema_example'])
        e['A'][0]['text']='Bounded evidence premise.'
        if p['phase']=='initial':e['decision']='yes'
        else:
            e['I']=[{'claim_id':p['claim_id_to_use'],'source_id':next(iter(p['fetched_sources'])),'relation':'support'}]
            e['E']=[next(iter(p['fetched_sources']))]
            e['C']['proposed_patch']={'operation':'set_answer','value':'no'if p['current_decision']=='yes'else 'yes','expected_version':p['current_version'],'evidence_ids':[next(iter(p['fetched_sources']))],'preserve_ids':[]}
        return {'report':e,'usage':{'prompt_tokens':100,'completion_tokens':70},'model':'fake-fixed-model'}

class UnifiedRunnerTests(unittest.TestCase):
    def test_real_online_step_observes_prior_accepted_state_and_budget(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);api=FakeAPI(root);r=task_case(0,task(),api,root)
            for method in ('single','fixed','central','game'):
                arm=r['methods'][method];reviews=[e for e in arm['events']if 'source_action'in e]
                self.assertTrue(reviews)
                for first,second in zip(reviews,reviews[1:]):
                    self.assertEqual(second['expected_state_version'],first['expected_state_version']+1)
                    self.assertEqual(second['before_decision'],first['after_decision'])
                self.assertLessEqual(arm['logical_calls'],12)
                self.assertEqual(arm['budget']['used_or_reserved']['calls'],arm['logical_calls'])
                self.assertTrue(all(e['source_action']['new_sources']==1 for e in reviews))
            self.assertEqual(r['methods']['independent']['logical_calls'],3)
            n=len(api.requests);self.assertEqual(task_case(0,task(),api,root),r);self.assertEqual(len(api.requests),n)
    def test_discovery_does_not_use_unfetched_evidence_body(self):
        t=task();s=common_state(t);before=proposals(t,s,[])
        for sid,c in s['catalog'].items():
            if sid not in s['visible_sources']:c['text']='TOTALLY DIFFERENT BODY OF SAME LENGTH'
        after=proposals(t,s,[])
        self.assertEqual([(x['source_id'],x['score'])for x in before],[(x['source_id'],x['score'])for x in after])
    def test_unclear_or_unreported_evidence_cannot_authorize_patch(self):
        report={'E':['s'],'I':[{'source_id':'s','relation':'unclear'}],'C':{'proposed_patch':{'operation':'set_answer','evidence_ids':['s']}}}
        self.assertFalse(linked_patch(report))
        report['I'][0]['relation']='support';self.assertTrue(linked_patch(report))
        report['E']=[];self.assertFalse(linked_patch(report))

    def test_homogeneous_gain_matrix_does_not_invent_agent_ability(self):
        r=allocate(proposals(task(),common_state(task()),[]),'game')
        self.assertLessEqual(len(r['actions']),3)
        self.assertLessEqual(r['game']['max_feasible_unilateral_gain'],1e-10)
        self.assertEqual(len({q for _,q in r['actions']}),len(r['actions']))

if __name__=='__main__':unittest.main()
