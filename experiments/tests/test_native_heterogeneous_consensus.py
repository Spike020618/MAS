import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.paired_deliberation.native_heterogeneous_consensus import case,preflight_rows,payload,target_features,METHODS,_source


def fixture():
    ids=[str(i) for i in range(40)]
    row={'id':'dev:1','text':'Seeker: I want a comedy.','candidates':[{'movie_id':m,'title':'Title '+m,'score':1/(61+int(m))} for m in ids],
        'expert_rankings':{'sparse':ids,'kbrd':ids[5:]+ids[:5],'rrf2':ids},'training_neighbors':[],
        'candidate_evidence':{m:{'neighbor_score':(40-int(m))/40,'association_score':0.1,'training_count':5,'kg_neighbors':[]} for m in ids},
        'predictor_scores':{m:{'sparse_score':1/(1+int(m)),'sparse_native_rank':1+int(m),'kbrd_logit':(40-int(m))/10,
             'kbrd_native_z':(20-int(m))/10,'kbrd_native_rank':1+(int(m)+35)%40} for m in ids},'sources':{}}
    for m in ids:
        row['sources']['predictor:kbrd:'+m]=_source('kbrd',row['id'],m,['z','rank'],[row['predictor_scores'][m]['kbrd_native_z'],row['predictor_scores'][m]['kbrd_native_rank']],{'fit_split':'train'})
    return row


class FakeRuntime:
    def __init__(self,row,unoffered=False):self.row=row;self.rank_payloads=[];self.unoffered=unoffered
    def rank(self,slot,data,focus,previous=None,fallback=None):
        assert data['predictor_columns'][1:] == ['sparse_score','sparse_native_rank','kbrd_logit','kbrd_native_z','kbrd_native_rank']
        self.rank_payloads.append(data)
        return {'slot':slot,'valid':True,'ranking':list((previous or fallback)[:10]),'usage':{'prompt_tokens':1,'completion_tokens':1}}
    def call(self,slot,req):
        p=json.loads(req['messages'][1]['content']);a,b=p['pair']
        assert len(p['candidate_pool'])==2 and all(m['movie_id'] in (a,b) for m in p['candidate_pool'])
        source='predictor:kbrd:'+b
        if self.unoffered:source=next(s for s in self.row['sources'] if s not in p['sources'])
        return {'report':{'preferred':a if self.unoffered else b,'source_id':source},'usage':{'prompt_tokens':1,'completion_tokens':1}}


class HeterogeneousRunnerTests(unittest.TestCase):
    def test_four_call_caps_and_scores_visible_from_first_lm_call(self):
        row=fixture();api=FakeRuntime(row)
        with tempfile.TemporaryDirectory() as d:
            r=case(0,row,api,Path(d))
        for method in METHODS:self.assertLessEqual(len(r['methods'][method]['calls']),4)
        self.assertEqual(len(r['methods']['hetero_base']['calls']),1)
        self.assertEqual(len(r['methods']['single4']['calls']),4)
        self.assertEqual(len(r['methods']['independent4']['calls']),4)
        self.assertEqual(len(api.rank_payloads),7)
        for p in api.rank_payloads:self.assertEqual(len(p['predictor_rows']),40)
        for mode in ('central_challenge','game_challenge'):
            self.assertTrue(all(e['valid'] for e in r['methods'][mode]['events']))
    def test_source_existing_globally_but_not_offered_is_rejected(self):
        row=fixture();api=FakeRuntime(row,unoffered=True)
        with tempfile.TemporaryDirectory() as d:r=case(0,row,api,Path(d))
        baseline=r['methods']['hetero_base']['ranking']
        for mode in ('central_challenge','game_challenge'):
            self.assertEqual(r['methods'][mode]['ranking'],baseline)
            self.assertTrue(r['methods'][mode]['events'])
            self.assertTrue(all(not e['valid'] for e in r['methods'][mode]['events']))
    def test_preflight_never_constructs_network_runtime_and_proxy_finite(self):
        row=fixture()
        with patch('experiments.paired_deliberation.native_heterogeneous_consensus.Runtime',side_effect=AssertionError('network runtime forbidden')):
            audit=preflight_rows([row])
        self.assertTrue(audit['pass']);self.assertTrue(audit['no_api_calls'])
        self.assertTrue(all(0<=v['kg_score']<=1 for v in target_features(row).values()))
if __name__=='__main__':unittest.main()
