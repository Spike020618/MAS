"""Actual existing AEIC workflow behind the same v1 app boundary.
Run locally by MAS_ENGINE_PLUGIN=system.application.legacy_adapter:LegacyAEICAdapter.
No hidden DeepSeek calls: role calls use the selected app provider and receipt budget.
"""
from .engine import CatalogEvidenceEngine,Client,UnknownCall,Cancelled
from system.game_consensus_recsys.workflow import ConsensusWorkflow
from system.game_consensus_recsys.domain import SearchRequest,MovieCandidate,CandidateSource
from system.game_consensus_recsys.agents import PreferenceAgent,ConstraintVerifierAgent
from system.game_consensus_recsys.deepseek_agents import DeepSeekReviewAgent

class LegacyAEICAdapter:
    def capabilities(self):
        return {'id':'legacy-aeic-adapter-v1','contract':'mas.engine.result.v1','methods':{'fixed':'原系统 AEIC 评审与聚合'},'supports':['events','citations','cancel','receipt_resume'],
                'scope':'原系统研究工作流的真实适配；未接入新贡献模型','value_estimator':'早期证据与信誉聚合'}
    def run(self,r,emit,checkpoint,cancelled):
        client=Client(r.provider,emit,checkpoint,cancelled,r.max_calls,r.max_cny)
        class FrozenCatalog:
            def search(self,request,limit=50):
                return [MovieCandidate(x['id'],x['title'],x['year']or 0,x['genres'],[CandidateSource.LEXICAL],min(1,x['retrieval_score']),{'rating':x['rating'],'semantic_tags':x['tags'],'evidence_refs':x['sources']})for x in r.candidates[:limit]]
        agents=[PreferenceAgent(),ConstraintVerifierAgent()]
        if r.provider!='offline'and r.max_calls:
            class Wrapper:
                def __init__(self,role):self.role=role
                def complete_json(self,system_prompt,user_prompt,**kwargs):
                    result=client.call({'role_prompt':system_prompt,'input':user_prompt},'legacy-'+self.role)
                    if result is None:raise ValueError('模型核查未返回有效响应')
                    result['_usage']={'total_tokens':0};return result
            for role in ['preference','diversity','critic'][:r.max_calls]:agents.append(DeepSeekReviewAgent(Wrapper(role),role,'legacy-'+role))
        workflow=ConsensusWorkflow(catalog=FrozenCatalog(),agents=agents)
        if cancelled():raise Cancelled()
        emit('legacy','调用原系统 AEIC 工作流',{'adapter':self.capabilities()['id']})
        old=workflow.run(SearchRequest(query=r.query,top_k=r.top_k,**{k:v for k,v in r.constraints.items()if k in SearchRequest.__dataclass_fields__}))
        if cancelled():raise Cancelled()
        if client.unknown:raise UnknownCall('原工作流包含完成状态未知的请求，不能自动重发')
        raw=old.to_dict();lookup={x['id']:x for x in r.candidates};movies=[]
        for d in raw['decisions']:
            m=dict(lookup[d['candidate']['movie_id']]);m.update(match_score=d['score'],initial_score=m['retrieval_score'],explanation='；'.join(e['claim']for e in d['accepted_evidence']),uncertainties=d['residual_dissent']);movies.append(m)
        for e in raw['trace']['events']:emit(e['stage'],e['event'],e['payload'])
        return {'schema':'mas.engine.result.v1','engine':self.capabilities(),'method':'fixed','provider':r.provider,'movies':movies,
                'initial_ranking':[x['id']for x in r.candidates],'final_ranking':[x['id']for x in movies],'reports':raw['trace']['reports'],'changes':[],
                'allocation':[],'usage':{'calls':client.calls,'tokens':client.tokens,'estimated_cny':client.cost,'reserved_cny':client.reserved},'summary':f'原系统 AEIC 工作流返回{len(movies)}部影片。'}
