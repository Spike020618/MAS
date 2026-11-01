"""Label-free NumPy interface for INSPIRED native predictor matrices."""
import json
from pathlib import Path
import numpy as np
from experiments.consensus_recommendation.common import ROOT

DEFAULT=ROOT/'experiments/results/native_inspired_strong_v1'

class InspiredPredictorScores:
    def __init__(self,root=DEFAULT,sparse_version='fixed_redial_transfer',kbrd_version='train_inner_tuned'):
        self.root=Path(root);self.ids=json.loads((self.root/'valid.ids.json').read_text());self.items=json.loads((self.root/'native_item_ids.json').read_text())
        self.ri={c:i for i,c in enumerate(self.ids)};self.ci={str(m):j for j,m in enumerate(self.items)}
        self.sparse_version=sparse_version;self.kbrd_version=kbrd_version
        s=self.root/'sparse'/sparse_version;k=self.root/'kbrd'/kbrd_version
        self.scores=np.load(s/'valid.native_scores.npy',mmap_mode='r');self.srank=np.load(s/'valid.native_ranks.npy',mmap_mode='r')
        self.logits=np.load(k/'valid.native_logits.npy',mmap_mode='r');self.krank=np.load(k/'valid.native_ranks.npy',mmap_mode='r');self.logz=np.load(k/'valid.full_entity_logsumexp.npy',mmap_mode='r')
        self.components={name:np.load(self.root/'sparse'/('valid.'+name+'.npy'),mmap_mode='r') for name in ('neighbors50','entity_a1.0_b0.25','recent_mentions')}
        self.kmeta=json.loads((k/'summary.json').read_text());self.catalog=json.loads((self.root/'native_catalog.json').read_text())
        self.names={r['movie_id']:r['native_entity_name'] for r in self.catalog}
        if len(self.items)!=1815 or any(m.shape!=(len(self.ids),len(self.items)) for m in (self.scores,self.srank,self.logits,self.krank)):
            raise ValueError('INSPIRED predictor native shape/ID mismatch')
    def score_pool(self,case_id,candidate_ids):
        i=self.ri[case_id];native=np.asarray(self.logits[i]);mean=float(native.mean());std=max(float(native.std()),1e-8);rows=[]
        for m in candidate_ids:
            j=self.ci[str(m)];rows.append([str(m),float(self.scores[i,j]),int(self.srank[i,j]),float(self.logits[i,j]),
                float((self.logits[i,j]-mean)/std),int(self.krank[i,j]),float(self.logits[i,j]-self.logz[i]),
                *[float(self.components[name][i,j]) for name in ('neighbors50','entity_a1.0_b0.25','recent_mentions')]])
        return {'case_id':case_id,'conversation_id':case_id.rsplit(':',1)[0],
            'source_id':'inspired:predictors:'+self.sparse_version+':'+self.kbrd_version+':'+case_id,
            'columns':['movie_id','sparse_score','sparse_native_rank','kbrd_logit','kbrd_native_z','kbrd_native_rank','kbrd_full_entity_log_probability',
                       'neighbor_score','association_score','recent_mention_score'],'rows':rows,
            'fitted_split':'INSPIRED train only','sparse_version':self.sparse_version,'kbrd_version':self.kbrd_version,
            'kbrd_checkpoint_epoch':self.kmeta['epoch'],'kbrd_checkpoint_sha256':self.kmeta['checkpoint_sha256'],
            'current_user_fact_authority':False,'semantic_truth_certified':False,'native_link_noise_retained':True}
    def candidate_pool(self,case_id,topk=100):
        i=self.ri[case_id];indices=np.argsort(self.srank[i],kind='stable')[:topk]
        return [{'movie_id':str(self.items[j]),'title':self.names[str(self.items[j])],'score':float(self.scores[i,j])} for j in indices]
