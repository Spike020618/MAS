"""Train-only native ReDial retrieval components; valid is explicitly development.

Never opens public test or legacy reserved data. Entity features are annotations
from the already visible dialogue prefix, not the target recommender response.
"""
import hashlib,json,re
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from experiments.paired_deliberation.native_redial_benchmark import ROOT,DATA,evaluate,save

RESULT=ROOT/'experiments/results/native_retrieval_optimization_v1'

def enhanced_contexts(raw,entities):
    rows=[]
    for conv in raw:
        turns=[]
        for u in conv['dialog']:
            movies=[entities[m] for m in u['movies'] if m in entities]
            ents=[entities[m] for m in u['entity'] if m in entities]
            if turns and turns[-1]['role']==u['role']:
                turns[-1]['tokens']+=u['text'];turns[-1]['movies']+=movies;turns[-1]['entities']+=ents
            else:turns.append({'role':u['role'],'tokens':list(u['text']),'movies':movies,'entities':ents})
        prefix=[];seen_movies=[];seen_entities=[]
        for i,t in enumerate(turns):
            if i and t['role']=='Recommender' and t['movies']:
                rows.append({'id':str(conv['conv_id'])+':'+str(i),'text':'\n'.join(prefix),'targets':list(t['movies']),
                    'context_movies':list(seen_movies),'context_entities':list(dict.fromkeys(seen_entities)),
                    'last_seeker_text':next((' '.join(x['tokens']) for x in reversed(turns[:i]) if x['role']=='Seeker'),'')})
            prefix.append(t['role']+': '+' '.join(t['tokens']))
            seen_movies.extend(t['movies']);seen_entities.extend(t['entities']+t['movies'])
    return rows


def load_train_valid():
    p=DATA/'released';entities=json.loads((p/'entity2id.json').read_text());items=json.loads((p/'movie_ids.json').read_text())
    train=enhanced_contexts(json.loads((p/'train_data.json').read_text()),entities)
    valid=enhanced_contexts(json.loads((p/'valid_data.json').read_text()),entities)
    return train,valid,entities,items


def maxnorm(a):
    a=np.asarray(a,dtype=np.float32);return np.divide(a,np.maximum(a.max(axis=1,keepdims=True),1e-12))


class NativeRetriever:
    def __init__(self,items,entities):
        self.original_items=list(items);self.entities=entities;self.names={v:k.removeprefix('<http://dbpedia.org/resource/').removesuffix('>').replace('_',' ') for k,v in entities.items()}
    def fit(self,train):
        self.train=train;counts=Counter(t for r in train for t in r['targets'])
        self.items=sorted(self.original_items,key=lambda m:(-counts[m],m));self.index={m:i for i,m in enumerate(self.items)}
        self.counts=np.array([counts[m] for m in self.items],dtype=np.float32)
        rr=[];cc=[];vv=[]
        for i,row in enumerate(train):
            for t,c in Counter(row['targets']).items():rr.append(i);cc.append(self.index[t]);vv.append(c/len(row['targets']))
        self.y=csr_matrix((np.array(vv,dtype=np.float32),(rr,cc)),shape=(len(train),len(self.items)))
        self.vectorizer=TfidfVectorizer(lowercase=True,ngram_range=(1,2),min_df=2,max_features=60000,sublinear_tf=True,dtype=np.float32)
        self.x=self.vectorizer.fit_transform([r['text'] for r in train]);self.itemdocs=normalize((self.y.T@self.x).tocsr(),copy=False)
        features=sorted({e for r in train for e in r['context_entities']});self.feature_index={e:i for i,e in enumerate(features)}
        rr=[];cc=[]
        for i,row in enumerate(train):
            for e in set(row['context_entities']):rr.append(i);cc.append(self.feature_index[e])
        self.context_matrix=csr_matrix((np.ones(len(rr),dtype=np.float32),(rr,cc)),shape=(len(train),len(features)))
        self.feature_counts=np.asarray(self.context_matrix.sum(axis=0)).ravel()
        self.assoc=(self.context_matrix.T@self.y).tocsr()
        return self
    def components(self,rows):
        q=self.vectorizer.transform([r['text'] for r in rows]);itemdocs=(q@self.itemdocs.T).toarray()
        neighbor={k:np.empty((len(rows),len(self.items)),dtype=np.float32) for k in (10,50,200)}
        for start in range(0,len(rows),64):
            sim=(q[start:start+64]@self.x.T).toarray()
            for offset,s in enumerate(sim):
                ns=np.argsort(-s,kind='stable')[:200]
                for k in neighbor:neighbor[k][start+offset]=(s[ns[:k]]@self.y[ns[:k]]).ravel()
        rr=[];cc=[];vv=[]
        for i,row in enumerate(rows):
            for e in set(row['context_entities']):
                if e in self.feature_index:rr.append(i);cc.append(self.feature_index[e]);vv.append(1)
        e=csr_matrix((np.array(vv,dtype=np.float32),(rr,cc)),shape=(len(rows),len(self.feature_index)))
        out={'popularity':np.tile(self.counts,(len(rows),1)),'itemdocs':itemdocs,**{'neighbors'+str(k):v for k,v in neighbor.items()}}
        for alpha in (.5,1.):
            a=self.assoc.multiply(1/np.maximum(self.feature_counts,1)[:,None]**alpha).tocsr()
            scores=(e@a).toarray()
            for beta in (0.,.25):out['entity_a'+str(alpha)+'_b'+str(beta)]=scores/np.maximum(self.counts,1)[None,:]**beta
        copy_scores=np.zeros((len(rows),len(self.items)),dtype=np.float32)
        for i,row in enumerate(rows):
            for age,m in enumerate(reversed(row['context_movies'])):
                if m in self.index:copy_scores[i,self.index[m]]+=.8**age
        out['recent_mentions']=copy_scores
        return out
    def rank(self,row,config,topk=100):
        components=self.components([row]);score=fuse(components,config)[0];indices=np.argsort(-score,kind='stable')[:topk]
        return [{'movie_id':str(self.items[j]),'title':self.names[self.items[j]],'score':float(score[j]),
                 'components':{k:float(maxnorm(v)[0,j]) for k,v in components.items()}} for j in indices]
    def evidence(self,row,candidate_ids,config=None,examples=3):
        """Return literal TRAIN annotations/statistics, explicitly OTHER users."""
        native=[int(x) for x in candidate_ids];q=self.vectorizer.transform([row['text']]);s=(q@self.x.T).toarray()[0]
        picked=[]
        for i in np.argsort(-s,kind='stable'):
            old=self.train[i]
            if old['id'].split(':')[0]==row['id'].split(':')[0]:continue
            overlap=[m for m in old['targets'] if m in native]
            if not overlap:continue
            picked.append({'source_id':'train:'+old['id'],'origin_case_id':old['id'],'role':'OTHER historical users',
                'text':old['text'],'observed_next_recommender_movie_ids':list(map(str,overlap)),
                'similarity':float(s[i]),'current_user_fact_authority':False})
            if len(picked)==examples:break
        columns=[]
        for mid in native:
            if mid not in self.index:continue
            j=self.index[mid];top=[]
            for ent in row['context_entities']:
                f=self.feature_index.get(ent)
                if f is None:continue
                c=float(self.assoc[f,j])
                if c>0:top.append({'prefix_entity_id':str(ent),'prefix_entity_name':self.names.get(ent,str(ent)),
                    'weighted_train_next_movie_count':c,'train_context_count':float(self.feature_counts[f])})
            top.sort(key=lambda x:-x['weighted_train_next_movie_count']/max(x['train_context_count'],1))
            columns.append({'movie_id':str(mid),'train_popularity_count':float(self.counts[j]),'prefix_associations':top[:3]})
        return {'historical_examples':picked,'training_associations':columns,'fitted_split':'train','current_user_fact_authority':False}


def fuse(components,config):
    total=np.zeros_like(next(iter(components.values())))
    for name,weight in config.items():total+=weight*maxnorm(components[name])
    return total


def configurations():
    configs={name:{name:1.} for name in ['popularity','neighbors10','neighbors50','neighbors200','itemdocs','entity_a0.5_b0.0','entity_a0.5_b0.25','entity_a1.0_b0.0','entity_a1.0_b0.25','recent_mentions']}
    for entity in ('entity_a0.5_b0.25','entity_a1.0_b0.0','entity_a1.0_b0.25'):
        for w in (.25,.5,1.,2.):
            configs['neighbors50+'+entity+'*'+str(w)]={'neighbors50':1.,entity:w}
    for w in (.25,.5,1.):
        configs['neighbors50+itemdocs*'+str(w)]={'neighbors50':1.,'itemdocs':w}
        configs['neighbors50+entity+recent*'+str(w)]={'neighbors50':1.,'entity_a1.0_b0.25':.5,'recent_mentions':w}
    return configs


def run():
    train,valid,entities,items=load_train_valid();model=NativeRetriever(items,entities).fit(train)
    old=json.loads((ROOT/'experiments/results/native_redial_benchmark_v1/single_llm.inputs.json').read_text());oldids=[r['id'] for r in old]
    selected=sorted([r for r in valid if r['id'] not in set(oldids)],key=lambda r:hashlib.sha256(('native-opt-v1:'+r['id']).encode()).hexdigest())[:64]
    save(RESULT/'selection_before_search.json',{'ids':[r['id'] for r in selected],'rule':'input-ID hash excluding prior32; no label-based selection','scope':'same public validation used for offline search; not independent confirmation'})
    save(RESULT/'protocol.json',{'fit':'train only','tune':'full public valid, explicitly exploratory development','public_test_read':False,'legacy_reserved_read':False,'native_items':len(items),
        'configs':configurations(),'prefix_entity_authority':'only already occurred utterances; no current recommender response entities',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    comp=model.components(valid);all_configs=configurations();results={};best=None;bestkey=None;best_ranks=None;best_scores=None
    lookup={r['id']:i for i,r in enumerate(valid)};old_idx=[lookup[x] for x in oldids]
    for name,config in all_configs.items():
        score=fuse(comp,config);ranks_idx=np.argsort(-score,axis=1,kind='stable')[:,:100]
        ranks=[[model.items[j] for j in rank] for rank in ranks_idx]
        metrics=evaluate(valid,ranks);fixed=evaluate([valid[i] for i in old_idx],[ranks[i] for i in old_idx]);results[name]={'config':config,'full_valid':metrics,'old_fixed32':fixed}
        key=(metrics['metrics']['recall@10'],metrics['retrieval_recall@100'],metrics['metrics']['ndcg@10'])
        if bestkey is None or key>bestkey:bestkey=key;best=name;best_ranks=ranks_idx.copy();best_scores=score.copy()
        print(name,round(key[0],5),round(key[1],5),'fixed32',round(fixed['metrics']['recall@10'],5),flush=True)
    save(RESULT/'search_results.json',results);save(RESULT/'selected_config.json',{'name':best,'config':all_configs[best],
        'selection':'maximum full valid Recall@10, then Recall@100 then NDCG@10; development tuned','result':results[best]})
    predictions=[{'id':row['id'],'ranking':[model.items[j] for j in best_ranks[i]],'scores':[float(best_scores[i,j]) for j in best_ranks[i]]} for i,row in enumerate(valid)]
    save(RESULT/'valid.predictions.json',predictions)
    norm={name:maxnorm(v) for name,v in comp.items()}
    for label,rows in [('old32', [valid[i] for i in old_idx]),('new64',selected)]:
        inputs=[];labels=[];evidence=[]
        for row in rows:
            i=lookup[row['id']];candidates=[{'movie_id':str(model.items[j]),'title':model.names[model.items[j]],'score':float(best_scores[i,j]),
                'components':{name:float(v[i,j]) for name,v in norm.items()}} for j in best_ranks[i]]
            inputs.append({k:v for k,v in row.items() if k!='targets'}|{'candidates':candidates});labels.append({'id':row['id'],'targets':row['targets']})
            evidence.append({'id':row['id'],**model.evidence(row,[m['movie_id'] for m in candidates])})
        save(RESULT/(label+'.inputs.json'),inputs);save(RESULT/(label+'.labels.json'),labels);save(RESULT/(label+'.evidence.json'),evidence)
    print('BEST',best,json.dumps(results[best]),flush=True)

if __name__=='__main__':run()
