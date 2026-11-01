"""CRSLab INSPIRED native train/valid baselines, test member never opened.

Original IDs and all author annotations retained, including noisy non-film
entities. Each Recommender response movie is one target; no seen-item removal.
"""
import ast,collections,hashlib,json,zipfile
from copy import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from experiments.consensus_recommendation.common import ROOT,create,sha
from experiments.paired_deliberation.benchmark_evaluation import native_item_metrics

DATA=ROOT/'experiments/data/raw/crslab_inspired_native'
OUT=ROOT/'experiments/results/native_inspired_benchmark_v1'
EXPECTED='776cadc7585abdbca2738addae40488826c82de3cfd4c2dc13dcdd63aefdc5c4'
SOURCE=ROOT/'docs/research/inspired_protocol_sources_2026-10-28/CRSLab/crslab/data/dataset/inspired/inspired.py'


def contexts(raw,entities,split):
    result=[]
    for index,conv in enumerate(raw):
        merged=[]
        for u in conv['dialog']:
            movies=[entities[m] for m in u['movies'] if m in entities]
            if merged and merged[-1]['role']==u['role']:
                merged[-1]['text']+=u['text'];merged[-1]['movies']+=movies
            else:merged.append({'role':u['role'],'text':list(u['text']),'movies':movies})
        prefix=[];earlier=[]
        for turn,u in enumerate(merged):
            if prefix and u['role']=='Recommender' and u['movies']:
                result.append({'id':'inspired:'+split+':'+str(index).zfill(4)+':'+str(turn),'text':'\n'.join(prefix),
                    'previous_movie_ids':list(earlier),'targets':list(u['movies'])})
            prefix.append(u['role']+': '+' '.join(u['text']));earlier+=u['movies']
    return result


def author_targets(raw,folder):
    tree=ast.parse(SOURCE.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    names={'_convert_to_id','_augment_and_add'};functions=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    scope={'copy':copy};exec(compile(ast.Module(body=functions,type_ignores=[]),str(SOURCE),'exec'),scope)
    obj=SimpleNamespace(entity2id=json.loads((folder/'entity2id.json').read_text()),tok2ind=json.loads((folder/'token2id.json').read_text()),
        word2id=json.loads((folder/'word2id.json').read_text()),unk_token_idx=3)
    return [u['items'] for conv in raw for u in scope['_augment_and_add'](obj,scope['_convert_to_id'](obj,conv)) if u['role']=='Recommender' and u['items']]


def evaluate(rows,rankings):
    total=collections.Counter();count=0;covered=0
    for row,ranking in zip(rows,rankings):
        for target in row['targets']:
            count+=1;covered+=target in ranking[:100];total.update(native_item_metrics(ranking,target))
    return {'contexts':len(rows),'item_targets':count,'hits@10':int(total['recall@10']),'metrics':{key:value/count for key,value in total.items()},'candidate_recall@100':covered/count}


def run():
    archive=DATA/'inspired_nltk.zip';assert sha(archive)==EXPECTED
    folder=DATA/'released';entities=json.loads((folder/'entity2id.json').read_text());items=json.loads((folder/'movie_ids.json').read_text());inv={v:k for k,v in entities.items()};index={m:i for i,m in enumerate(items)}
    raw={split:json.loads((folder/(split+'_data.json')).read_text()) for split in ('train','valid')}
    splits={split:contexts(raw[split],entities,split) for split in raw}
    assert all(t in index for rows in splits.values() for row in rows for t in row['targets'])
    parity={split:[r['targets'] for r in rows]==author_targets(raw[split],folder) for split,rows in splits.items()};assert all(parity.values())
    train=splits['train'];valid=splits['valid'];counts=collections.Counter(t for r in train for t in r['targets']);pop=sorted(items,key=lambda m:(-counts[m],m));tie={m:i for i,m in enumerate(pop)}
    vec=TfidfVectorizer(lowercase=True,ngram_range=(1,2),min_df=2,max_features=60000,sublinear_tf=True,dtype=np.float32)
    x=vec.fit_transform([r['text'] for r in train]);q=vec.transform([r['text'] for r in valid]);rr=[];cc=[];vv=[]
    for i,row in enumerate(train):
        for target,n in collections.Counter(row['targets']).items():rr.append(i);cc.append(index[target]);vv.append(n/len(row['targets']))
    y=csr_matrix((np.array(vv,dtype=np.float32),(rr,cc)),shape=(len(train),len(items)));ranks=[];scores=[]
    for start in range(0,len(valid),64):
        sims=(q[start:start+64]@x.T).toarray()
        for sim in sims:
            neighbors=np.argsort(-sim,kind='stable')[:50];values=(sim[neighbors]@y[neighbors]).ravel()
            order=sorted(range(len(items)),key=lambda j:(-float(values[j]),tie[items[j]]))[:100]
            ranks.append([items[j] for j in order]);scores.append([float(values[j]) for j in order])
    # No model input contains the current response or any evaluation target.
    inputs=[{'id':row['id'],'text':row['text'],'previous_movie_ids':[str(m) for m in row['previous_movie_ids']],
        'candidates':[{'movie_id':str(m),'title':inv[m],'score':round(score,7)} for m,score in zip(rank,values)]}
        for row,rank,values in zip(valid,ranks,scores)]
    create(OUT/'valid.inputs.json',inputs);create(OUT/'valid.predictions.json',[{'id':r['id'],'ranking':rank} for r,rank in zip(valid,ranks)])
    create(OUT/'valid.labels.json',[{'id':r['id'],'targets':r['targets']} for r in valid])
    create(OUT/'native_catalog.json',[{'movie_id':str(m),'native_entity_name':inv[m]} for m in items])
    plan={'version':'native-inspired-nltk-v0.3-baseline-v1','source_commit':'6497938919998b9cd7aa9c2f43effb39ad0c7088','archive_sha256':EXPECTED,
        'author_target_parity':parity,'test_member_read':False,'test_labels_available':False,'model_api_calls':0,'native_item_count':len(items),
        'split_unit':'released conversation order; stable split+conversation-index+merged-turn IDs','sample_unit':'one item in each recommender merged response, duplicates retained',
        'role_mapping':{'Seeker':'current user expression','Recommender':'recommendation context, not user preference'},
        'fit_split':'train only','parameters':{'ngram':[1,2],'min_df':2,'max_features':60000,'neighbors':50,'tie_break':'train popularity then native item ID'},
        'candidate_policy':'full native item universe for scoring, retrieved100 for serving, no gold insertion or seen-item exclusion',
        'source_sha256':sha(Path(__file__)),'source_files_sha256':{name:sha(folder/name) for name in ('train_data.json','valid_data.json','entity2id.json','movie_ids.json')},
        'limitations':['author annotations include non-film or wrongly linked names; retained for native protocol, not factual satisfaction truth','no KBRD/UniCRS trained reproduction claimed','validation only, untouched bundled test is not evaluated']}
    create(OUT/'plan.json',plan)
    summary={'native_items':len(items),'train_conversations':len(raw['train']),'valid_conversations':len(raw['valid']),
        'train_contexts':len(train),'train_item_targets':sum(len(r['targets']) for r in train),'valid_contexts':len(valid),'valid_item_targets':sum(len(r['targets'])for r in valid),
        'popularity':evaluate(valid,[pop[:100]]*len(valid)),'tfidf_neighbors':evaluate(valid,ranks),'author_target_parity':parity,'test_member_read':False}
    create(OUT/'summary.json',summary);print(json.dumps(summary,indent=2));return summary
if __name__=='__main__':run()
