"""Select retrieval weights on TRAIN dialogue holdout before scoring public valid."""
import gc,hashlib,json
import numpy as np
from experiments.paired_deliberation.native_retrieval_optimization import (ROOT,RESULT,NativeRetriever,load_train_valid,configurations,fuse,maxnorm,save,evaluate)

INNER=ROOT/'experiments/results/native_retrieval_inner_v1'

def run():
    train,valid,entities,items=load_train_valid()
    inner_train=[r for r in train if int(hashlib.sha256(('inner-retrieval-20261028:'+r['id'].split(':')[0]).encode()).hexdigest(),16)%5!=0]
    inner_dev=[r for r in train if int(hashlib.sha256(('inner-retrieval-20261028:'+r['id'].split(':')[0]).encode()).hexdigest(),16)%5==0]
    assert {r['id'].split(':')[0] for r in inner_train}.isdisjoint({r['id'].split(':')[0] for r in inner_dev})
    configs=configurations()
    save(INNER/'protocol.json',{'selection':'train dialogues hash heldout 1/5; no public valid labels for selection',
         'inner_train_contexts':len(inner_train),'inner_dev_contexts':len(inner_dev),'configs':configs,
         'public_test_opened':False,'legacy_reserved_opened':False,
         'source_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()})
    model=NativeRetriever(items,entities).fit(inner_train);comp=model.components(inner_dev);results={};best=None;bestkey=None
    for name,config in configs.items():
        score=fuse(comp,config);ix=np.argsort(-score,axis=1,kind='stable')[:,:100]
        metrics=evaluate(inner_dev,[[model.items[j] for j in row] for row in ix]);results[name]={'config':config,'inner_dev':metrics}
        key=(metrics['metrics']['recall@10'],metrics['retrieval_recall@100'],metrics['metrics']['ndcg@10'])
        if bestkey is None or key>bestkey:bestkey=key;best=name
        print(name,key,flush=True)
    save(INNER/'inner_search_results.json',results)
    # The config commitment precedes scoring any public validation row here.
    save(INNER/'selected_config.json',{'name':best,'config':configs[best],'selection':'only train conversation holdout',
         'inner_dev_result':results[best]})
    del model,comp,score,ix;gc.collect()
    model=NativeRetriever(items,entities).fit(train);comp=model.components(valid);score=fuse(comp,configs[best]);ix=np.argsort(-score,axis=1,kind='stable')[:,:100]
    ranks=[[model.items[j] for j in row] for row in ix];metrics=evaluate(valid,ranks)
    old_inputs=json.loads((ROOT/'experiments/results/native_redial_benchmark_v1/single_llm.inputs.json').read_text());old_ids=[r['id'] for r in old_inputs];lookup={r['id']:i for i,r in enumerate(valid)}
    old_idx=[lookup[c] for c in old_ids]
    selected_ids=json.loads((RESULT/'selection_before_search.json').read_text())['ids']
    summary={'selected_config':configs[best],'name':best,'fit':'full train after train-only inner selection','full_valid':metrics,
        'old32':evaluate([valid[i] for i in old_idx],[ranks[i] for i in old_idx]),
        'new64':evaluate([valid[lookup[c]] for c in selected_ids],[ranks[lookup[c]] for c in selected_ids]),
        'caveat':'same public valid had prior independent development searches in project; no longer globally untouched confirmation'}
    save(INNER/'summary.json',summary);save(INNER/'valid.predictions.json',[{'id':r['id'],'ranking':ranks[i],
        'scores':[float(score[i,j]) for j in ix[i]]} for i,r in enumerate(valid)])
    norm={name:maxnorm(v) for name,v in comp.items()}
    for split,ids in [('old32',old_ids),('new64',selected_ids)]:
        inputs=[];labels=[];evidence=[]
        for c in ids:
            i=lookup[c];row=valid[i];candidates=[{'movie_id':str(model.items[j]),'title':model.names[model.items[j]],
                'score':float(score[i,j]),'components':{name:float(v[i,j]) for name,v in norm.items()}} for j in ix[i]]
            inputs.append({k:v for k,v in row.items() if k!='targets'}|{'candidates':candidates})
            labels.append({'id':c,'targets':row['targets']});evidence.append({'id':c,**model.evidence(row,[m['movie_id'] for m in candidates])})
        save(INNER/(split+'.inputs.json'),inputs);save(INNER/(split+'.labels.json'),labels);save(INNER/(split+'.evidence.json'),evidence)
    print('FINAL',json.dumps(summary),flush=True)
if __name__=='__main__':run()
