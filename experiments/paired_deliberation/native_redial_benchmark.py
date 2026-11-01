"""CRSLab released ReDial item-target protocol; train-only text-neighbor baseline."""
import json, hashlib, zipfile, argparse
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from experiments.paired_deliberation.benchmark_evaluation import native_item_metrics

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'experiments/data/raw/crslab_redial_native'
OUT=ROOT/'experiments/results/native_redial_benchmark_v1'

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def contexts(raw,entities):
    """Merge adjacent roles; expose prefixes only; retain repeated response targets."""
    out=[]
    for conv in raw:
        merged=[]
        for u in conv['dialog']:
            movies=[entities[m] for m in u['movies'] if m in entities]
            if merged and merged[-1]['role']==u['role']:
                merged[-1]['tokens']+=u['text']; merged[-1]['targets']+=movies
            else: merged.append({'role':u['role'],'tokens':list(u['text']),'targets':movies})
        prefix=[]
        for i,u in enumerate(merged):
            if prefix and u['role']=='Recommender' and u['targets']:
                out.append({'id':str(conv['conv_id'])+':'+str(i),
                            'text':'\n'.join(prefix),'targets':list(u['targets'])})
            prefix.append(u['role']+': '+' '.join(u['tokens']))
    return out

def evaluate(rows,ranks):
    totals=Counter(); n=0; covered=0
    for row,rank in zip(rows,ranks):
        for target in row['targets']:
            n+=1; covered+=target in rank[:100]
            totals.update(native_item_metrics(list(map(int,rank)),target))
    return {'contexts':len(rows),'item_targets':n,'metrics':{k:v/n for k,v in totals.items()},'retrieval_recall@100':covered/n}

def run():
    p=DATA/'released'; z=DATA/'redial_nltk.zip'
    expected='01dc2ebf15a0988a92112daa7015ada3e95d855e80cc1474037a86e536de3424'
    assert hashlib.sha256(z.read_bytes()).hexdigest()==expected
    entities=json.loads((p/'entity2id.json').read_text()); items=json.loads((p/'movie_ids.json').read_text())
    index={m:i for i,m in enumerate(items)}
    splits={}
    with zipfile.ZipFile(z) as archive:
        for split in ('train','valid','test'):
            splits[split]=contexts(json.loads(archive.read(split+'_data.json')),entities)
    assert all(t in index for rows in splits.values() for r in rows for t in r['targets'])
    train=splits['train']; counts=Counter(t for r in train for t in r['targets'])
    pop=sorted(items,key=lambda m:(-counts[m],m)); tie={m:i for i,m in enumerate(pop)}
    vec=TfidfVectorizer(lowercase=True,ngram_range=(1,2),min_df=2,max_features=60000,sublinear_tf=True,dtype=np.float32)
    x=vec.fit_transform([r['text'] for r in train])
    rr=[]; cc=[]; vv=[]
    for i,r in enumerate(train):
        for t,c in Counter(r['targets']).items():rr.append(i);cc.append(index[t]);vv.append(c/len(r['targets']))
    y=csr_matrix((np.array(vv,dtype=np.float32),(rr,cc)),shape=(len(train),len(items)))
    manifest={'protocol':'CRSLab nltk v0.31, merged roles, recommender-prefix per-item targets; duplicates retained',
      'archive_sha256':expected,'native_items':len(items),'source_commit':'6497938919998b9cd7aa9c2f43effb39ad0c7088',
      'retrieval':'TF-IDF cosine top-50 training recommendation contexts, weighted item vote; popularity tie-break',
      'hyperparameters':{'neighbors':50,'ngram':[1,2],'min_df':2,'max_features':60000},
      'candidate_policy':'full native item universe, no target insertion, no prior-mention exclusion',
      'test_status':'public benchmark test; project previously used official test, not pristine confirmation',
      'fit_split':'train only','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(OUT/'protocol.json',manifest)
    names={v:k.removeprefix('<http://dbpedia.org/resource/').removesuffix('>').replace('_',' ') for k,v in entities.items()}
    summary={}
    for split in ('valid','test'):
        rows=splits[split]; q=vec.transform([r['text'] for r in rows]); ranks=[]
        for start in range(0,len(rows),64):
            sim=(q[start:start+64]@x.T).toarray()
            for s in sim:
                neighbors=np.argsort(-s,kind='stable')[:50]
                scores=(s[neighbors]@y[neighbors]).ravel()
                ranked=sorted(range(len(items)),key=lambda j:(-float(scores[j]),tie[items[j]]))[:100]
                ranks.append([items[j] for j in ranked])
        predictions=[{'id':r['id'],'ranking':rank} for r,rank in zip(rows,ranks)]
        save(OUT/(split+'.retrieval.predictions.json'),predictions)
        # Predictions are materialized before the evaluator consumes response labels.
        save(OUT/(split+'.labels.json'),[{'id':r['id'],'targets':r['targets']} for r in rows])
        summary[split]={'popularity':evaluate(rows,[pop[:100]]*len(rows)),'tfidf_neighbors':evaluate(rows,ranks)}
        if split=='valid':
            selected=sorted(range(len(rows)),key=lambda i:hashlib.sha256(('native-v1:'+rows[i]['id']).encode()).hexdigest())[:32]
            save(OUT/'single_llm.inputs.json',[{'id':rows[i]['id'],'text':rows[i]['text'],
                'candidates':[{'movie_id':str(m),'title':names[m]} for m in ranks[i]]} for i in selected])
            save(OUT/'single_llm.labels.json',[{'id':rows[i]['id'],'targets':rows[i]['targets']} for i in selected])
            summary['valid_fixed32']={'popularity':evaluate([rows[i] for i in selected],[pop[:100]]*len(selected)),
                'tfidf_neighbors':evaluate([rows[i] for i in selected],[ranks[i] for i in selected])}
    summary['train']={'contexts':len(train),'item_targets':sum(len(r['targets']) for r in train),'conversations':8104}
    save(OUT/'summary.json',summary); print(json.dumps(summary,indent=2))

if __name__=='__main__':run()
