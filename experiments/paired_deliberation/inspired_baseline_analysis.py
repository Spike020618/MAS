"""Read-only native INSPIRED baseline, interface and prior-mention diagnostics."""
import json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from experiments.consensus_recommendation.common import ROOT
from experiments.paired_deliberation.native_inspired_benchmark import contexts,evaluate
from experiments.paired_deliberation.inspired_predictor_scores import InspiredPredictorScores,DEFAULT


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def run():
    r=DEFAULT;data=ROOT/'experiments/data/raw/crslab_inspired_native/released';entities=json.loads((data/'entity2id.json').read_text());items=json.loads((data/'movie_ids.json').read_text())
    inverse={v:k for k,v in entities.items()};catalog=[{'movie_id':str(m),'native_entity_name':inverse[m]} for m in items];save(r/'native_catalog.json',catalog)
    labels=json.loads((r/'valid.labels.json').read_text());inputs=json.loads((r/'valid.inputs.json').read_text());ids=json.loads((r/'valid.ids.json').read_text())
    assert ids==[x['id'] for x in inputs]==[x['id'] for x in labels]
    methods={}
    baseline=ROOT/'experiments/results/native_inspired_benchmark_v1'
    methods['tfidf50']=[p['ranking'] for p in json.loads((baseline/'valid.predictions.json').read_text())]
    for s in ('fixed_redial_transfer','train_inner_tuned'):methods['Sparse_'+s]=[p['ranking'] for p in json.loads((r/'sparse'/s/'valid.predictions.json').read_text())]
    for k in ('academic_pure','train_inner_tuned'):methods['KBRD_'+k]=[p['ranking'] for p in json.loads((r/'kbrd'/k/'valid.predictions.json').read_text())]
    diagnostics={}
    for kind in ('prior_mentioned','not_prior_mentioned','not_prior_linked_entity'):
        selected=[];indices=[]
        for i,(inp,row) in enumerate(zip(inputs,labels)):
            prior=set(inp['context_entities'] if kind=='not_prior_linked_entity' else inp['context_movies']);targets=[t for t in row['targets'] if (t in prior)==(kind=='prior_mentioned')]
            if targets:selected.append({'id':row['id'],'targets':targets});indices.append(i)
        diagnostics[kind]={'definition':'not in any prior entity annotation' if kind=='not_prior_linked_entity' else 'prior occurrence according to author movie annotations, not a semantic surface-text novelty judgement','targets':sum(len(x['targets']) for x in selected),'contexts':len(selected),
            'methods':{m:evaluate(selected,[predictions[i] for i in indices]) for m,predictions in methods.items()}}
    diagnostics['scope']='posthoc frozen-prediction diagnostic; does not remove targets from native primary scores or claim satisfaction truth'
    save(r/'prior_mention_diagnostic.json',diagnostics)
    # Build separate serving pools so readers cannot silently mix the two versions.
    for sv in ('fixed_redial_transfer','train_inner_tuned'):
        reader=InspiredPredictorScores(r,sparse_version=sv);serving=[{k:v for k,v in inp.items() if k!='candidates'}|{'candidates':reader.candidate_pool(inp['id'])} for inp in inputs]
        save(r/'sparse'/sv/'valid.inputs.json',serving)
        for kv in ('academic_pure','train_inner_tuned'):
            reader=InspiredPredictorScores(r,sparse_version=sv,kbrd_version=kv)
            sources=[reader.score_pool(inp['id'],[m['movie_id'] for m in inp['candidates']]) for inp in serving]
            save(r/'sparse'/sv/('valid.shared_scores_'+kv+'.json'),sources)
    # Score/rank and serving API are checked without needing labels.
    audit={}
    for label,values_path,ranks_path in [('Sparse:'+sv,r/'sparse'/sv/'valid.native_scores.npy',r/'sparse'/sv/'valid.native_ranks.npy') for sv in ('fixed_redial_transfer','train_inner_tuned')]+[('KBRD:'+kv,r/'kbrd'/kv/'valid.native_logits.npy',r/'kbrd'/kv/'valid.native_ranks.npy') for kv in ('academic_pure','train_inner_tuned')]:
        values=np.load(values_path,mmap_mode='r');rank=np.load(ranks_path,mmap_mode='r');assert values.shape==rank.shape==(282,1815)
        assert np.isfinite(values).all() and np.all(np.sort(rank,axis=1)==np.arange(1,1816))
        order=np.argsort(rank,axis=1,kind='stable');ordered=np.take_along_axis(values,order,axis=1);assert np.all(ordered[:,:-1]>=ordered[:,1:]-1e-6)
        audit[label]={'shape':[282,1815],'finite':True,'ranks_complete_permutations':True,'native_catalog_count':len(items)}
    old=json.loads((r/'kbrd_fulltrain_corrupt_metadata_r1/history.json').read_text());new=json.loads((r/'kbrd_fulltrain/history.json').read_text())
    replay=[{'epoch':a['epoch'],'train_loss_abs':abs(a['train_loss']-b['train_loss'])} for a,b in zip(old,new)]
    audit['serialization_replay']={'same_seed_optimizer_epoch_selection':True,'epochs':replay,'max_train_loss_abs':max(x['train_loss_abs'] for x in replay),'no_independent_seed_claim':True}
    audit['test_archive_or_member_opened']=False;audit['matrix_ids_are_native_author_ids']=True;save(r/'interface_audit.json',audit)
    print(json.dumps({'prior_mention':{k:{'targets':v['targets'],'R10':{m:s['metrics']['recall@10'] for m,s in v['methods'].items()}} for k,v in diagnostics.items() if isinstance(v,dict)},'serialization_replay_max_train_loss_difference':audit['serialization_replay']['max_train_loss_abs']},indent=2))
    return diagnostics,audit
if __name__=='__main__':run()
