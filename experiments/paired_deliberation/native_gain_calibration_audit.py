"""Post-generation fold-only predictor matrix/data audit (no public evaluation)."""
import hashlib,json
from pathlib import Path
import numpy as np
from experiments.paired_deliberation.native_redial_benchmark import ROOT,evaluate,save

RESULT=ROOT/'experiments/results/native_gain_calibration_v1'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(2**20),b''):h.update(b)
    return h.hexdigest()


def ranks_to_predictions(rank,items,cutoff=100):
    ix=np.argsort(rank,axis=1,kind='stable')[:,:cutoff]
    return [[items[j] for j in row] for row in ix]


def run():
    r=RESULT
    for name in ('sparse','kbrd'):
        if not (r/name/'complete.json').exists():raise RuntimeError('predictor generation incomplete; no label audit yet')
    ids=json.loads((r/'hold20.ids.json').read_text());items=json.loads((r/'native_item_ids.json').read_text());inputs=json.loads((r/'hold20.inputs.json').read_text())
    fitids=json.loads((r/'fold80.ids.json').read_text());fitconv={x.split(':')[0] for x in fitids};holdconv={x.split(':')[0] for x in ids}
    assert fitconv.isdisjoint(holdconv) and len(set(ids))==len(ids) and [x['id'] for x in inputs]==ids
    assert len(set(items))==len(items)==6924 and all('targets' not in x for x in inputs)
    digest=hashlib.sha256(json.dumps(fitids).encode()).hexdigest()
    sm=json.loads((r/'sparse/metadata.json').read_text());km=json.loads((r/'kbrd/metadata.json').read_text())
    assert sm['fit_id_hash']==km['fit_id_hash']==digest
    history=json.loads((r/'kbrd/history.json').read_text());assert len(history)==9 and all(x['holdout_evaluated'] is False for x in history)
    sr=np.load(r/'sparse/hold20.native_ranks.npy',mmap_mode='r');kr=np.load(r/'kbrd/hold20.native_ranks.npy',mmap_mode='r')
    ss=np.load(r/'sparse/hold20.native_scores.npy',mmap_mode='r');ks=np.load(r/'kbrd/hold20.native_logits.npy',mmap_mode='r')
    assert sr.shape==kr.shape==ss.shape==ks.shape==(len(ids),len(items)) and sr.dtype==kr.dtype==np.uint16
    assert np.isfinite(ss).all() and np.isfinite(ks).all()
    tie=np.array(sm['native_popularity_tie_order']);max_error=0.
    expected_rank=np.arange(1,len(items)+1,dtype=np.uint16)
    components={name:np.load(r/'sparse'/('hold20.'+name+'.npy'),mmap_mode='r') for name in sm['fixed_config']}
    for start in range(0,len(ids),64):
        end=start+64
        for values,ranks,order in ((ss[start:end],sr[start:end],tie),(ks[start:end],kr[start:end],np.arange(len(items)))):
            assert np.all(np.sort(ranks,axis=1)==expected_rank)
            sorted_indices=order[np.argsort(-values[:,order],axis=1,kind='stable')]
            assert np.all(np.take_along_axis(ranks,sorted_indices,axis=1)==expected_rank)
        recomputed=np.zeros_like(ss[start:end])
        for name,weight in sm['fixed_config'].items():
            part=np.asarray(components[name][start:end]);recomputed+=weight*part/np.maximum(part.max(axis=1,keepdims=True),1e-12)
        max_error=max(max_error,float(np.abs(recomputed-ss[start:end]).max()))
    assert max_error<1e-6
    z=np.load(r/'kbrd/hold20.full_entity_logsumexp.npy');native_z=ks.max(axis=1)+np.log(np.exp(ks-ks.max(axis=1,keepdims=True)).sum(axis=1))
    assert z.shape==(len(ids),) and np.isfinite(z).all() and np.all(z>=native_z-1e-5)
    access={}
    for phase in ('sparse','kbrd'):
        a=json.loads((r/('access_audit_'+phase+'.json')).read_text());assert all('valid_data.json' not in p and 'test_data.json' not in p for p in a['opened_dataset_json_paths']);access[phase]=a
    # Both predictor matrices are now frozen before consuming calibration labels.
    labels=json.loads((r/'hold20.labels.json').read_text());assert [x['id'] for x in labels]==ids
    sp=ranks_to_predictions(sr,items);kp=ranks_to_predictions(kr,items)
    n=sum(len(x['targets']) for x in labels);union100=sum(t in set(a)|set(b) for row,a,b in zip(labels,sp,kp) for t in row['targets'])
    union10=sum(t in set(a[:10])|set(b[:10]) for row,a,b in zip(labels,sp,kp) for t in row['targets'])
    audit={'fit_hold_conversations_disjoint':True,'fit_conversations':len(fitconv),'hold_conversations':len(holdconv),'hold_prefixes':len(ids),'hold_item_targets':n,
       'two_predictor_fit_id_hash_equal':True,'matrix_shapes':[len(ids),len(items)],'rank_permutations_all_rows_verified':True,
       'Sparse_component_fusion_max_abs_difference':max_error,'native_movie_logits_probability_bound_verified':True,
       'KBRD_epoch_fixed_no_holdout_eval':9,'labels_read_after_both_generation_complete_flags':True,'no_public_valid_test_read':True,
       'access_audits':access,'Sparse_hold20_development_metrics':evaluate(labels,sp),'KBRD_hold20_development_metrics':evaluate(labels,kp),
       'union_top100_covered_target_count':union100,'union_top100_coverage':union100/n,'union_top10_covered_target_count':union10,
       'calibration_label_scope':'only original train holdout, not public valid; historically reused inner split, not globally pristine',
       'source_sha256':sha(Path(__file__))}
    save(r/'data_audit.json',audit)
    save(r/'prediction_complete.json',{'hold_ids_sha256':sha(r/'hold20.ids.json'),'labels_sha256':sha(r/'hold20.labels.json'),
       'matrix_sha256':{str(p.relative_to(r)):sha(p) for p in [r/'sparse/hold20.native_scores.npy',r/'sparse/hold20.native_ranks.npy',r/'kbrd/hold20.native_logits.npy',r/'kbrd/hold20.native_ranks.npy']},
       'score_labels_not_used_to_train_predictors':True})
    print(json.dumps({k:v for k,v in audit.items() if k!='access_audits'},indent=2));return audit
if __name__=='__main__':run()
