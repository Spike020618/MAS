"""Frozen calibrated-count transfer to public validation, no re-fitting/tuning.

Only Sparse teacher is refit on all ORIGINAL TRAIN, matching the earlier serving
pipeline. Calibrator stays byte-identical. Dense calibrated arrays carry NaN
outside the predeclared two-top100 union; unsupported items are not extrapolated.
"""
import argparse,gc,json
from pathlib import Path
import numpy as np
from experiments.consensus_recommendation.common import ROOT,create,sha,fingerprint
from experiments.paired_deliberation.native_calibrated_challenge import RESULT as CALIBRATOR,CalibratedChallengeModel,pool_indices,candidate_features,FEATURES
from experiments.paired_deliberation.native_retrieval_optimization import NativeRetriever,enhanced_contexts,fuse
from experiments.paired_deliberation.native_kbrd_scores import PrecomputedKBRD
from experiments.paired_deliberation.native_redial_benchmark import DATA,evaluate

RESULT=ROOT/'experiments/results/native_calibrated_transfer_v1'
SPARSE_CONFIG=ROOT/'experiments/results/native_retrieval_inner_v1/selected_config.json'
KBRD=ROOT/'experiments/results/native_kbrd_reproduction_v1/train_recall_r1/export'


def save_array(path,array):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:np.save(stream,array)
    temporary.replace(path)


def provenance_card(kind,case_id,mid,values,provenance):
    return {'source_id':'predictor:'+kind+':'+mid,'kind':'model_prediction','allowed_use':'prediction_basis',
        'origin_case_id':'original_train_fitted_predictor','query_case_id':case_id,'movie_id':mid,
        'text':json.dumps({'predictor':kind,'movie_id':mid,'values':values,'interpretation':'uncertain next-movie prediction, not current-user fact or semantic truth'},separators=(',',':')),
        'provenance':provenance,'current_user_fact_authority':False,'semantic_truth_certified':False}


class PrecomputedCalibratedCounts:
    """Label-free scores and shared material for the sealed valid input cohort."""
    def __init__(self,root=RESULT):
        self.root=Path(root);self.ids=json.loads((self.root/'valid.ids.json').read_text());self.items=json.loads((self.root/'native_item_ids.json').read_text())
        self.row_index={c:i for i,c in enumerate(self.ids)};self.item_index={str(m):j for j,m in enumerate(self.items)}
        self.count=np.load(self.root/'valid.expected_count.npy',mmap_mode='r');self.sd=np.load(self.root/'valid.count_sd.npy',mmap_mode='r')
        self.sparse_score=np.load(self.root/'valid.sparse_scores.npy',mmap_mode='r');self.sparse_rank=np.load(self.root/'valid.sparse_ranks.npy',mmap_mode='r')
        self.kbrd_rank=np.load(self.root/'valid.kbrd_ranks.npy',mmap_mode='r');self.kbrd=PrecomputedKBRD(KBRD)
        self.offset=np.load(self.root/'valid.candidate_offsets.npy',mmap_mode='r');self.candidates=np.load(self.root/'valid.candidate_ids.npy',mmap_mode='r')
        self.members=np.load(self.root/'valid.member_expected_count.npy',mmap_mode='r');self.meta=json.loads((self.root/'serving_metadata.json').read_text())
    def score_pool(self,case_id,candidate_ids):
        q=self.row_index[case_id];ids=list(map(str,candidate_ids));j=[self.item_index[m] for m in ids]
        sl=slice(self.offset[q],self.offset[q+1]);lookup={str(m):k for k,m in enumerate(self.candidates[sl])}
        rows=[];members=[{} for _ in range(3)];counts={}
        for mid,k in zip(ids,j):
            valid=bool(np.isfinite(self.count[q,k]));value=float(self.count[q,k]) if valid else None
            rows.append([mid,value,float(self.sd[q,k]) if valid else None,valid,float(self.sparse_score[q,k]),int(self.sparse_rank[q,k]),int(self.kbrd_rank[q,k])])
            if valid:
                counts[mid]=value
                for n in range(3):members[n][mid]=float(self.members[n,sl][lookup[mid]])
        return {'case_id':case_id,'columns':['movie_id','expected_target_count','model_stability_sd','supported_union_item','sparse_score','sparse_native_rank','kbrd_native_rank'],
            'rows':rows,'expected_counts':counts,'member_expected_counts':members,'kbrd':self.kbrd.score_pool(case_id,ids),
            'scope':'frozen original-train calibrator; unsupported union-external counts are None, not zero estimates',
            'current_user_fact_authority':False,'semantic_truth_certified':False,'actual_lm_review_effect_estimated':False}
    def augment(self,row):
        q=self.row_index[row['id']];sl=slice(self.offset[q],self.offset[q+1]);ids=list(map(str,self.candidates[sl]));scored=self.score_pool(row['id'],ids)
        by={r[0]:r for r in scored['rows']};kbrd={r[0]:r for r in scored['kbrd']['rows']};entities=json.loads((DATA/'released/entity2id.json').read_text())
        names={str(v):k.removeprefix('<http://dbpedia.org/resource/').removesuffix('>').replace('_',' ') for k,v in entities.items()}
        ordered=sorted(ids,key=lambda m:(-scored['expected_counts'][m],ids.index(m)));sources={}
        for mid in ids:
            r=by[mid];kb=kbrd[mid]
            sources['predictor:sparse:'+mid]=provenance_card('sparse',row['id'],mid,{'score':r[4],'native_rank':r[5]},self.meta['sparse'])
            sources['predictor:kbrd:'+mid]=provenance_card('kbrd',row['id'],mid,{'logit':kb[1],'native_z':kb[2],'native_rank':kb[3]},self.meta['kbrd'])
            sources['predictor:calibrated:'+mid]=provenance_card('calibrated',row['id'],mid,{'expected_occurrence_count':r[1],'model_stability_sd':r[2],
                'member_expected_counts':[m[mid] for m in scored['member_expected_counts']]},self.meta['calibrator'])
        return {k:v for k,v in row.items() if k not in ('targets','candidates','sources')}|{
            'candidates':[{'movie_id':m,'title':names[m],'score':scored['expected_counts'][m]} for m in ordered],
            'sources':sources,'calibrated_expected_counts':scored['expected_counts'],'calibrated_member_expected_counts':scored['member_expected_counts'],
            'calibrated_score_columns':['movie_id','expected_occurrence_count','model_stability_sd','member0','member1','member2'],
            'calibrated_score_rows':[[m,by[m][1],by[m][2],*[member[m] for member in scored['member_expected_counts']]] for m in ordered],
            'predictor_score_columns':['movie_id','sparse_score','sparse_native_rank','kbrd_logit','kbrd_z','kbrd_native_rank'],
            'predictor_score_rows':[[m,by[m][4],by[m][5],*kbrd[m][1:]] for m in ordered],
            'predictor_rankings':{'sparse':sorted(ids,key=lambda m:by[m][5]),'kbrd':sorted(ids,key=lambda m:by[m][6]),
                'rrf60':sorted(ids,key=lambda m:(-(.5/(60+by[m][5])+.5/(60+by[m][6])),by[m][5])),'calibrated':ordered},
            'candidate_policy':'both trained predictors top100 union, no target insertion; calibrated ranker sorts only supported union',
            'all_methods_must_receive_same_scores':True}


def run(root=RESULT):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);calibrator=CalibratedChallengeModel();kbrd=PrecomputedKBRD(KBRD)
    cfg=json.loads(SPARSE_CONFIG.read_text())['config'];calhash=sha(CALIBRATOR/'calibrated_model.joblib')
    plan={'version':'native-calibrated-transfer-v1','calibrator_sha256':calhash,'calibrator_frozen_no_valid_fit':True,'sparse_fixed_config':cfg,
        'source_sha256':sha(Path(__file__)),'no_public_valid_parameter_selection':True,'no_test_or_legacy_reserved_read':True,
        'outside_union':'NaN unsupported, never pretend calibrated zero; original native target miss denominator retained',
        'teacher_shift':'calibrator fitted on fold80->hold20, serving teacher fitted fulltrain; rank/z/relative features retained, drift audited only'}
    create(root/'protocol.json',plan)
    path=DATA/'released';entities=json.loads((path/'entity2id.json').read_text());items=json.loads((path/'movie_ids.json').read_text())
    train=enhanced_contexts(json.loads((path/'train_data.json').read_text()),entities)
    validation=enhanced_contexts(json.loads((path/'valid_data.json').read_text()),entities)
    inputs=[{k:v for k,v in row.items() if k!='targets'} for row in validation];ids=[r['id'] for r in inputs]
    if set(ids)!=set(kbrd.ids) or items!=kbrd.items:raise ValueError('native exported KBRD validation alignment')
    create(root/'valid.ids.json',ids);create(root/'native_item_ids.json',items);create(root/'valid.inputs.json',inputs)
    if not(root/'sparse_complete.json').exists():
        model=NativeRetriever(items,entities).fit(train);comp=model.components(inputs);values=fuse(comp,cfg)
        order=np.argsort(-values,axis=1,kind='stable');ranks=np.empty_like(order,dtype=np.uint16);ranks[np.arange(len(order))[:,None],order]=np.arange(1,len(items)+1,dtype=np.uint16)
        native=np.array([model.index[m] for m in items]);save_array(root/'valid.sparse_scores.npy',values[:,native]);save_array(root/'valid.sparse_ranks.npy',ranks[:,native])
        create(root/'sparse_complete.json',{'fit_contexts':len(train),'configuration':cfg,'public_valid_labels_used_to_fit':False,'shape':[len(inputs),len(items)]})
        del model,comp,values,ranks,order;gc.collect()
    ss=np.load(root/'valid.sparse_scores.npy',mmap_mode='r');sr=np.load(root/'valid.sparse_ranks.npy',mmap_mode='r')
    gather=[kbrd.row_index[c] for c in ids];logits=np.asarray(kbrd.matrix[gather]);ko=np.argsort(-logits,axis=1,kind='stable');kr=np.empty_like(ko,dtype=np.uint16)
    kr[np.arange(len(ko))[:,None],ko]=np.arange(1,len(items)+1,dtype=np.uint16);save_array(root/'valid.kbrd_ranks.npy',kr)
    shape=(len(ids),len(items));temp=root/'valid.expected_count.tmp.npy';dense=np.lib.format.open_memmap(temp,mode='w+',dtype=np.float32,shape=shape);dense[:]=np.nan
    sdtemp=root/'valid.count_sd.tmp.npy';deviation=np.lib.format.open_memmap(sdtemp,mode='w+',dtype=np.float32,shape=shape);deviation[:]=np.nan
    offsets=[0];candidate_rows=[];member_rows=[[],[],[]];predictions=[];feature_sum=np.zeros(len(FEATURES));feature_square=np.zeros(len(FEATURES));feature_n=0
    for i,row in enumerate(inputs):
        ix=pool_indices(sr[i],kr[i]);served=calibrator.score_pool(row,items,ss[i],sr[i],logits[i],kr[i],ix)
        vals=np.array([r[1] for r in served['rows']]);sds=np.array([r[2] for r in served['rows']]);dense[i,ix]=vals;deviation[i,ix]=sds
        candidate_rows.extend(np.asarray(items)[ix]);offsets.append(len(candidate_rows))
        for n in range(3):member_rows[n].extend(served['member_expected_counts'][n][str(items[j])] for j in ix)
        x=candidate_features(row,items,ix,ss[i],sr[i],logits[i],kr[i]);feature_sum+=x.sum(axis=0);feature_square+=(x.astype(float)**2).sum(axis=0);feature_n+=len(x)
        sparse_order=np.argsort(sr[i],kind='stable');kbrd_order=np.argsort(kr[i],kind='stable');rrf=.5/(60+sr[i].astype(float))+.5/(60+kr[i].astype(float));rrfo=np.lexsort((sr[i],-rrf))
        predictions.append({'id':row['id'],'calibrated':list(map(int,served['ranking'])),'sparse':[items[j] for j in sparse_order[:100]],
            'kbrd':[items[j] for j in kbrd_order[:100]],'rrf60_native':[items[j] for j in rrfo[:100]],'union_size':len(ix)})
        if i%200==0:print('transfer',i,'/',len(inputs),flush=True)
    dense.flush();deviation.flush();del dense,deviation;temp.replace(root/'valid.expected_count.npy');sdtemp.replace(root/'valid.count_sd.npy')
    save_array(root/'valid.candidate_offsets.npy',np.array(offsets,dtype=np.int64));save_array(root/'valid.candidate_ids.npy',np.array(candidate_rows,dtype=np.int32));save_array(root/'valid.member_expected_count.npy',np.array(member_rows,dtype=np.float32))
    create(root/'valid.predictions.json',predictions)
    meta={'calibrator':{'model_sha256':calhash,'fit_split':'original train calibration-fit only','selected_model':calibrator.bundle['selected_config'],'outcome':'expected target occurrence count; not real review efficacy'},
        'sparse':{'fit_split':'full original train','configuration':cfg},'kbrd':kbrd.meta,'matrix_order':'valid.ids.json x native_item_ids.json',
        'unsupported_count_is_nan':True,'candidate_pool':'two predictor native top100 union <=200','no_validation_recalibration':True,
        'feature_names':list(FEATURES),'serving_feature_mean':(feature_sum/feature_n).tolist(),
        'serving_feature_std':np.sqrt(np.maximum(0,feature_square/feature_n-(feature_sum/feature_n)**2)).tolist()}
    create(root/'serving_metadata.json',meta);create(root/'generation_complete.json',{'calibrator_sha256':calhash,'predictions_sha256':sha(root/'valid.predictions.json'),'no_validation_recalibration':True})
    serving=PrecomputedCalibratedCounts(root);lookup={r['id']:r for r in inputs}
    for name in ('old32','new64'):
        selected=json.loads((ROOT/'experiments/results/native_retrieval_inner_v1'/(name+'.inputs.json')).read_text());materials=[serving.augment(lookup[r['id']]) for r in selected]
        create(root/(name+'.inputs.json'),materials)
    if sha(CALIBRATOR/'calibrated_model.joblib')!=calhash:raise ValueError('calibrator changed during inference')
    # Performance evaluation is downstream of frozen predictions, never tuning.
    summary={'scope':'public valid transfer; no validation refit or parameter selection; historical public validation development remains disclosed',
        'calibrator_sha256_before_after_equal':True,'full_valid':{},'sets':{}}
    for method in ('sparse','kbrd','rrf60_native','calibrated'):summary['full_valid'][method]=evaluate(validation,[p[method] for p in predictions])
    index={r['id']:i for i,r in enumerate(validation)}
    for name in ('old32','new64'):
        selected=json.loads((ROOT/'experiments/results/native_retrieval_inner_v1'/(name+'.inputs.json')).read_text());ix=[index[r['id']] for r in selected]
        summary['sets'][name]={method:evaluate([validation[i] for i in ix],[predictions[i][method] for i in ix]) for method in ('sparse','kbrd','rrf60_native','calibrated')}
    create(root/'summary.json',summary);print(json.dumps(summary,indent=2));return summary

if __name__=='__main__':run()
