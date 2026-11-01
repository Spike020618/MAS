"""Train-only cross-fitted target-count calibration and bounded challenge proxies.

Teacher scores must be out-of-sample fold80 -> hold20 predictions. The original
hold20 is split by conversation into calibration-fit and untouched diagnostic;
all model/hyperparameter/calibration fitting remains in calibration-fit. No API,
public validation/test, or legacy reserved labels are used here.

c(m) estimates the expected occurrence count among next-movie targets, retaining
repeated targets. Ideal swap b for a has expected micro-hit numerator gain c(b)-c(a),
insertion benefit c(b), and removal harm c(a). Future target count is not used in serving. This is NOT an actual LM review effect/VOI.
"""
import argparse,copy,hashlib,json,math
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,PoissonRegressor
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from experiments.consensus_recommendation.common import ROOT,create,sha,fingerprint
from experiments.paired_deliberation.native_redial_benchmark import evaluate
from experiments.paired_deliberation.native_candidate_challenge import consensus_ranking,build_targets as fixed_targets,allocate

DATA=ROOT/'experiments/results/native_gain_calibration_v1'
RESULT=ROOT/'experiments/results/native_calibrated_challenge_v3'
OUTER_SEED='native-calibration-diagnostic-20261028:'
INNER_SEED='native-calibration-selection-20261028:'
OOF_SEED='native-calibration-oof-20261028:'
CONFIGS=({'family':'poisson','alpha':.01},{'family':'poisson','alpha':.1},{'family':'poisson','alpha':1.},
 {'family':'hist_poisson','max_leaf_nodes':7,'max_iter':80},{'family':'hist_poisson','max_leaf_nodes':15,'max_iter':80})
FEATURES=('log_sparse_rank','sparse_inverse_sqrt_rank','log_kbrd_rank','kbrd_inverse_sqrt_rank',
 'sparse_relative_max','sparse_native_z','kbrd_native_z','kbrd_native_probability',
 'sparse_top10','kbrd_top10','log_rank_gap','abs_log_rank_gap','both_top10','rrf60_relative_max',
 'candidate_previous_mention_count','candidate_previous_recency','candidate_is_last_mention',
 'log_prefix_movie_count','log_prefix_entity_count','log_prefix_word_count','log_seeker_turn_count',
 'log_recommender_turn_count','log_last_seeker_word_count','current_negative_cues','novelty_request_cues',
 'sparse_relative_x_has_mentions','kbrd_z_x_has_entities','log_rank_gap_x_novelty','log_candidate_pool_size')


def conv(case_id):return case_id.rsplit(':',1)[0]
def bucket(case_id,seed,n):return int(hashlib.sha256((seed+conv(case_id)).encode()).hexdigest(),16)%n

def split_indices(ids):
    fit=np.array([i for i,c in enumerate(ids) if bucket(c,OUTER_SEED,4)!=0],dtype=np.int32)
    diagnostic=np.array([i for i,c in enumerate(ids) if bucket(c,OUTER_SEED,4)==0],dtype=np.int32)
    inner_fit=np.array([i for i in fit if bucket(ids[i],INNER_SEED,5)!=0],dtype=np.int32)
    select=np.array([i for i in fit if bucket(ids[i],INNER_SEED,5)==0],dtype=np.int32)
    if any(not len(x) for x in (fit,diagnostic,inner_fit,select)):raise ValueError('conversation calibration partitions must be nonempty')
    assert {conv(ids[i]) for i in fit}.isdisjoint({conv(ids[i]) for i in diagnostic})
    return {'fit':fit,'diagnostic':diagnostic,'inner_fit':inner_fit,'select':select}


def context_features(row):
    text=row.get('original_text',row.get('text',''));last=row.get('last_seeker_text','');low=last.lower()
    movies=list(map(str,row.get('context_movies',[])));entities=row.get('context_entities',[])
    neg=sum(low.count(word) for word in ('not ','dislike','hate','avoid','boring'))
    novelty=sum(low.count(word) for word in ('new','different','else','another','modern','recent'))
    return movies,(math.log1p(len(movies)),math.log1p(len(entities)),math.log1p(len(text.split())),
        math.log1p(text.count('Seeker:')),math.log1p(text.count('Recommender:')),math.log1p(len(last.split())),float(neg),float(novelty))


def candidate_features(row,item_ids,indices,sparse_scores,sparse_ranks,kbrd_logits,kbrd_ranks):
    """All arrays are one query in the SAME full native item order."""
    s=np.asarray(sparse_scores,dtype=np.float64);z=np.asarray(kbrd_logits,dtype=np.float64)
    sr=np.asarray(sparse_ranks,dtype=np.float64)[indices];kr=np.asarray(kbrd_ranks,dtype=np.float64)[indices]
    sz=(s-s.mean())/max(float(s.std()),1e-8);kz=(z-z.mean())/max(float(z.std()),1e-8)
    relative=s/max(float(s.max()),1e-12);kp=np.exp(z-z.max());kp/=max(float(kp.sum()),1e-12)
    rrf=.5/(60+sr)+.5/(60+kr);rrf/=max(float(rrf.max()),1e-12)
    movies,cf=context_features(row);mentioned=Counter(movies);recency={}
    for age,m in enumerate(reversed(movies)):recency[m]=recency.get(m,0)+.8**age
    mids=[str(item_ids[j]) for j in indices];count=np.array([mentioned[m] for m in mids],dtype=float)
    recent=np.array([recency.get(m,0) for m in mids]);last=np.array([m==movies[-1] if movies else False for m in mids],dtype=float)
    ls=np.log1p(sr);lk=np.log1p(kr);gap=ls-lk;n=len(indices)
    columns=[ls,1/np.sqrt(sr),lk,1/np.sqrt(kr),relative[indices],np.clip(sz[indices],-20,20),np.clip(kz[indices],-20,20),kp[indices],
       (sr<=10).astype(float),(kr<=10).astype(float),gap,np.abs(gap),((sr<=10)&(kr<=10)).astype(float),rrf,
       count,recent,last,*[np.full(n,c) for c in cf],relative[indices]*bool(movies),np.clip(kz[indices],-20,20)*bool(row.get('context_entities')),
       gap*cf[-1],np.full(n,math.log1p(n))]
    x=np.column_stack(columns).astype(np.float32)
    if x.shape!=(n,len(FEATURES)) or not np.isfinite(x).all():raise ValueError('candidate feature shape/finite')
    return x


def pool_indices(sparse_ranks,kbrd_ranks,top=100):
    sr=np.asarray(sparse_ranks);kr=np.asarray(kbrd_ranks);ix=np.flatnonzero((sr<=top)|(kr<=top))
    score=.5/(60+sr[ix])+.5/(60+kr[ix]);return ix[np.lexsort((sr[ix],-score))]


def load_fold(data=DATA):
    data=Path(data)
    for part in ('sparse','kbrd'):
        if not (data/part/'complete.json').exists():raise RuntimeError(part+' fold-only teacher not yet complete')
    ids=json.loads((data/'hold20.ids.json').read_text());items=json.loads((data/'native_item_ids.json').read_text())
    rows=json.loads((data/'hold20.inputs.json').read_text());labels=json.loads((data/'hold20.labels.json').read_text())
    if [r['id'] for r in rows]!=ids or [r['id'] for r in labels]!=ids:raise ValueError('teacher row alignment')
    if any(int(hashlib.sha256(('inner-retrieval-20261028:'+conv(c)).encode()).hexdigest(),16)%5!=0 for c in ids):raise ValueError('input outside original train hold20')
    arrays={'sparse_scores':np.load(data/'sparse/hold20.native_scores.npy',mmap_mode='r'),
        'sparse_ranks':np.load(data/'sparse/hold20.native_ranks.npy',mmap_mode='r'),
        'kbrd_logits':np.load(data/'kbrd/hold20.native_logits.npy',mmap_mode='r'),
        'kbrd_ranks':np.load(data/'kbrd/hold20.native_ranks.npy',mmap_mode='r')}
    if any(a.shape!=(len(ids),len(items)) for a in arrays.values()):raise ValueError('teacher native shape/order mismatch')
    fit_ids=json.loads((data/'fold80.ids.json').read_text())
    if {conv(c) for c in ids}&{conv(c) for c in fit_ids}:raise ValueError('teacher training/holdout conversations overlap')
    for part in ('sparse','kbrd'):
        meta=json.loads((data/part/'metadata.json').read_text())
        expected=hashlib.sha256(json.dumps(fit_ids).encode()).hexdigest()
        observed=meta.get('fit_id_hash',meta.get('fit_ids_hash',meta.get('fit_ids_sha256')))
        if observed is not None and observed!=expected:raise ValueError('teacher fit ID fingerprint mismatch')
    return ids,items,rows,labels,arrays


def make_table(ids,items,rows,labels,arrays):
    xs=[];ys=[];counts=[];candidate_ids=[];query_index=[];offsets=[0];weights=[];itemlookup={int(m):j for j,m in enumerate(items)}
    for i,(row,label) in enumerate(zip(rows,labels)):
        ix=pool_indices(arrays['sparse_ranks'][i],arrays['kbrd_ranks'][i]);kwargs={k:np.asarray(v[i]) for k,v in arrays.items()}
        x=candidate_features(row,items,ix,**kwargs);target=Counter(label['targets']);n=len(label['targets'])
        if n==0 or any(int(t) not in itemlookup for t in target):raise ValueError('nonempty native target list')
        mass=np.array([target[int(items[j])]/n for j in ix],dtype=np.float32)
        xs.append(x);ys.append(mass);counts.append(np.array([target[int(items[j])] for j in ix],dtype=np.float32));candidate_ids.append(np.asarray(items,dtype=np.int32)[ix]);query_index.append(np.full(len(ix),i,dtype=np.int32))
        weights.append(np.full(len(ix),1/len(ix),dtype=np.float32));offsets.append(offsets[-1]+len(ix))
    w=np.concatenate(weights);w/=float(w.mean())
    return {'X':np.concatenate(xs),'mass':np.concatenate(ys),'count':np.concatenate(counts),'candidate_ids':np.concatenate(candidate_ids),'query_index':np.concatenate(query_index),
        'offsets':np.array(offsets,dtype=np.int64),'weights':w,'ids':ids,'rows':rows,'labels':labels,'feature_names':list(FEATURES)}


def subset_rows(table,queries):return np.flatnonzero(np.isin(table['query_index'],queries))


def binary_training(table,queries):
    ix=subset_rows(table,queries);x=table['X'][ix];mass=table['mass'][ix];w=table['weights'][ix];positive=mass>0;negative=mass<1
    xb=np.concatenate((x[negative],x[positive]));y=np.concatenate((np.zeros(negative.sum(),dtype=np.int8),np.ones(positive.sum(),dtype=np.int8)))
    wb=np.concatenate((w[negative]*(1-mass[negative]),w[positive]*mass[positive]))
    return x,w,xb,y,wb


def fit_estimator(table,queries,config,seed=271028):
    ix=subset_rows(table,queries);x=table['X'][ix];y=table['count'][ix];w=table['weights'][ix]
    if config['family']=='poisson':
        scaler=StandardScaler().fit(x,sample_weight=w);estimator=PoissonRegressor(alpha=config['alpha'],max_iter=300,tol=1e-7)
        estimator.fit(scaler.transform(x),y,sample_weight=w)
    elif config['family']=='hist_poisson':
        scaler=None;estimator=HistGradientBoostingRegressor(loss='poisson',learning_rate=.05,max_iter=config['max_iter'],
            max_leaf_nodes=config['max_leaf_nodes'],min_samples_leaf=50,l2_regularization=1.,early_stopping=False,random_state=seed)
        estimator.fit(x,y,sample_weight=w)
    else:raise ValueError('fixed count-model grid family')
    return {'config':dict(config),'scaler':scaler,'estimator':estimator}


def raw_expected_count(model,x):
    if model['scaler'] is not None:x=model['scaler'].transform(x)
    return np.maximum(np.asarray(model['estimator'].predict(x),dtype=float),1e-12)


def fit_mass_baseline(table,queries):
    x,w,xb,y,weights=binary_training(table,queries);scaler=StandardScaler().fit(x,sample_weight=w)
    estimator=LogisticRegression(C=1.,max_iter=300,tol=1e-6).fit(scaler.transform(xb),y,sample_weight=weights)
    return {'scaler':scaler,'estimator':estimator,'scope':'prefix-macro target-mass probability baseline, not micro count utility'}


def normalize_mass(values,query_index):
    p=np.clip(np.asarray(values,dtype=float),0,1).copy();groups=np.asarray(query_index,dtype=np.int32)
    if not len(p):return p
    totals=np.bincount(groups,weights=p)
    return p/np.maximum(1.,totals[groups])


def metrics_for(table,queries,expectation,calibration=True):
    ranks=[];labels=[];macro=[]
    for q in queries:
        sl=slice(table['offsets'][q],table['offsets'][q+1]);p=expectation[sl];items=table['candidate_ids'][sl]
        ranking=items[np.argsort(-p,kind='stable')].tolist();ranks.append(ranking);labels.append(table['labels'][q])
        macro.append(sum(t in ranking[:10] for t in table['labels'][q]['targets'])/len(table['labels'][q]['targets']))
    metric=evaluate(labels,ranks);metric['prefix_macro_recall@10']=float(np.mean(macro))
    if not calibration:return metric
    ix=subset_rows(table,queries);p=np.maximum(expectation[ix],1e-12);t=table['count'][ix];w=table['weights'][ix]
    logpart=np.zeros(len(t));positive=t>0;logpart[positive]=t[positive]*np.log(t[positive]/p[positive])
    metric['mean_poisson_deviance']=float(np.average(2*(logpart-(t-p)),weights=w))
    metric['count_mse']=float(np.average((p-t)**2,weights=w));metric['count_mae']=float(np.average(abs(p-t),weights=w))
    metric['mean_predicted_pool_target_count']=float(np.mean([expectation[table['offsets'][q]:table['offsets'][q+1]].sum() for q in queries]))
    metric['mean_actual_pool_target_count']=float(np.mean([table['count'][table['offsets'][q]:table['offsets'][q+1]].sum() for q in queries]))
    edges=(0,.001,.0025,.005,.01,.025,.05,.1,.2,.5,1,2,float('inf'));bins=[]
    for low,high in zip(edges[:-1],edges[1:]):
        mask=(p>=low)&(p<high)
        if mask.any():bins.append({'low':low,'high':high if math.isfinite(high) else 'infinity','candidate_rows':int(mask.sum()),'weight':float(w[mask].sum()),
            'mean_prediction':float(np.average(p[mask],weights=w[mask])),'actual_target_count':float(np.average(t[mask],weights=w[mask]))})
    metric['count_reliability_bins']=bins
    metric['weighted_count_calibration_error']=sum(b['weight']*abs(b['mean_prediction']-b['actual_target_count']) for b in bins)/sum(b['weight'] for b in bins)
    return metric


def baseline_ranking_scores(table,arrays,items):
    # These are rank-only baselines, not claimed calibrated probabilities.
    lookup={int(m):j for j,m in enumerate(items)};out={name:np.zeros(len(table['X']),dtype=float) for name in ('sparse','kbrd','rrf60')}
    for q in range(len(table['ids'])):
        sl=slice(table['offsets'][q],table['offsets'][q+1]);j=[lookup[int(m)] for m in table['candidate_ids'][sl]]
        sr=arrays['sparse_ranks'][q,j].astype(float);kr=arrays['kbrd_ranks'][q,j].astype(float)
        out['sparse'][sl]=1/sr;out['kbrd'][sl]=1/kr;out['rrf60'][sl]=.5/(60+sr)+.5/(60+kr)
    return out


def calibrated_targets(ranking,expected_counts,member_expected_counts=None,initial_rankings=None,max_targets=6,protected_prefix=3,uncertainty_weight=.5):
    """Full-pool proposals using calibrated insertion benefit and removal harm.

    mean positive member delta is a model-stability perfect-information envelope,
    NOT estimated LM review efficacy. No outside-union target is forced in.
    """
    if max_targets==0:return []
    if type(max_targets) is not int or max_targets<0:raise ValueError('nonnegative target count')
    ranking=list(map(str,ranking));p={str(m):float(v) for m,v in expected_counts.items()}
    if set(ranking)!=set(p) or len(ranking)!=len(set(ranking)) or any(not math.isfinite(v) or v<0 for v in p.values()):raise ValueError('closed nonnegative expected-count table')
    members=[{str(m):float(v) for m,v in q.items()} for q in (member_expected_counts or [p])]
    if any(set(q)!=set(p) for q in members):raise ValueError('member expected-count coverage')
    if any(not math.isfinite(v) or v<0 for q in members for v in q.values()):raise ValueError('finite nonnegative member expected counts')
    options=[]
    for a in ranking[protected_prefix:10]:
        for b in ranking[10:]:
            delta=p[b]-p[a];ds=np.array([q[b]-q[a] for q in members]);positive=float(np.maximum(ds,0).mean())
            unresolved=max(0.,positive-max(delta,0.));value=max(delta,0.)+uncertainty_weight*unresolved
            if value<=0:continue
            options.append({'a':a,'b':b,'value':value,'priority':value,'gain_count':p[b],'harm_count':p[a],
                'ideal_swap_gain':delta,'parameter_stability_sd':float(ds.std()),'conservative_ideal_gain':delta-float(ds.std()),
                'positive_member_delta_envelope':positive,'uncertainty_resolution_proxy':unresolved,
                'utility_scope':'target-count ideal swap plus optimistic model-disagreement envelope; NOT actual LM review gain'})
    options.sort(key=lambda t:(-t['value'],t['harm_count'],ranking.index(t['b']),ranking.index(t['a'])))
    out=[];used=set();initial=[[str(m) for m in r] for r in initial_rankings] if initial_rankings else None
    for target in options:
        if target['a'] in used or target['b'] in used:continue
        owner=max(range(len(initial)),key=lambda i:(10-initial[i].index(target['a']) if target['a'] in initial[i][:10] else 0,-i)) if initial else len(out)%3
        out.append({**target,'owner':owner,'target_id':'calibrated:'+target['a']+':'+target['b'],'protected_prefix':protected_prefix,
            'base_ranking_hash':fingerprint(ranking),'novel_to_initial_union':not any(target['b'] in r[:10] for r in initial) if initial else None})
        used.update((target['a'],target['b']))
        if len(out)>=max_targets:break
    return out


def allocate_calibrated(targets,cost_cny,mode='game',expected_hits_per_cny=1.,players=3):
    """Explicit units: utility is expected native hit count minus lambda * CNY cost."""
    if expected_hits_per_cny<0:raise ValueError('nonnegative declared cost conversion')
    costs=[[max(1e-12,float(c)*expected_hits_per_cny) for c in row] for row in cost_cny]
    result=allocate(targets,costs,mode,players)
    return {**result,'cost_units':'expected native hit count','expected_hits_per_cny':expected_hits_per_cny,
        'interpretation':'same-table calibrated proxy allocation; actual review accuracy still unmeasured'}


class CalibratedChallengeModel:
    """Trusted local joblib artifact only; numeric serving does not load labels."""
    def __init__(self,root=RESULT):
        root=Path(root);self.root=root;self.bundle=joblib.load(root/'calibrated_model.joblib');self.feature_names=self.bundle['feature_names']
        if self.feature_names!=list(FEATURES):raise ValueError('calibrator feature version')
    def score_pool(self,row,item_ids,sparse_scores,sparse_ranks,kbrd_logits,kbrd_ranks,candidate_indices=None):
        ix=pool_indices(sparse_ranks,kbrd_ranks) if candidate_indices is None else np.asarray(candidate_indices,dtype=np.int32)
        x=candidate_features(row,item_ids,ix,sparse_scores,sparse_ranks,kbrd_logits,kbrd_ranks);member=[]
        for model in self.bundle['models']:
            p=self.bundle['isotonic'].predict(raw_expected_count(model,x));member.append(np.maximum(p,0.))
        matrix=np.array(member);mean=matrix.mean(axis=0);std=matrix.std(axis=0);ids=[str(item_ids[j]) for j in ix]
        rows=[[m,float(p),float(s)] for m,p,s in zip(ids,mean,std)]
        return {'case_id':row['id'],'columns':['movie_id','predicted_target_count','model_stability_sd'],'rows':rows,
            'expected_counts':dict(zip(ids,map(float,mean))),'member_expected_counts':[dict(zip(ids,map(float,p))) for p in matrix],
            'ranking':[ids[i] for i in np.argsort(-mean,kind='stable')],'count_scope':'expected native next-movie occurrence count; may exceed1; not user satisfaction',
            'source_id':'model:calibrated-ranker:'+row['id'],'allowed_use':'prediction_basis','origin_case_id':'original_train_calibration_fit',
            'current_user_fact_authority':False,'semantic_truth_certified':False,'actual_lm_review_effect_estimated':False}


def training_plan(root=RESULT,data=DATA):
    root=Path(root);plan={'version':'native-calibrated-challenge-v3-native-occurrence-count','teacher_source':str(Path(data).resolve()),
        'teacher_fit':'original train80 by inner-retrieval-20261028 conv hash; scores only on disjoint hold20',
        'teacher_kbrd_epoch':'fixed9, no hold20 checkpoint selection','calibration_split':{'seed':OUTER_SEED,'diagnostic':'sha256(conv)%4==0','fit':'remaining convs'},
        'model_selection_split':{'seed':INNER_SEED,'select':'calibration-fit conv sha256%5==0','inner_fit':'remaining calibration-fit'},
        'model_grid':list(CONFIGS),'model_selection_key':'inner-select native micro Recall@10, negative Poisson deviance, negative count MSE; config order ties',
        'calibration':'conversation3fold OOF selected model -> nonnegative count isotonic on calibration-fit only; serving ensemble of3 cross-fitted models',
        'target':'native occurrence count, duplicates preserved, may exceed1; swap gain expected_count_b-expected_count_a aligns micro hit numerator; no serving future target count',
        'loss_weight':'each candidate weight1/pool_size (explicit feature), global mean1; count targets preserve all occurrences; selection evaluates exact native micro metrics',
        'candidate_pool':'fold80 Sparse top100 union fold80 KBRD top100, <=200 native movies',
        'features':list(FEATURES),'fulltrain_on_train_scores_forbidden':True,
        'public_valid_labels_used':False,'public_test_opened':False,'legacy_reserved_opened':False,
        'research_history_caveat':'teacher inner split/config had previous project development use; diagnostic is new calibrator held-conversation evaluation, not global untouched confirmation',
        'ranker_attribution':'any gain from calibrated ranker belongs to supervised predictor baseline, not multi-agent/game mechanism',
        'review_utility_scope':'ideal swap benefit/harm and model-uncertainty envelope only; no actual LM review responses in calibration data',
        'uncertainty_weight':.5,'source_sha256':sha(Path(__file__))}
    create(root/'training_plan.json',plan);return plan


def fit(root=RESULT,data=DATA):
    root=Path(root);training_plan(root,data)
    ids,items,rows,labels,arrays=load_fold(data);splits=split_indices(ids)
    create(root/'split_commitment.json',{name:{'ids':[ids[i] for i in idx],'conversation_count':len({conv(ids[i]) for i in idx})} for name,idx in splits.items()})
    table=make_table(ids,items,rows,labels,arrays)
    create(root/'data_audit.json',{'query_count':len(ids),'candidate_rows':len(table['X']),'native_items':len(items),
        'candidate_pool_no_gold_insertion':True,'feature_names':list(FEATURES),'teacher_source':str(data),
        'teacher_file_sha256':{str(p.relative_to(data)):sha(p) for p in [Path(data)/'sparse/complete.json',Path(data)/'kbrd/complete.json',Path(data)/'hold20.ids.json',Path(data)/'native_item_ids.json']},
        'public_valid_or_test_files_read':False,'same_conversation_partitions_disjoint':True,
        'item_target_denominator':sum(len(r['targets']) for r in labels),'prefixes_with_repeated_targets':sum(len(r['targets'])!=len(set(r['targets'])) for r in labels),
        'max_single_movie_occurrence_count':max(max(Counter(r['targets']).values()) for r in labels)})
    selection=[];best_config=None;best_key=None
    for config in CONFIGS:
        model=fit_estimator(table,splits['inner_fit'],config);prob=np.zeros(len(table['X']),dtype=float)
        ix=subset_rows(table,splits['select']);prob[ix]=raw_expected_count(model,table['X'][ix])
        metric=metrics_for(table,splits['select'],prob);selection.append({'config':config,'inner_select':metric})
        key=(metric['metrics']['recall@10'],-metric['mean_poisson_deviance'],-metric['count_mse'])
        if best_key is None or key>best_key:best_config,best_key=config,key
        print('selection',config,key,flush=True)
    create(root/'model_selection.json',selection);create(root/'selected_model.json',{'config':best_config,'diagnostic_used':False,'selection_sha256':sha(root/'model_selection.json')})
    oof=np.full(len(table['X']),np.nan,dtype=float);models=[];folds=[]
    for fold in range(3):
        held=np.array([i for i in splits['fit'] if bucket(ids[i],OOF_SEED,3)==fold],dtype=np.int32)
        train=np.array([i for i in splits['fit'] if bucket(ids[i],OOF_SEED,3)!=fold],dtype=np.int32)
        model=fit_estimator(table,train,best_config,seed=271028+fold);models.append(model)
        ix=subset_rows(table,held);oof[ix]=raw_expected_count(model,table['X'][ix]);folds.append({'fold':fold,'fit_ids':[ids[i] for i in train],'held_ids':[ids[i] for i in held]})
        print('oof',fold,'train',len(train),'held',len(held),flush=True)
    fit_rows=subset_rows(table,splits['fit'])
    if not np.isfinite(oof[fit_rows]).all():raise ValueError('OOF predictions missing')
    iso=IsotonicRegression(y_min=0,out_of_bounds='clip').fit(oof[fit_rows],table['count'][fit_rows],sample_weight=table['weights'][fit_rows])
    bundle={'models':models,'isotonic':iso,'feature_names':list(FEATURES),'selected_config':best_config,'source_sha256':sha(Path(__file__)),
        'fit_ids':[ids[i] for i in splits['fit']],'diagnostic_ids':[ids[i] for i in splits['diagnostic']],
        'target':'expected native occurrence count; not actual LM review efficacy'}
    joblib.dump(bundle,root/'calibrated_model.joblib');create(root/'oof_groups.json',folds)
    create(root/'fitting_complete.json',{'model_sha256':sha(root/'calibrated_model.joblib'),'selected_config':best_config,'diagnostic_metrics_not_read_for_selection':True})
    member=[]
    for model in models:member.append(np.maximum(0.,iso.predict(raw_expected_count(model,table['X']))))
    matrix=np.array(member);prob=matrix.mean(axis=0);baselines=baseline_ranking_scores(table,arrays,items)
    mass_model=fit_mass_baseline(table,splits['fit']);mass_p=mass_model['estimator'].predict_proba(mass_model['scaler'].transform(table['X']))[:,1]
    baselines['macro_mass_logistic']=mass_p
    joblib.dump(mass_model,root/'macro_mass_baseline.joblib')
    summary={'selected_config':best_config,'sets':{},'scope':'original-train fold20 diagnostic only, no public validation label tuning',
        'attribution':'supervised calibration/fusion gain is a predictor baseline, not game or multi-agent gain',
        'review_effect':'not estimated; no LM review responses used'}
    for name,idx in [('calibration_fit',splits['fit']),('diagnostic',splits['diagnostic'])]:
        summary['sets'][name]={'queries':len(idx),'conversations':len({conv(ids[i]) for i in idx}),
            'calibrated_ranker':metrics_for(table,idx,prob),
            'baselines':{model:metrics_for(table,idx,p,calibration=False) for model,p in baselines.items()},
            'interpretation':'fit rows are in-sample for2/3 ensemble members; only diagnostic is honest held-conversation evaluation' if name=='calibration_fit' else 'never used to select model/calibrator parameters'}
    create(root/'summary.json',summary)
    np.save(root/'hold20.expected_count.npy',prob);np.save(root/'hold20.member_expected_count.npy',matrix)
    np.save(root/'candidate_offsets.npy',table['offsets']);np.save(root/'candidate_ids.npy',table['candidate_ids'])
    create(root/'hold20.ids.json',ids)
    predictions=[]
    for q,cid in enumerate(ids):
        sl=slice(table['offsets'][q],table['offsets'][q+1]);order=np.argsort(-prob[sl],kind='stable');m=table['candidate_ids'][sl]
        predictions.append({'id':cid,'ranking':m[order].tolist(),'expected_target_count':prob[sl][order].tolist(),'model_stability_sd':matrix[:,sl].std(axis=0)[order].tolist()})
    create(root/'hold20.predictions.json',predictions)
    (root/'source_snapshot').mkdir(exist_ok=True)
    (root/'source_snapshot'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary,indent=2),flush=True);return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',default=str(DATA));parser.add_argument('--output',default=str(RESULT));parser.add_argument('--fit',action='store_true');args=parser.parse_args()
    if args.fit:
        import builtins,io
        from unittest.mock import patch
        output=Path(args.output).resolve();data=Path(args.data).resolve();opened=[]
        original_builtin,original_io=builtins.open,io.open
        def wrap(original):
            def guard(file,mode='r',*a,**kw):
                if isinstance(file,(str,Path)) and 'r' in mode:
                    path=Path(file).resolve()
                    if path.suffix=='.json' and '/experiments/' in str(path):
                        if not path.is_relative_to(data) and not path.is_relative_to(output):raise RuntimeError('non-fold experiment JSON forbidden during calibration fit: '+str(path))
                        opened.append(str(path))
                return original(file,mode,*a,**kw)
            return guard
        with patch('builtins.open',wrap(original_builtin)),patch('io.open',wrap(original_io)):
            fit(output,data)
        create(output/'runtime_access_audit.json',{'opened_experiment_json_paths':sorted(set(opened)),'public_valid_test_or_reserved_read_forbidden':True,'source_sha256':sha(Path(__file__))})
    else:print(json.dumps(training_plan(Path(args.output),Path(args.data)),indent=2))
