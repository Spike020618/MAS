"""Locked small-dev weighted RRF of native Sparse and trained KBRD predictors.

The selection prefix contains eight existing development inputs. Remaining old
and new validation labels are read ONLY after the parameter commitment and
label-free shared union materials have been written. This is predictor fusion,
not evidence for a multi-agent/game advantage or untouched confirmation.
"""
import json
from pathlib import Path
import numpy as np
from experiments.consensus_recommendation.common import ROOT,create,sha,fingerprint
from experiments.paired_deliberation.native_retrieval_optimization import NativeRetriever,enhanced_contexts,fuse
from experiments.paired_deliberation.native_redial_benchmark import DATA,evaluate
from experiments.paired_deliberation.native_kbrd_scores import PrecomputedKBRD

SPARSE=ROOT/'experiments/results/native_retrieval_inner_v1'
KBRD=ROOT/'experiments/results/native_kbrd_reproduction_v1/train_recall_r1/export'
RESULT=ROOT/'experiments/results/native_predictor_fusion_v1'
ALPHAS=(0.,.1,.25,.5,.75,.9,1.)
CONSTANTS=(1,5,10,20,60,100)


def read_array_prefix(path,count):
    """Parse exactly count JSON objects; do not inspect later array entries."""
    result=[]
    with Path(path).open(encoding='utf-8') as stream:
        c=stream.read(1)
        while c and c.isspace():c=stream.read(1)
        if c!='[':raise ValueError('JSON array required')
        for _ in range(count):
            c=stream.read(1)
            while c and (c.isspace() or c==','):c=stream.read(1)
            if c!='{':raise ValueError('not enough object entries')
            chars=[c];depth=1;quoted=False;escaped=False
            while depth:
                c=stream.read(1)
                if not c:raise ValueError('truncated JSON entry')
                chars.append(c)
                if quoted:
                    if escaped:escaped=False
                    elif c=='\\':escaped=True
                    elif c=='"':quoted=False
                elif c=='"':quoted=True
                elif c=='{':depth+=1
                elif c=='}':depth-=1
            result.append(json.loads(''.join(chars)))
    return result


def rank_positions(order):
    order=np.asarray(order);positions=np.empty_like(order,dtype=np.int32)
    positions[np.arange(len(order))[:,None],order]=np.arange(1,order.shape[1]+1,dtype=np.int32)
    return positions


def rrf_scores(sparse_ranks,kbrd_ranks,alpha,k):
    if not 0<=alpha<=1 or k<=0:raise ValueError('RRF weight [0,1], positive k')
    a=np.asarray(sparse_ranks);b=np.asarray(kbrd_ranks)
    if a.shape!=b.shape or np.any(a<=0) or np.any(b<=0):raise ValueError('matching positive full-universe ranks')
    return alpha/(k+a)+(1-alpha)/(k+b)


def rrf_order(sparse_ranks,kbrd_ranks,alpha,k):
    values=rrf_scores(sparse_ranks,kbrd_ranks,alpha,k)
    return np.lexsort((np.asarray(sparse_ranks),-values),axis=1)


def complementary(labels,sparse,kbrd):
    result={'contexts':len(labels),'item_targets':sum(len(r['targets']) for r in labels)}
    for k in (10,100):
        counts={'both':0,'sparse_only':0,'kbrd_only':0,'neither':0}
        for row,left,right in zip(labels,sparse,kbrd):
            a=set(left[:k]);b=set(right[:k])
            for target in row['targets']:
                counts['both' if target in a and target in b else 'sparse_only' if target in a else 'kbrd_only' if target in b else 'neither']+=1
        counts['union_covered']=counts['both']+counts['sparse_only']+counts['kbrd_only']
        counts['union_recall']=counts['union_covered']/result['item_targets']
        result['top'+str(k)]=counts
    return result


def score_card(name,case_id,movie_ids,columns,rows,provenance):
    return {'source_id':'predictor:'+name+':'+case_id,'kind':'model_prediction','allowed_use':'prediction_basis',
        'origin_case_id':'released_train_fitted_predictor','query_case_id':case_id,'movie_ids':list(map(str,movie_ids)),
        'text':json.dumps({'predictor':name,'query_case_id':case_id,'columns':columns,'rows':rows,
             'interpretation':'train-fitted prediction scores, not current-user facts or semantic truth'},ensure_ascii=False,separators=(',',':')),
        'columns':columns,'rows':rows,'provenance':provenance,'current_user_fact_authority':False,
        'semantic_truth_certified':False}


class PrecomputedPredictorFusion:
    """No labels/model training: exported full native score access for these IDs."""
    def __init__(self,root=RESULT):
        self.root=Path(root);self.ids=json.loads((self.root/'case_ids.json').read_text());self.items=json.loads((self.root/'native_item_ids.json').read_text())
        self.row_index={c:i for i,c in enumerate(self.ids)};self.item_index={str(m):j for j,m in enumerate(self.items)}
        self.sparse=np.load(self.root/'sparse.native_scores.npy',mmap_mode='r');self.sparse_rank=np.load(self.root/'sparse.native_ranks.npy',mmap_mode='r')
        self.fusion=np.load(self.root/'fusion.native_scores.npy',mmap_mode='r');self.fusion_rank=np.load(self.root/'fusion.native_ranks.npy',mmap_mode='r')
        self.kbrd=PrecomputedKBRD(KBRD)
    def score_pool(self,case_id,candidate_ids):
        i=self.row_index[case_id];ids=list(map(str,candidate_ids));j=[self.item_index[m] for m in ids]
        return {'case_id':case_id,'columns':['movie_id','sparse_score','sparse_native_rank','rrf_score','rrf_native_rank'],
            'rows':[[m,float(self.sparse[i,k]),int(self.sparse_rank[i,k]),float(self.fusion[i,k]),int(self.fusion_rank[i,k])] for m,k in zip(ids,j)],
            'kbrd':self.kbrd.score_pool(case_id,ids),'current_user_fact_authority':False,'semantic_truth_certified':False}


def run(root=RESULT):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    old=json.loads((SPARSE/'old32.inputs.json').read_text());new=json.loads((SPARSE/'new64.inputs.json').read_text());inputs=old+new
    ids=[r['id'] for r in inputs]
    if len(ids)!=len(set(ids)):raise ValueError('query sets overlap')
    config=json.loads((SPARSE/'selected_config.json').read_text())
    kbrd=PrecomputedKBRD(KBRD)
    plan={'version':'native-predictor-fusion-v1','native_items':len(kbrd.items),'selection_ids':ids[:8],
        'old24_ids':ids[8:32],'new64_ids':ids[32:],'alpha_sparse_grid':list(ALPHAS),'rrf_k_grid':list(CONSTANTS),
        'selection_key':'selection8 Recall@10, NDCG@10, Recall@100; ties nearest alpha=.5 then k=60 then smaller k',
        'rrf_formula':'alpha/(k+rank_sparse)+(1-alpha)/(k+rank_kbrd); ranks 1-based on all native6924 items',
        'ties':'ascending Sparse native rank','candidate_policy':'shared union of each predictor top100, <=200; no target insertion',
        'source_sha256':sha(Path(__file__)),'sparse_config_sha256':sha(SPARSE/'selected_config.json'),
        'kbrd_serving_metadata':kbrd.meta,'inputs_sha256':{name:sha(SPARSE/name) for name in ('old32.inputs.json','new64.inputs.json')},
        'scope':'predictor fusion development; base KBRD checkpoint selected on public validation, not pristine confirmation',
        'no_public_test_opened':True,'no_legacy_reserved_opened':True,'holdout_labels_not_used_for_fusion_selection':True}
    create(root/'protocol.json',plan)
    p=DATA/'released';entities=json.loads((p/'entity2id.json').read_text());items=json.loads((p/'movie_ids.json').read_text())
    train=enhanced_contexts(json.loads((p/'train_data.json').read_text()),entities)
    model=NativeRetriever(items,entities).fit(train);components=model.components(inputs);sparse_scores=fuse(components,config['config'])
    sparse_order=np.argsort(-sparse_scores,axis=1,kind='stable');sparse_rank=rank_positions(sparse_order)
    item_array=np.asarray(model.items,dtype=np.int32);gather=np.array([kbrd.item_index[str(m)] for m in model.items])
    kbrd_rank=np.empty_like(sparse_rank)
    for i,cid in enumerate(ids):
        logits=np.asarray(kbrd.matrix[kbrd.row_index[cid]]);order=np.argsort(-logits,kind='stable');positions=np.empty(len(order),dtype=np.int32);positions[order]=np.arange(1,len(order)+1)
        kbrd_rank[i]=positions[gather]
    kbrd_order=np.argsort(kbrd_rank,axis=1,kind='stable')
    def ranks(order):return item_array[order[:,:100]].tolist()
    sparse_predictions=ranks(sparse_order);kbrd_predictions=ranks(kbrd_order)
    # Only these eight objects are parsed before locking the chosen parameters.
    selection_labels=read_array_prefix(SPARSE/'old32.labels.json',8)
    if [r['id'] for r in selection_labels]!=ids[:8]:raise ValueError('selection label alignment')
    create(root/'selection8.labels.json',selection_labels)
    grid=[];best=None;best_key=None
    for alpha in ALPHAS:
        for k in CONSTANTS:
            order=rrf_order(sparse_rank[:8],kbrd_rank[:8],alpha,k);metrics=evaluate(selection_labels,ranks(order))
            entry={'alpha_sparse':alpha,'k':k,'selection8':metrics};grid.append(entry)
            key=(metrics['metrics']['recall@10'],metrics['metrics']['ndcg@10'],metrics['retrieval_recall@100'],-abs(alpha-.5),-abs(k-60),-k)
            if best_key is None or key>best_key:best,best_key=entry,key
    create(root/'grid_selection8.json',grid)
    selected={'alpha_sparse':best['alpha_sparse'],'k':best['k'],'selection8':best['selection8'],
        'selection_key':plan['selection_key'],'grid_sha256':sha(root/'grid_selection8.json'),
        'old24_or_new64_labels_loaded':False,'scope':'locked before remaining labels; never reselect on later results'}
    create(root/'selected_params.json',selected)
    alpha,k=selected['alpha_sparse'],selected['k'];fusion_values=rrf_scores(sparse_rank,kbrd_rank,alpha,k)
    fusion_order=rrf_order(sparse_rank,kbrd_rank,alpha,k);fusion_rank=rank_positions(fusion_order);fusion_predictions=ranks(fusion_order)
    for name,array in [('sparse.native_scores.npy',sparse_scores),('sparse.native_ranks.npy',sparse_rank),('kbrd.native_ranks.npy',kbrd_rank),('fusion.native_scores.npy',fusion_values),('fusion.native_ranks.npy',fusion_rank)]:
        np.save(root/name,array)
    create(root/'case_ids.json',ids);create(root/'native_item_ids.json',model.items)
    union_predictions=[];serving=[]
    sparse_provenance={'fit_split':'train','configuration':config['config'],'config_sha256':sha(SPARSE/'selected_config.json')}
    kbrd_provenance={'fit_split':'train','checkpoint_sha256':kbrd.meta['checkpoint_sha256'],'selected_epoch':kbrd.meta['selected_epoch'],'selection_metric':kbrd.meta['selection_metric']}
    for i,row in enumerate(inputs):
        universe=set(sparse_predictions[i])|set(kbrd_predictions[i]);union=[int(m) for m in item_array[fusion_order[i]] if int(m) in universe]
        if not 100<=len(union)<=200 or not universe<=set(model.items):raise ValueError('native predictor union')
        union_predictions.append(union)
        positions=[model.index[m] for m in union];kb=kbrd.score_pool(row['id'],union)
        sparse_rows=[[str(m),round(float(sparse_scores[i,j]),8),int(sparse_rank[i,j])] for m,j in zip(union,positions)]
        kbrd_rows=[[r[0],round(r[1],8),round(r[2],8),r[3]] for r in kb['rows']]
        cards=[score_card('sparse',row['id'],union,['movie_id','score','native_rank'],sparse_rows,sparse_provenance),
               score_card('kbrd',row['id'],union,kb['columns'],kbrd_rows,kbrd_provenance)]
        candidates=[{'movie_id':str(m),'title':model.names[m],'score':float(fusion_values[i,j]),'sparse_score':float(sparse_scores[i,j]),
            'sparse_rank':int(sparse_rank[i,j]),'kbrd_rank':int(kbrd_rank[i,j]),'rrf_native_rank':int(fusion_rank[i,j])} for m,j in zip(union,positions)]
        serving.append({key:value for key,value in row.items() if key not in ('candidates','targets')}|{'candidates':candidates,'sources':{c['source_id']:c for c in cards},
            'candidate_policy':'Sparse top100 union KBRD top100, order fixed weighted RRF; no labels','current_user_fact_authority':False})
    # All serving material is sealed before parsing holdout labels below.
    for name,sl in [('old32',slice(0,32)),('old24',slice(8,32)),('new64',slice(32,None))]:create(root/(name+'.union.inputs.json'),serving[sl])
    create(root/'serving_generation_complete.json',{'parameters_sha256':sha(root/'selected_params.json'),
        'new64_inputs_sha256':sha(root/'new64.union.inputs.json'),'old24_or_new64_labels_loaded':False})
    # Evaluation only: no parameter or serving-material changes are allowed here.
    old_labels=json.loads((SPARSE/'old32.labels.json').read_text());new_labels=json.loads((SPARSE/'new64.labels.json').read_text());labels=old_labels+new_labels
    if [r['id'] for r in labels]!=ids:raise ValueError('evaluation label alignment')
    predictions={'sparse':sparse_predictions,'kbrd':kbrd_predictions,'rrf_native':fusion_predictions,'rrf_union':union_predictions}
    summary={'parameters':selected,'native_items':len(items),'scope':plan['scope'],'sets':{},
        'new64_union_size':{'min':min(len(s['candidates']) for s in serving[32:]),'max':max(len(s['candidates']) for s in serving[32:]),'mean':float(np.mean([len(s['candidates']) for s in serving[32:]]))},
        'attribution':'any gain is predictor fusion, NOT multi-agent/game benefit'}
    for name,sl in [('selection8',slice(0,8)),('old24',slice(8,32)),('new64',slice(32,None)),('old32',slice(0,32))]:
        chosen_labels=labels[sl]
        summary['sets'][name]={'metrics':{method:evaluate(chosen_labels,pred[sl]) for method,pred in predictions.items()},
            'complementarity':complementary(chosen_labels,sparse_predictions[sl],kbrd_predictions[sl])}
    create(root/'summary.json',summary)
    create(root/'native.predictions.json',[{'id':cid,**{m:pred[i] for m,pred in predictions.items()}} for i,cid in enumerate(ids)])
    (root/'source_snapshot').mkdir(exist_ok=True)
    for name in ('native_predictor_fusion.py','native_kbrd_scores.py','native_retrieval_optimization.py'):(root/'source_snapshot'/name).write_bytes((ROOT/'experiments/paired_deliberation'/name).read_bytes())
    print(json.dumps(summary,ensure_ascii=False,indent=2));return summary

if __name__=='__main__':run()
