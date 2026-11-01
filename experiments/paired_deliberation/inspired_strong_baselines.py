"""INSPIRED train-only strong baselines, academic recipe and inner tuning separate.

Never opens the distribution ZIP or any test member. Native 1815 author items
and original noisy entity links are retained. Only train selects parameters.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='2'
import argparse,ast,gc,hashlib,json,math,random,sys,time
from pathlib import Path
from collections import OrderedDict
from types import SimpleNamespace
import numpy as np
import torch
from experiments.consensus_recommendation.common import ROOT
from experiments.paired_deliberation.native_retrieval_optimization import NativeRetriever,enhanced_contexts,fuse,configurations
from experiments.paired_deliberation.native_kbrd_reproduction import Graph,RecommendationOnly,full_step,MemoryGuard,save
from experiments.paired_deliberation.native_inspired_benchmark import contexts as native_contexts,evaluate as native_evaluate

DATA=ROOT/'experiments/data/raw/crslab_inspired_native/released'
RESULT=ROOT/'experiments/results/native_inspired_strong_v1'
SPLIT='inspired-inner-20261028:'
SEED=271028
TRANSFER={'neighbors50':1.,'entity_a1.0_b0.25':.5,'recent_mentions':1.}
EPOCH_GRID=(1,5,10,20,40)
ALLOW_VALID=False
ACCESSED=[]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(2**20),b''):h.update(part)
    return h.hexdigest()


def access_hook(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
    path=os.fsdecode(args[0]);mode=str(args[1])
    if path.endswith('test_data.json') or ('inspired' in path and path.endswith('.zip')) or 'reserved_test' in path:
        raise RuntimeError('INSPIRED untouched test/ZIP is forbidden')
    if 'crslab_inspired_native' in path and path.endswith('valid_data.json') and not ALLOW_VALID:
        raise RuntimeError('public validation is forbidden before training-internal selection')
    if 'crslab_inspired_native' in path and 'r' in mode:ACCESSED.append(path)


def rows(raw,entities,split):
    with_ids=[dict(conv,conv_id='inspired:'+split+':'+str(i).zfill(4)) for i,conv in enumerate(raw)]
    enhanced=enhanced_contexts(with_ids,entities)
    reference=native_contexts(raw,entities,split)
    assert [(r['id'],r['text'],r['targets']) for r in enhanced]==[(r['id'],r['text'],r['targets']) for r in reference]
    return enhanced


def fold(row):
    conv=row['id'].rsplit(':',1)[0]
    return int(hashlib.sha256((SPLIT+conv).encode()).hexdigest(),16)%5


def inspired_graph(entities,lines):
    """Author InspiredDataset graph: all relation types, both ends selfloops."""
    mapping={};edges=set()
    for line in lines:
        head,rel,tail=line.strip().split('\t');a=entities[head];b=entities[tail]
        triples=[(a,b,rel),(b,a,rel),(a,a,'SELF_LOOP')]
        if a!=b:triples.append((b,b,'SELF_LOOP'))
        for h,t,r in triples:
            if r not in mapping:mapping[r]=len(mapping)
            edges.add((h,t,mapping[r]))
    return max(entities.values())+1,len(mapping),sorted(edges,key=lambda x:(x[2],x[1],x[0])),mapping


def author_graph_parity(entities):
    source=ROOT/'docs/research/inspired_protocol_sources_2026-10-28/CRSLab/crslab/data/dataset/inspired/inspired.py'
    tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef));node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_entity_kg_process')
    scope={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
    with (DATA/'dbpedia_subkg.txt').open() as stream:
        author=scope['_entity_kg_process'](SimpleNamespace(entity2id=entities,id2entity={v:k for k,v in entities.items()},entity_kg=stream))
    with (DATA/'dbpedia_subkg.txt').open() as stream:n,r,edges,mapping=inspired_graph(entities,stream)
    assert r==author['n_relation'] and set(edges)==set(map(tuple,author['edge']))
    return Graph(n,r,edges),mapping


def sparse_run(train,inner_train,inner_dev,entities,items):
    folder=RESULT/'sparse'
    if (folder/'selection.json').exists():return json.loads((folder/'selection.json').read_text())
    model=NativeRetriever(items,entities).fit(inner_train);comp=model.components(inner_dev);results={};best=None;best_key=None
    for name,config in configurations().items():
        values=fuse(comp,config);order=np.argsort(-values,axis=1,kind='stable')[:,:100];ranks=[[model.items[j] for j in row] for row in order]
        metric=native_evaluate(inner_dev,ranks);key=(metric['metrics']['recall@10'],metric['candidate_recall@100'],metric['metrics']['ndcg@10'])
        results[name]={'config':config,'inner_dev':metric}
        if best_key is None or key>best_key:best_key=key;best=name
    save(folder/'search.json',results);selection={'name':best,'config':results[best]['config'],'rule':'train-only conversation holdout R10, coverage100, NDCG10',
        'transfer_fixed_config':TRANSFER,'inner_train':len(inner_train),'inner_dev':len(inner_dev)}
    save(folder/'selection.json',selection);print('SPARSE INNER SELECT',selection,flush=True);return selection


def score_kbrd(model,rows,items):
    predictions=[];loss=0;targets=0;model.eval();index=torch.tensor(items)
    with torch.no_grad():
        kg=model.kg_embedding()
        for start in range(0,len(rows),128):
            batch=rows[start:start+128];values=model.logits([r['context_entities'] for r in batch],kg);order=torch.topk(values[:,index],100,dim=1).indices
            for i,(row,rank) in enumerate(zip(batch,order)):
                predictions.append([items[j] for j in rank.tolist()]);n=len(row['targets']);targets+=n
                loss+=float(torch.logsumexp(values[i],0))*n-sum(float(values[i,t]) for t in row['targets'])
    metric=native_evaluate(rows,predictions);metric['full_entity_CE']=loss/targets;return metric


def fit_epochs(graph,train,epochs,folder,inner_dev=None,items=None,checkpoint_epochs=()):
    random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(2);model=RecommendationOnly(graph);optimizer=torch.optim.Adam(model.parameters(),lr=.003,foreach=False)
    samples=[{'context_entities':r['context_entities'],'item':t} for r in train for t in r['targets']];history=[];start_all=time.monotonic()
    for epoch in range(1,epochs+1):
        model.train();indices=list(range(len(samples)));random.shuffle(indices);total=0;started=time.monotonic()
        for i in range(0,len(indices),1024):
            batch=[samples[j] for j in indices[i:i+1024]];loss=full_step(model,optimizer,batch,micro=64);total+=loss*len(batch)
        row={'epoch':epoch,'train_loss':total/len(samples),'seconds_train':time.monotonic()-started,'logical_batch_size':1024}
        if inner_dev is not None and epoch in EPOCH_GRID:row['inner_dev']=score_kbrd(model,inner_dev,items)
        history.append(row);save(folder/'history.json',history)
        if epoch in checkpoint_epochs:
            torch.save({'model':model.state_dict(),'epoch':epoch,'seed':SEED,'model_name':'KBRD inspired rec-only mathematical port'},folder/('epoch'+str(epoch)+'.pt'))
        if epoch in EPOCH_GRID:print(str(folder.name),'epoch',epoch,row,flush=True)
    save(folder/'training_complete.json',{'epochs':epochs,'targets':len(samples),'updates':epochs*math.ceil(len(samples)/1024),'seconds':time.monotonic()-start_all})
    return model,history


def kbrd_select(graph,train,inner_train,inner_dev,items):
    folder=RESULT/'kbrd_inner'
    if (folder/'selection.json').exists():return json.loads((folder/'selection.json').read_text())
    model,history=fit_epochs(graph,inner_train,max(EPOCH_GRID),folder,inner_dev=inner_dev,items=items)
    candidates=[x for x in history if 'inner_dev' in x]
    best=max(candidates,key=lambda x:(x['inner_dev']['metrics']['recall@10'],x['inner_dev']['metrics']['ndcg@10'],-x['inner_dev']['full_entity_CE']))
    selection={'selected_epoch':best['epoch'],'grid':list(EPOCH_GRID),'selection_metric':'train-inner Recall@10, then NDCG10, then lower CE',
         'selected_metrics':best['inner_dev'],'public_valid_labels_used':False,'author_pure_epoch':1,'author_batch_size':1024,'lr':.003}
    save(folder/'selection.json',selection);del model;gc.collect();return selection


def native_rank_matrix(values,tie):
    permutation=tie[np.argsort(-values[:,tie],axis=1,kind='stable')];rank=np.empty_like(permutation,dtype=np.uint16)
    rank[np.arange(len(rank))[:,None],permutation]=np.arange(1,len(tie)+1,dtype=np.uint16);return rank


def export_sparse(train,valid,entities,items,selection):
    folder=RESULT/'sparse';model=NativeRetriever(items,entities).fit(train);components=model.components(valid);native_indices=np.array([model.index[m] for m in items]);tie=np.argsort(native_indices)
    summaries={}
    for version,config in [('fixed_redial_transfer',TRANSFER),('train_inner_tuned',selection['config'])]:
        values=fuse(components,config)[:,native_indices];ranks=native_rank_matrix(values,tie);dest=folder/version;dest.mkdir(parents=True,exist_ok=True)
        np.save(dest/'valid.native_scores.npy',values);np.save(dest/'valid.native_ranks.npy',ranks)
        order=np.argsort(ranks,axis=1,kind='stable')[:,:100];predictions=[{'id':r['id'],'ranking':[items[j] for j in ix],'scores':[float(values[i,j]) for j in ix]} for i,(r,ix) in enumerate(zip(valid,order))]
        save(dest/'valid.predictions.json',predictions);summaries[version]=native_evaluate(valid,[p['ranking'] for p in predictions]);save(dest/'summary.json',summaries[version])
    for name in TRANSFER:np.save(folder/('valid.'+name+'.npy'),components[name][:,native_indices])
    return summaries


def export_kbrd(graph,valid,items,epoch,checkpoint,folder):
    model=RecommendationOnly(graph);stored=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);model.load_state_dict(stored['model']);model.eval()
    values=np.empty((len(valid),len(items)),dtype=np.float32);logz=[];index=torch.tensor(items)
    with torch.no_grad():
        kg=model.kg_embedding()
        for start in range(0,len(valid),128):
            current=valid[start:start+128];scores=model.logits([r['context_entities'] for r in current],kg)
            values[start:start+len(current)]=scores[:,index].numpy();logz.extend(torch.logsumexp(scores,dim=1).tolist())
    rank=native_rank_matrix(values,np.arange(len(items)));folder.mkdir(parents=True,exist_ok=True)
    np.save(folder/'valid.native_logits.npy',values);np.save(folder/'valid.native_ranks.npy',rank);np.save(folder/'valid.full_entity_logsumexp.npy',np.array(logz,dtype=np.float32))
    order=np.argsort(rank,axis=1,kind='stable')[:,:100];predictions=[{'id':r['id'],'ranking':[items[j] for j in ix],'logits':[float(values[i,j]) for j in ix]} for i,(r,ix) in enumerate(zip(valid,order))]
    save(folder/'valid.predictions.json',predictions);metric=native_evaluate(valid,[p['ranking'] for p in predictions]);metric.update(epoch=epoch,checkpoint_sha256=sha(checkpoint));save(folder/'summary.json',metric)
    del model;gc.collect();return metric


def run(phase):
    global ALLOW_VALID
    sys.addaudithook(access_hook);RESULT.mkdir(parents=True,exist_ok=True);guard=MemoryGuard(RESULT,limit_gib=5).start()
    entities=json.loads((DATA/'entity2id.json').read_text());items=json.loads((DATA/'movie_ids.json').read_text());assert len(items)==1815
    train=rows(json.loads((DATA/'train_data.json').read_text()),entities,'train');inner_train=[r for r in train if fold(r)!=0];inner_dev=[r for r in train if fold(r)==0]
    assert {r['id'].rsplit(':',1)[0] for r in inner_train}.isdisjoint({r['id'].rsplit(':',1)[0] for r in inner_dev})
    save(RESULT/'protocol.json',{'version':'native-inspired-strong-v1','fit':'only train','selection':'only train conversation hash holdout',
       'inner_train_contexts':len(inner_train),'inner_dev_contexts':len(inner_dev),'train_contexts':len(train),'native_items':len(items),
       'native_link_noise':'all1815 author items retained without self-cleaning','test_and_archive_open_forbidden':True,
       'CPU_threads':2,'RSS_guard_GiB':5,'source_sha256':sha(Path(__file__)),'KBRD_author_recipe':{'epochs':1,'batch_size':1024,'lr':.003,'kg_emb_dim':128,'bases':8},
       'KBRD_tuned_epoch_grid':list(EPOCH_GRID),'Sparse_transfer_config':TRANSFER,'public_valid_for_selection':False})
    sparse_selection=sparse_run(train,inner_train,inner_dev,entities,items)
    if phase=='sparse':
        ALLOW_VALID=True;valid=rows(json.loads((DATA/'valid_data.json').read_text()),entities,'valid');summary=export_sparse(train,valid,entities,items,sparse_selection)
        save(RESULT/'sparse/versions_summary.json',summary)
    else:
        graph,mapping=author_graph_parity(entities);save(RESULT/'graph_audit.json',{'entities':graph.n,'relations':graph.relations,'edges':len(graph.src),'relation_mapping':mapping,'author_function_edge_set_parity':True})
        selected=kbrd_select(graph,train,inner_train,inner_dev,items)
        final=RESULT/'kbrd_fulltrain';epoch=selected['selected_epoch']
        if not(final/'training_complete.json').exists():model,history=fit_epochs(graph,train,max(epoch,1),final,checkpoint_epochs=tuple(set((1,epoch))));del model;gc.collect()
        # All choices and fitted weights are committed before public validation.
        save(RESULT/'training_selection_complete.json',{'sparse':sparse_selection,'kbrd':selected,'public_valid_labels_available':False})
        ALLOW_VALID=True;valid=rows(json.loads((DATA/'valid_data.json').read_text()),entities,'valid')
        summary={'sparse':export_sparse(train,valid,entities,items,sparse_selection),'kbrd':{}}
        for version,ep in [('academic_pure',1),('train_inner_tuned',epoch)]:summary['kbrd'][version]=export_kbrd(graph,valid,items,ep,final/('epoch'+str(ep)+'.pt'),RESULT/'kbrd'/version)
        save(RESULT/'summary.json',summary)
    if 'valid' in locals():
        save(RESULT/'valid.ids.json',[r['id'] for r in valid]);save(RESULT/'native_item_ids.json',items)
        # Serving inputs omit current/future target fields; labels remain separate.
        modeldir=RESULT/'sparse/train_inner_tuned';pred=json.loads((modeldir/'valid.predictions.json').read_text());names={v:k for k,v in entities.items()}
        inputs=[{k:v for k,v in r.items() if k!='targets'}|{'candidates':[{'movie_id':str(m),'title':names[m],'score':score} for m,score in zip(p['ranking'],p['scores'])]} for r,p in zip(valid,pred)]
        save(RESULT/'valid.inputs.json',inputs);save(RESULT/'valid.labels.json',[{'id':r['id'],'targets':r['targets']} for r in valid])
    guard.stop.set();save(RESULT/('access_audit_'+phase+'.json'),{'dataset_paths_read':sorted(set(ACCESSED)),'test_zip_or_member_opened':False,'monitor_peak_RSS_bytes':guard.peak})
    print('DONE',json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('sparse','all'),default='all');args=p.parse_args();run(args.phase)
