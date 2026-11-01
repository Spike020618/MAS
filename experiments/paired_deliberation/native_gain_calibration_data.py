"""Strict conversation-heldout train calibration predictions, no public valid/test.

Both Sparse and KBRD train only fold80. KBRD takes fixed epoch9 without using
hold20 labels for model selection. Labels are separate from serving inputs.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_key]='2'
import argparse,gc,hashlib,json,math,random,sys,time
from pathlib import Path
import numpy as np
import torch
from experiments.paired_deliberation.native_retrieval_optimization import NativeRetriever,enhanced_contexts,fuse
from experiments.paired_deliberation.native_kbrd_reproduction import (ROOT,DATA,Graph,RecommendationOnly,author_graph,full_step,MemoryGuard,save)

RESULT=ROOT/'experiments/results/native_gain_calibration_v1'
CONFIG={'neighbors50':1.,'entity_a1.0_b0.25':.5,'recent_mentions':1.}
SPLIT='inner-retrieval-20261028:'
OPENED=[]

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def audit_open(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
    path=os.fsdecode(args[0]);mode=args[1]
    if path.endswith(('valid_data.json','test_data.json')) or 'reserved_test' in path or '/data/dev.labels' in path:
        raise RuntimeError('forbidden public/legacy evaluation input access in fold calibration: '+path)
    if 'experiments/data/' in path and path.endswith(('.json','.jsonl')) and 'r' in str(mode):OPENED.append(path)


def fold(row):return int(hashlib.sha256((SPLIT+row['id'].split(':')[0]).encode()).hexdigest(),16)%5


def prepare(root):
    entities=json.loads((DATA/'entity2id.json').read_text());items=json.loads((DATA/'movie_ids.json').read_text())
    raw=json.loads((DATA/'train_data.json').read_text());rows=enhanced_contexts(raw,entities);del raw;gc.collect()
    fit=[r for r in rows if fold(r)!=0];hold=[r for r in rows if fold(r)==0]
    train_conversations={r['id'].split(':')[0] for r in fit};hold_conversations={r['id'].split(':')[0] for r in hold}
    assert train_conversations.isdisjoint(hold_conversations)
    inputs=[{k:v for k,v in r.items() if k!='targets'} for r in hold];labels=[{'id':r['id'],'targets':r['targets']} for r in hold]
    save(root/'hold20.inputs.json',inputs);save(root/'hold20.labels.json',labels);save(root/'hold20.ids.json',[r['id'] for r in hold]);save(root/'native_item_ids.json',items)
    save(root/'fold80.ids.json',[r['id'] for r in fit]);save(root/'fold80.conversations.json',sorted(train_conversations));save(root/'hold20.conversations.json',sorted(hold_conversations))
    manifest={'version':'native-train-fold-only-calibration-v1','split_rule':'sha256('+SPLIT+'conversation_id) modulo5 ==0 heldout, else fit',
       'fit_contexts':len(fit),'hold_contexts':len(hold),'fit_conversations':len(train_conversations),'hold_conversations':len(hold_conversations),
       'fit_targets':sum(len(r['targets']) for r in fit),'hold_targets':sum(len(r['targets']) for r in hold),
       'fit_hold_conversations_disjoint':True,'sparse_fixed_config':CONFIG,'kbrd_fixed_epoch':9,'kbrd_seed':271028,
       'selection_scope':'weights learned only fold80; no hold20 labels used for new checkpoint/hyperparameter selection',
       'research_history_caveat':'same inner split and Sparse config had earlier project development uses; labels are not globally pristine. New predictor weights are strictly out-of-conversation and calibrator must split this hold20 again.',
       'inputs_have_no_targets':all('targets' not in r for r in inputs),'native_item_count':len(items),
       'data_train_sha256':file_sha(DATA/'train_data.json'),'source_sha256':file_sha(Path(__file__)),
       'public_valid_test_opened':False,'legacy_reserved_opened':False,'matrix_rows':'hold20.ids.json','matrix_columns':'native_item_ids.json',
       'rank_convention':'uint16 one-based rank in native universe; lower is better'}
    save(root/'manifest.json',manifest);return fit,hold,entities,items


def write_matrix(path,array):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:np.save(f,array)
    tmp.replace(path)


def export_ranks(scores,native_items,tie_order):
    # Stable tie handling follows the trained predictor's popularity item order.
    inverse=np.empty_like(tie_order);inverse[tie_order]=np.arange(len(tie_order))
    permutation=np.argsort(-scores[:,tie_order],axis=1,kind='stable');rank=np.empty_like(permutation,dtype=np.uint16)
    rank[np.arange(len(rank))[:,None],tie_order[permutation]]=np.arange(1,len(tie_order)+1,dtype=np.uint16)
    return rank


def sparse_phase(root,fit,hold,entities,items):
    folder=root/'sparse'
    if (folder/'complete.json').exists():print('reuse completed Sparse fold',flush=True);return
    started=time.monotonic();model=NativeRetriever(items,entities).fit(fit);comp=model.components(hold);scores=fuse(comp,CONFIG)
    native_order=np.array([model.index[m] for m in items]);native_scores=scores[:,native_order]
    tie_order=np.argsort(np.array([model.index[m] for m in items]));ranks=export_ranks(native_scores,items,tie_order)
    write_matrix(folder/'hold20.native_scores.npy',native_scores);write_matrix(folder/'hold20.native_ranks.npy',ranks)
    for name in CONFIG:write_matrix(folder/('hold20.'+name+'.npy'),comp[name][:,native_order])
    save(folder/'metadata.json',{'fit_contexts':len(fit),'hold_contexts':len(hold),'fixed_config':CONFIG,
        'fit_id_hash':hashlib.sha256(json.dumps([r['id'] for r in fit]).encode()).hexdigest(),'native_popularity_tie_order':tie_order.tolist(),
        'labels_available_in_components':False,'matrix_dtype':'float32','rank_dtype':'uint16','score':'weighted per-prefix max-normalized component sum'})
    save(folder/'complete.json',{'seconds':time.monotonic()-started,'shape':list(native_scores.shape),'labels_used_to_select_config':False})
    print('SPARSE COMPLETE',time.monotonic()-started,flush=True)


def kbrd_phase(root,fit,hold,entities,items):
    folder=root/'kbrd'
    if (folder/'complete.json').exists():print('reuse completed KBRD fold',flush=True);return
    torch.set_num_threads(2);random.seed(271028);torch.manual_seed(271028)
    kg=json.loads((DATA/'dbpedia_subkg.json').read_text());n,r,edges,mapping=author_graph(entities,kg);del kg;graph=Graph(n,r,edges);del edges;gc.collect()
    model=RecommendationOnly(graph);optimizer=torch.optim.Adam(model.parameters(),lr=.003,foreach=False)
    samples=[{'id':row['id'],'context_entities':row['context_entities'],'item':target} for row in fit for target in row['targets']]
    protocol={'fit_contexts':len(fit),'fit_targets':len(samples),'hold_contexts':len(hold),'epochs':9,'checkpoint_selection':'fixed epoch9, no holdout evaluation or selection',
         'seed':271028,'optimizer':'Adam','lr':.003,'batch_size':4096,'microbatch_size':64,'classes':n,'relations':r,'edges':len(graph.src),
         'full_entity_softmax':True,'source_sha256':file_sha(Path(__file__)),'torch':str(torch.__version__)}
    save(folder/'protocol.json',protocol);history=[];started=time.monotonic()
    for epoch in range(9):
        order=list(range(len(samples)));random.shuffle(order);total=0;begin=time.monotonic()
        for step,start in enumerate(range(0,len(order),4096)):
            batch=[samples[i] for i in order[start:start+4096]];loss=full_step(model,optimizer,batch,micro=64);total+=loss*len(batch)
            print('KBRD fold epoch',epoch+1,'step',step+1,'loss',loss,flush=True)
        history.append({'epoch':epoch+1,'train_loss':total/len(samples),'seconds':time.monotonic()-begin,'holdout_evaluated':False});save(folder/'history.json',history)
    torch.save({'model':model.state_dict(),'epoch':9,'protocol':protocol},folder/'epoch9.pt');del optimizer;gc.collect();model.eval()
    # Serving receives only held prefix entities; no held target list enters forward.
    native_scores=np.lib.format.open_memmap(folder/'hold20.native_logits.tmp.npy',mode='w+',dtype=np.float32,shape=(len(hold),len(items)))
    log_z=[];item_index=torch.tensor(items)
    with torch.no_grad():
        embedding=model.kg_embedding()
        for start in range(0,len(hold),128):
            contexts=[row['context_entities'] for row in hold[start:start+128]];scores=model.logits(contexts,embedding)
            native_scores[start:start+len(contexts)]=scores[:,item_index].numpy();log_z.extend(torch.logsumexp(scores,dim=1).tolist())
    native_scores.flush();del native_scores;(folder/'hold20.native_logits.tmp.npy').replace(folder/'hold20.native_logits.npy')
    scores=np.load(folder/'hold20.native_logits.npy',mmap_mode='r');ranks=export_ranks(scores,items,np.arange(len(items)))
    write_matrix(folder/'hold20.native_ranks.npy',ranks);write_matrix(folder/'hold20.full_entity_logsumexp.npy',np.array(log_z,dtype=np.float32))
    save(folder/'metadata.json',{'fixed_epoch':9,'fit_conversation_disjoint':True,'fit_id_hash':hashlib.sha256(json.dumps([r['id'] for r in fit]).encode()).hexdigest(),
        'score':'raw logits for native movie entities; full_entity_logsumexp permits full-class softmax',
        'hold_labels_used_for_training_or_selection':False,'checkpoint_sha256':file_sha(folder/'epoch9.pt')})
    save(folder/'complete.json',{'seconds':time.monotonic()-started,'shape':list(scores.shape),'optimization_steps':len(history)*math.ceil(len(samples)/4096),'holdout_used_for_model_selection':False})
    print('KBRD COMPLETE',time.monotonic()-started,flush=True)


def run(phase):
    sys.addaudithook(audit_open);RESULT.mkdir(parents=True,exist_ok=True);guard=MemoryGuard(RESULT,limit_gib=5).start()
    fit,hold,entities,items=prepare(RESULT)
    if phase in ('sparse','all'):sparse_phase(RESULT,fit,hold,entities,items);gc.collect()
    if phase in ('kbrd','all'):kbrd_phase(RESULT,fit,hold,entities,items)
    guard.stop.set();save(RESULT/('access_audit_'+phase+'.json'),{'opened_dataset_json_paths':sorted(set(OPENED)),
        'public_valid_test_open_forbidden_by_runtime_hook':True,'source_sha256':file_sha(Path(__file__)),'monitor_peak_rss_bytes':guard.peak})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('sparse','kbrd','all'),default='all');args=p.parse_args();run(args.phase)
