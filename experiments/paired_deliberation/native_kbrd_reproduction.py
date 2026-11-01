"""Memory-bounded mathematical port of CRSLab KBRD recommendation only.

RGCN basis distribution and detached-KG microbatch backward are algebraically
exact (up to floating summation). Tested against official PyG RGCN and author
attention equations. No generator, no sampled softmax, no public test access.
"""
import argparse,copy,gc,hashlib,json,math,os,platform,random,resource,threading,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'experiments/data/raw/crslab_redial_native/released'
RESULT=ROOT/'experiments/results/native_kbrd_reproduction_v1'

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def native_samples(raw,entities):
    samples=[]
    for conv in raw:
        turns=[]
        for u in conv['dialog']:
            m=[entities[x] for x in u['movies'] if x in entities];e=[entities[x] for x in u['entity'] if x in entities]
            if turns and turns[-1]['role']==u['role']:turns[-1]['movies']+=m;turns[-1]['entities']+=e
            else:turns.append({'role':u['role'],'movies':m,'entities':e})
        seen=[];known=set()
        for i,t in enumerate(turns):
            if i and t['role']=='Recommender':
                for target in t['movies']:samples.append({'id':str(conv['conv_id'])+':'+str(i),'context_entities':list(seen),'item':target})
            for e in t['entities']+t['movies']:
                if e not in known:known.add(e);seen.append(e)
    return samples


def author_graph(entities,kg):
    """CRSLab selfloops185, undirected edges, count>1000 then deduplicate."""
    from collections import Counter
    n=max(entities.values())+1;raw=[]
    for h in range(n):
        if str(h) not in kg:continue
        raw.append((h,h,185))
        for relation,tail in kg[str(h)]:
            if h!=tail and relation!=185:raw.extend(((h,tail,relation),(tail,h,relation)))
    counts=Counter(r for h,t,r in raw);mapping={};edges=set()
    for h,t,r in raw:
        if counts[r]>1000:
            if r not in mapping:mapping[r]=len(mapping)
            edges.add((h,t,mapping[r]))
    return n,len(mapping),sorted(edges,key=lambda x:(x[2],x[1],x[0])),mapping


class Graph:
    def __init__(self,n,relations,edges):
        self.n=n;self.relations=relations;e=torch.tensor(edges,dtype=torch.long)
        self.src=e[:,0].contiguous();self.dst=e[:,1].contiguous();self.rel=e[:,2].contiguous()
        deg=torch.bincount(self.rel*n+self.dst,minlength=relations*n)
        self.norm=1/deg[self.rel*n+self.dst].to(torch.float64)
        linear=self.dst*n+self.src;unique,self.inverse=torch.unique(linear,sorted=True,return_inverse=True)
        self.indices=torch.stack((unique//n,unique%n));self.unique_edges=len(unique)
    def adj(self,coeff):
        values=coeff.new_zeros(self.unique_edges)
        values.index_add_(0,self.inverse,self.norm.to(coeff.dtype)*coeff[self.rel])
        return torch.sparse_coo_tensor(self.indices,values,(self.n,self.n),is_coalesced=True,check_invariants=True)


class _BasisRelational(torch.autograd.Function):
    @staticmethod
    def forward(ctx,bases,comp,graph):
        ctx.graph=graph;ctx.save_for_backward(bases,comp)
        out=bases.new_zeros((graph.n,bases.shape[-1]))
        for b in range(len(bases)):out.add_(torch.sparse.mm(graph.adj(comp[:,b]),bases[b]))
        return out
    @staticmethod
    def backward(ctx,grad):
        bases,comp=ctx.saved_tensors;g=ctx.graph
        grad_bases=torch.empty_like(bases);grad_comp=torch.zeros_like(comp);norm=g.norm.to(bases.dtype)
        for b in range(len(bases)):
            grad_bases[b]=torch.sparse.mm(g.adj(comp[:,b]).transpose(0,1),grad)
            for start in range(0,len(g.src),8192):
                end=start+8192;dot=(bases[b,g.src[start:end]]*grad[g.dst[start:end]]).sum(dim=1)*norm[start:end]
                grad_comp[:,b].index_add_(0,g.rel[start:end],dot)
        return grad_bases,grad_comp,None


class RecommendationOnly(nn.Module):
    def __init__(self,graph,dim=128,bases=8):
        super().__init__();self.graph=graph;self.dim=dim
        self.bases=nn.Parameter(torch.empty(bases,graph.n,dim));self.comp=nn.Parameter(torch.empty(graph.relations,bases))
        self.root=nn.Parameter(torch.empty(graph.n,dim));self.kg_bias=nn.Parameter(torch.zeros(dim))
        for p in (self.bases,self.comp,self.root):nn.init.uniform_(p,-math.sqrt(6/(p.shape[-2]+p.shape[-1])),math.sqrt(6/(p.shape[-2]+p.shape[-1])))
        self.attn_a=nn.Parameter(torch.empty(dim,dim));self.attn_b=nn.Parameter(torch.empty(dim,1))
        nn.init.xavier_uniform_(self.attn_a,gain=1.414);nn.init.xavier_uniform_(self.attn_b,gain=1.414)
        # rec_bias.weight is unused by author recommend; only its bias is needed.
        self.rec_bias=nn.Parameter(torch.empty(graph.n));nn.init.uniform_(self.rec_bias,-1/math.sqrt(dim),1/math.sqrt(dim))
    def kg_embedding(self):return _BasisRelational.apply(self.bases,self.comp,self.graph)+self.root+self.kg_bias
    def encode_user(self,contexts,kg):
        length=max(max((len(c) for c in contexts),default=0),1)
        indices=torch.zeros((len(contexts),length),dtype=torch.long);mask=torch.zeros_like(indices,dtype=torch.bool)
        for i,c in enumerate(contexts):
            if c:indices[i,:len(c)]=torch.tensor(c);mask[i,:len(c)]=True
        h=kg[indices];energy=(torch.tanh(h@self.attn_a)@self.attn_b).squeeze(-1)
        energy=energy.masked_fill(~mask,float('-inf'));energy=torch.where(mask.any(dim=1,keepdim=True),energy,torch.zeros_like(energy))
        attn=F.softmax(energy,dim=1).masked_fill(~mask,0)
        return (attn.unsqueeze(-1)*h).sum(dim=1)
    def logits(self,contexts,kg):return F.linear(self.encode_user(contexts,kg),kg,self.rec_bias)


def full_step(model,optimizer,batch,micro=64):
    optimizer.zero_grad(set_to_none=True);kg_graph=model.kg_embedding();leaf=kg_graph.detach().requires_grad_(True);total=0.
    for start in range(0,len(batch),micro):
        rows=batch[start:start+micro];logits=model.logits([r['context_entities'] for r in rows],leaf)
        labels=torch.tensor([r['item'] for r in rows]);loss=F.cross_entropy(logits,labels,reduction='sum')/len(batch)
        total+=float(loss.detach());loss.backward();del loss,logits
    kg_graph.backward(leaf.grad);optimizer.step();return total


def _toy_parity():
    """Independent official PyG module forward AND all relevant parameter grads."""
    from torch_geometric.nn import RGCNConv
    import torch_geometric.backend
    torch_geometric.backend.use_segment_matmul=False
    torch.manual_seed(721);g=Graph(7,3,[(i,i,0) for i in range(7)]+[(0,1,1),(2,1,1),(3,4,1),(0,1,2),(4,3,2),(5,6,2)])
    model=RecommendationOnly(g,dim=4,bases=2).double()
    ref=RGCNConv(7,4,3,num_bases=2).double()
    with torch.no_grad():
        ref.weight.copy_(model.bases);ref.comp.copy_(model.comp);ref.root.copy_(model.root);ref.bias.copy_(model.kg_bias)
    expected=ref(None,torch.stack((g.src,g.dst)),g.rel);actual=model.kg_embedding()
    forward=float((expected-actual).abs().max().detach())
    w=torch.randn_like(expected);(expected*w).sum().backward();(actual*w).sum().backward()
    errors={name:float((getattr(model,name).grad-getattr(ref,other).grad).abs().max()) for name,other in [('bases','weight'),('comp','comp'),('root','root'),('kg_bias','bias')]}
    assert forward<1e-12 and max(errors.values())<1e-11,(forward,errors)
    # Compare full-batch direct autograd with detached-KG microbatch accumulation.
    direct=copy.deepcopy(model);chunked=copy.deepcopy(model);contexts=[[],[0,1],[2,3,4],[1],[6,0]]
    batch=[{'context_entities':c,'item':t} for c,t in zip(contexts,[1,2,3,4,5])]
    oa=torch.optim.Adam(direct.parameters(),lr=.003);ob=torch.optim.Adam(chunked.parameters(),lr=.003)
    oa.zero_grad(set_to_none=True);kg=direct.kg_embedding()
    logits=direct.logits(contexts,kg);loss=F.cross_entropy(logits,torch.tensor([r['item'] for r in batch]));loss.backward();oa.step()
    chunk_loss=full_step(chunked,ob,batch,micro=2)
    gradients={name:float((a.grad-b.grad).abs().max()) for (name,a),(other,b) in zip(direct.named_parameters(),chunked.named_parameters())}
    updates={name:float((a-b).abs().max().detach()) for (name,a),(other,b) in zip(direct.named_parameters(),chunked.named_parameters())}
    assert abs(float(loss.detach())-chunk_loss)<1e-12 and max(gradients.values())<1e-11 and max(updates.values())<1e-10
    # Author SelfAttentionBatch is per-list softmax; empty lists return zero.
    h=torch.randn(7,4,dtype=torch.double);batched=direct.encode_user(contexts,h);references=[]
    for c in contexts:
        subset=h[c];a=torch.softmax((torch.tanh(subset@direct.attn_a)@direct.attn_b).squeeze(1),dim=0);references.append(a@subset)
    attention=float((batched-torch.stack(references)).abs().max().detach());assert attention<1e-12
    return {'official_pyg_version':__import__('torch_geometric').__version__,'rgcn_forward_max_abs':forward,'rgcn_gradient_max_abs':errors,
        'attention_forward_max_abs':attention,'full_vs_micro_loss_abs':abs(float(loss.detach())-chunk_loss),'full_vs_micro_gradient_max_abs':gradients,
        'adam_update_max_abs':updates,'dtype':'float64','pass':True}


def toy_parity():
    # Parity fixtures must not perturb the declared training initialization.
    with torch.random.fork_rng(devices=[]):return _toy_parity()


class MemoryGuard:
    def __init__(self,root,limit_gib=5):self.root=root;self.limit=limit_gib*2**30;self.peak=0;self.stop=threading.Event()
    def start(self):
        import psutil
        def watch():
            p=psutil.Process()
            while not self.stop.wait(.05):
                rss=p.memory_info().rss;self.peak=max(self.peak,rss)
                if rss>self.limit:
                    save(self.root/'memory_guard_abort.json',{'rss_bytes':rss,'limit_bytes':self.limit});os._exit(98)
        threading.Thread(target=watch,daemon=True).start();return self


def load_native():
    entities=json.loads((DATA/'entity2id.json').read_text());items=json.loads((DATA/'movie_ids.json').read_text())
    kg=json.loads((DATA/'dbpedia_subkg.json').read_text());n,r,edges,mapping=author_graph(entities,kg);del kg
    graph=Graph(n,r,edges);del edges
    datasets={}
    for split in ('train','valid'):
        raw=json.loads((DATA/(split+'_data.json')).read_text());datasets[split]=native_samples(raw,entities);del raw;gc.collect()
    return graph,items,datasets,mapping


def evaluate(model,data,items,micro=128):
    model.eval();totals={k:0. for k in ('recall@1','recall@10','recall@50','recall@100','ndcg@10','mrr@10')};loss=0.;preds=[]
    with torch.no_grad():
        kg=model.kg_embedding();item_index=torch.tensor(items)
        for start in range(0,len(data),micro):
            rows=data[start:start+micro];scores=model.logits([r['context_entities'] for r in rows],kg);labels=torch.tensor([r['item'] for r in rows])
            loss+=float(F.cross_entropy(scores,labels,reduction='sum'));rank=torch.topk(scores[:,item_index],100,dim=1).indices
            mapped=item_index[rank]
            for row,pred in zip(rows,mapped.tolist()):
                target=row['item'];position=pred.index(target) if target in pred else None
                for k in (1,10,50,100):totals['recall@'+str(k)]+=position is not None and position<k
                if position is not None and position<10:totals['ndcg@10']+=1/math.log2(position+2);totals['mrr@10']+=1/(position+1)
                preds.append({'id':row['id'],'target':target,'ranking':pred})
    return {'targets':len(data),'full_entity_ce_loss':loss/len(data),'metrics':{k:v/len(data) for k,v in totals.items()}},preds


def run(probe=True,epochs=10,micro=64,seed=271028,root=RESULT,selection_metric="valid_ce"):
    root.mkdir(parents=True,exist_ok=True);guard=MemoryGuard(root).start();torch.set_num_threads(2);random.seed(seed);torch.manual_seed(seed)
    parity=toy_parity();save(root/'toy_parity.json',parity);print('parity passed',flush=True)
    graph,items,data,mapping=load_native();gc.collect();model=RecommendationOnly(graph);optimizer=torch.optim.Adam(model.parameters(),lr=.003,foreach=False)
    parameters=sum(p.numel() for p in model.parameters());protocol={'source_commit':'6497938919998b9cd7aa9c2f43effb39ad0c7088','torch':torch.__version__,
       'type':'CRSLab KBRD rec-only mathematical port; no generator or unused rec_bias.weight','device':'CPU','threads':2,
       'native_entities':graph.n,'native_relations':graph.relations,'native_edges':len(graph.src),'relation_mapping':mapping,'train_targets':len(data['train']),
       'valid_targets':len(data['valid']),'selection_metric':selection_metric,'parameters':parameters,'parameter_bytes':parameters*4,'config':{'kg_emb_dim':128,'num_bases':8,'batch_size':4096,'microbatch':micro,'epochs':epochs,'Adam_lr':.003,'seed':seed},
       'loss':'full64363-entity mean cross entropy; microbatch sum divided by actual logical batch size; one optimizer update per4096',
       'differences':['unused token/generator/rec_bias.weight parameters omitted, so RNG draw sequence differs from full author model',
          'basis/relations distributed with exact custom backward to avoid expanded relation tensor; floating summation order differs',
          'validation checkpoint selection only; no public test access'],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(root/'protocol.json',protocol);save(root/'environment.json',{'platform':platform.platform(),'python':platform.python_version()})
    order=list(range(len(data['train'])));random.shuffle(order);batch=[data['train'][i] for i in order[:4096]]
    started=time.monotonic();loss=full_step(model,optimizer,batch,micro);elapsed=time.monotonic()-started
    measured={'logical_batch':len(batch),'full_entity_classes':graph.n,'loss':loss,'seconds':elapsed,'monitor_peak_rss_bytes':guard.peak,
       'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'darwin_rusage_units':'bytes','estimated70_step_minutes_excluding_validation':elapsed*70/60}
    save(root/'real_step_probe.json',measured);print('PROBE',json.dumps(measured),flush=True)
    if probe:guard.stop.set();return measured
    # Start training from a fresh initialization so the probe does not add a hidden update.
    del model,optimizer;gc.collect();torch.manual_seed(seed);random.seed(seed);model=RecommendationOnly(graph);optimizer=torch.optim.Adam(model.parameters(),lr=.003,foreach=False)
    history=[];best_loss=float('inf');best_key=None;selected_epoch=None
    for epoch in range(epochs):
        model.train();order=list(range(len(data['train'])));random.shuffle(order);total=0.;start=time.monotonic()
        for step,offset in enumerate(range(0,len(order),4096)):
            batch=[data['train'][i] for i in order[offset:offset+4096]];loss=full_step(model,optimizer,batch,micro);total+=loss*len(batch)
            print('epoch',epoch+1,'step',step+1,'loss',loss,'peak_gib',round(guard.peak/2**30,3),flush=True)
        metrics,predictions=evaluate(model,data['valid'],items,micro=max(micro,128));row={'epoch':epoch+1,'train_loss':total/len(order),'valid':metrics,'seconds':time.monotonic()-start,'peak_rss_bytes':guard.peak};history.append(row)
        save(root/'history.json',history)
        key=(-metrics['full_entity_ce_loss'],) if selection_metric=='valid_ce' else (metrics['metrics']['recall@10'],metrics['metrics']['ndcg@10'],-metrics['full_entity_ce_loss'])
        best_loss=min(best_loss,metrics['full_entity_ce_loss'])
        if best_key is None or key>best_key:
            best_key=key;selected_epoch=epoch+1;torch.save({'model':model.state_dict(),'epoch':epoch+1,'valid':metrics,'protocol':protocol},root/'best.pt');save(root/'best.valid.predictions.json',predictions)
        print('EPOCH',json.dumps(row),flush=True)
    guard.stop.set();save(root/'training_complete.json',{'epochs':len(history),'best_valid_loss':best_loss,'selected_epoch':selected_epoch,'selection_metric':selection_metric,'max_rss_bytes':guard.peak,'public_test_opened':False});return history

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',action='store_true');p.add_argument('--epochs',type=int,default=10);p.add_argument('--microbatch',type=int,default=64);p.add_argument('--root',type=Path,default=RESULT);p.add_argument('--selection',choices=('valid_ce','valid_recall10'),default='valid_ce');a=p.parse_args();run(not a.train,a.epochs,a.microbatch,root=a.root,selection_metric=a.selection)
