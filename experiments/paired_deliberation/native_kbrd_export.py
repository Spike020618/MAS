"""Export rec-only KBRD checkpoints for native validation; no test access."""
import argparse,gc,hashlib,json
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from experiments.paired_deliberation.native_kbrd_reproduction import (ROOT,DATA,RESULT,author_graph,Graph,RecommendationOnly,native_samples,save)
from experiments.paired_deliberation.native_kbrd_scores import PrecomputedKBRD
from experiments.paired_deliberation.native_redial_benchmark import evaluate


def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(2**20),b''):h.update(b)
    return h.hexdigest()


class KBRDServing:
    def __init__(self,checkpoint):
        torch.set_num_threads(2);entities=json.loads((DATA/'entity2id.json').read_text());kg=json.loads((DATA/'dbpedia_subkg.json').read_text())
        n,r,edges,mapping=author_graph(entities,kg);del kg
        self.graph=Graph(n,r,edges);del edges
        self.items=json.loads((DATA/'movie_ids.json').read_text());self.item_index=torch.tensor(self.items)
        self.names={v:k.removeprefix('<http://dbpedia.org/resource/').removesuffix('>').replace('_',' ') for k,v in entities.items()}
        self.checkpoint=Path(checkpoint);stored=torch.load(self.checkpoint,map_location='cpu',weights_only=False,mmap=True)
        self.model=RecommendationOnly(self.graph);self.model.load_state_dict(stored['model']);self.model.eval()
        self.epoch=stored['epoch'];self.protocol=stored['protocol'];self.validation=stored['valid'];self.hash=file_sha(self.checkpoint)
        with torch.no_grad():self.kg=self.model.kg_embedding().detach()
        gc.collect()
    def logits(self,contexts):
        with torch.no_grad():
            users=self.model.encode_user(contexts,self.kg)
            return (users@self.kg[self.item_index].T+self.model.rec_bias[self.item_index]).numpy()
    def rank(self,row,topk=100):
        values=self.logits([row['context_entities']])[0];indices=np.argsort(-values,kind='stable')[:topk]
        return [{'movie_id':str(self.items[j]),'title':self.names[self.items[j]],'logit':float(values[j])} for j in indices]


def run(train_root):
    train_root=Path(train_root);root=train_root/'export';root.mkdir(parents=True,exist_ok=True)
    serving=KBRDServing(train_root/'best.pt')
    entities=json.loads((DATA/'entity2id.json').read_text());raw=json.loads((DATA/'valid_data.json').read_text());samples=native_samples(raw,entities);del raw
    groups=OrderedDict()
    for row in samples:
        if row['id'] not in groups:groups[row['id']]={'id':row['id'],'context_entities':row['context_entities'],'targets':[]}
        assert groups[row['id']]['context_entities']==row['context_entities'];groups[row['id']]['targets'].append(row['item'])
    rows=list(groups.values());ids=[r['id'] for r in rows];inputs=[{'id':r['id'],'context_entities':r['context_entities']} for r in rows]
    save(root/'valid.ids.json',ids);save(root/'native_item_ids.json',serving.items)
    metric=train_root.name
    selection=json.loads((train_root/'protocol.json').read_text()).get('selection_metric','valid_ce')
    save(root/'serving_metadata.json',{'checkpoint_sha256':serving.hash,'selected_epoch':serving.epoch,'selection_metric':selection,
         'fit':'train only','model':'CRSLab KBRD rec-only mathematical port','matrix':'native movie raw logits in exported ID order',
         'input_fields':['id','context_entities'],'generation_labels_available':False,'public_test_read':False})
    temp=root/'valid.native_logits.tmp.npy';matrix=np.lib.format.open_memmap(temp,mode='w+',dtype=np.float32,shape=(len(rows),len(serving.items)))
    predictions=[]
    for start in range(0,len(inputs),128):
        current=inputs[start:start+128];values=serving.logits([r['context_entities'] for r in current]);matrix[start:start+len(current)]=values
        ranks=np.argsort(-values,axis=1,kind='stable')[:,:100]
        for row,score,rank in zip(current,values,ranks):predictions.append({'id':row['id'],'ranking':[serving.items[j] for j in rank],'logits':[float(score[j]) for j in rank]})
    matrix.flush();del matrix;temp.replace(root/'valid.native_logits.npy');save(root/'valid.predictions.json',predictions)
    save(root/'generation_complete.json',{'input_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),'generation_labels_available':False})
    # Labels are consumed only after all serving predictions and matrix exist.
    full=evaluate(rows,[p['ranking'] for p in predictions]);lookup={r['id']:i for i,r in enumerate(rows)};summary={'selected_epoch':serving.epoch,'selection_metric':selection,
        'full_valid':full,'checkpoint_validation':serving.validation,'source_sha256':file_sha(Path(__file__))}
    cache=PrecomputedKBRD(root)
    for name in ('old32','new64'):
        prior=ROOT/'experiments/results/native_retrieval_inner_v1';pool_inputs=json.loads((prior/(name+'.inputs.json')).read_text());selected=[lookup[r['id']] for r in pool_inputs]
        summary[name]=evaluate([rows[i] for i in selected],[predictions[i]['ranking'] for i in selected]);save(root/(name+'.predictions.json'),[predictions[i] for i in selected])
        save(root/(name+'.shared_pool_scores.json'),[cache.score_pool(r['id'],[m['movie_id'] for m in r['candidates']]) for r in pool_inputs])
        summary[name+'_shared_retrieval_pool_oracle_recall100']=sum(t in {int(m['movie_id']) for m in inp['candidates']} for inp,i in zip(pool_inputs,selected) for t in rows[i]['targets'])/sum(len(rows[i]['targets']) for i in selected)
    summary['history_metric_recompute_abs']={k:abs(full['metrics'][k]-serving.validation['metrics'][k]) for k in ('recall@1','recall@10','recall@50','recall@100','ndcg@10','mrr@10') if k in full['metrics'] and k in serving.validation['metrics']}
    save(root/'summary.json',summary);print(json.dumps(summary,indent=2));return summary
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train-root',type=Path,required=True);args=p.parse_args();run(args.train_root)
