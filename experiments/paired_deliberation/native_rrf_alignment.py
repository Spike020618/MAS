"""Read-only reconciliation of full-native vs truncated-list RRF60 baselines."""
import json
from pathlib import Path
import numpy as np
from experiments.consensus_recommendation.common import ROOT,create,sha
from experiments.paired_deliberation.native_redial_benchmark import evaluate

TRANSFER=ROOT/'experiments/results/native_calibrated_transfer_v1'


def run(root=TRANSFER):
    root=Path(root);ids=json.loads((root/'valid.ids.json').read_text());items=np.array(json.loads((root/'native_item_ids.json').read_text()))
    sr=np.load(root/'valid.sparse_ranks.npy',mmap_mode='r');kr=np.load(root/'valid.kbrd_ranks.npy',mmap_mode='r');scores=np.load(root/'valid.sparse_scores.npy',mmap_mode='r')
    original_sparse={r['id']:r for r in json.loads((ROOT/'experiments/results/native_retrieval_inner_v1/valid.predictions.json').read_text())}
    original_kbrd={r['id']:r for r in json.loads((ROOT/'experiments/results/native_kbrd_reproduction_v1/train_recall_r1/export/valid.predictions.json').read_text())}
    original_rrf={r['id']:r['ranking'] for r in json.loads((ROOT/'experiments/results/native_kbrd_reproduction_v1/fixed_rrf60.valid.predictions.json').read_text())}
    labels={r['id']:r for r in json.loads((ROOT/'experiments/results/native_redial_benchmark_v1/valid.labels.json').read_text())};selected_labels=[labels[c] for c in ids]
    parity={'query_count':len(ids),'sparse_top100_equal_rows':0,'kbrd_top100_equal_rows':0,'sparse_reported_score_max_abs_difference':0.}
    indices={int(m):j for j,m in enumerate(items)}
    for i,cid in enumerate(ids):
        s=items[np.argsort(sr[i],kind='stable')[:100]].tolist();k=items[np.argsort(kr[i],kind='stable')[:100]].tolist()
        parity['sparse_top100_equal_rows']+=s==original_sparse[cid]['ranking'];parity['kbrd_top100_equal_rows']+=k==original_kbrd[cid]['ranking']
        difference=max(abs(float(scores[i,indices[int(m)]])-float(v)) for m,v in zip(original_sparse[cid]['ranking'],original_sparse[cid]['scores']))
        parity['sparse_reported_score_max_abs_difference']=max(parity['sparse_reported_score_max_abs_difference'],difference)
    variants={};metrics={}
    for truncation in ('full_native','top100_lists'):
        value=.5/(60+sr.astype(np.float64))+.5/(60+kr.astype(np.float64))
        if truncation=='top100_lists':value=.5*np.where(sr<=100,1/(60+sr.astype(np.float64)),0)+.5*np.where(kr<=100,1/(60+kr.astype(np.float64)),0)
        for tie in ('sparse_rank','native_id'):
            secondary=sr if tie=='sparse_rank' else np.broadcast_to(items,sr.shape)
            order=np.lexsort((secondary,-value),axis=1)[:,:100];ranks=items[order].tolist();name=truncation+'__'+tie
            variants[name]=ranks;metrics[name]=evaluate(selected_labels,ranks)
    legacy_parity=sum(variants['top100_lists__native_id'][i]==original_rrf[c] for i,c in enumerate(ids))
    transitions=[]
    for left,right in [('top100_lists__native_id','top100_lists__sparse_rank'),('top100_lists__native_id','full_native__native_id'),('full_native__native_id','full_native__sparse_rank')]:
        gained=lost=changed=0
        for i,row in enumerate(selected_labels):
            a=variants[left][i];b=variants[right][i];changed+=a[:10]!=b[:10]
            for t in row['targets']:gained+=t in b[:10] and t not in a[:10];lost+=t in a[:10] and t not in b[:10]
        transitions.append({'from':left,'to':right,'changed_top10_queries':changed,'gained_item_hits':gained,'lost_item_hits':lost})
    strongest=max(metrics,key=lambda k:(metrics[k]['metrics']['recall@10'],metrics[k]['metrics']['ndcg@10']))
    result={'source_sha256':sha(Path(__file__)),'teacher_parity':parity,'legacy_truncated_native_id_equal_rows':legacy_parity,
        'variant_metrics':metrics,'transitions':transitions,'strongest_legal_rrf_variant':strongest,
        'interpretation':'baseline reconciliation only, no calibrator or RRF parameter retuning; all use alpha=.5,k60; missing-list zero vs full ranks and tie policy explicitly differ'}
    create(root/'rrf_alignment_audit.json',result);print(json.dumps(result,indent=2));return result

if __name__=='__main__':run()
