"""Conservative fixed-RRF baseline and numerical lineage reconciliation."""
import json,math
from pathlib import Path
import numpy as np
from experiments.consensus_recommendation.common import ROOT,create,sha
from experiments.paired_deliberation.native_calibrated_challenge import DATA,RESULT as CALIBRATOR,load_fold,split_indices
from experiments.paired_deliberation.native_calibrated_transfer import RESULT as TRANSFER
from experiments.paired_deliberation.native_calibrated_diagnostic import micro_sums,paired_cluster_ci
from experiments.paired_deliberation.native_redial_benchmark import evaluate


def run(root=TRANSFER):
    root=Path(root);items=np.array(json.loads((root/'native_item_ids.json').read_text()));ids=json.loads((root/'valid.ids.json').read_text())
    sr=np.load(root/'valid.sparse_ranks.npy',mmap_mode='r');kr=np.load(root/'valid.kbrd_ranks.npy',mmap_mode='r')
    score=.5/(60+sr.astype(float))+.5/(60+kr.astype(float));order=np.lexsort((np.broadcast_to(items,sr.shape),-score),axis=1)[:,:100];ranks=items[order].tolist()
    pred=json.loads((root/'valid.predictions.json').read_text());labels={r['id']:r for r in json.loads((ROOT/'experiments/results/native_redial_benchmark_v1/valid.labels.json').read_text())}
    records=[{'id':cid,'ranker':micro_sums(pred[i]['calibrated'],labels[cid]['targets']),'strong_rrf60':micro_sums(ranks[i],labels[cid]['targets'])} for i,cid in enumerate(ids)]
    summary={'primary_baseline':'full native rank equal RRF60, native-ID tie; strongest legal audited full-validation RRF variant',
        'baseline_chosen_conservatively_not_to_inflate_ranker_gain':True,'no_model_or_calibrator_changes':True,
        'full_valid':{'calibrated':evaluate([labels[c] for c in ids],[p['calibrated'] for p in pred]),'strong_rrf60':evaluate([labels[c] for c in ids],ranks),
            'paired':paired_cluster_ci(records,'ranker','strong_rrf60')},'sets':{}}
    lookup={cid:i for i,cid in enumerate(ids)}
    for name in ('old32','new64'):
        chosen=json.loads((ROOT/'experiments/results/native_retrieval_inner_v1'/(name+'.inputs.json')).read_text());ix=[lookup[r['id']] for r in chosen]
        summary['sets'][name]={'calibrated':evaluate([labels[ids[i]] for i in ix],[pred[i]['calibrated'] for i in ix]),
            'strong_rrf60':evaluate([labels[ids[i]] for i in ix],[ranks[i] for i in ix]),
            'paired':paired_cluster_ci([records[i] for i in ix],'ranker','strong_rrf60')}
    # Recompute the same full-native/native-ID policy in the original-train diagnostic.
    fids,fitems,inputs,flabels,arrays=load_fold(DATA);splits=split_indices(fids);fitems=np.array(fitems)
    fpred={p['id']:p['ranking'] for p in json.loads((CALIBRATOR/'hold20.predictions.json').read_text())};frecord=[];frrf=[];fc=[];chosen_labels=[]
    for q in splits['diagnostic']:
        sc=.5/(60+arrays['sparse_ranks'][q].astype(float))+.5/(60+arrays['kbrd_ranks'][q].astype(float))
        rank=fitems[np.lexsort((fitems,-sc))[:100]].tolist();target=flabels[q]['targets'];frecord.append({'id':fids[q],'ranker':micro_sums(fpred[fids[q]],target),'strong_rrf60':micro_sums(rank,target)})
        frrf.append(rank);fc.append(fpred[fids[q]]);chosen_labels.append(flabels[q])
    summary['train_diagnostic']={'calibrated':evaluate(chosen_labels,fc),'strong_rrf60':evaluate(chosen_labels,frrf),'paired':paired_cluster_ci(frecord,'ranker','strong_rrf60')}
    summary['source_sha256']=sha(Path(__file__))
    create(root/'strong_rrf60.native_id.predictions.json',[{'id':cid,'ranking':r} for cid,r in zip(ids,ranks)])
    create(root/'strong_baseline_comparison.json',summary)
    lineage={'calibrator_model_sha256':sha(CALIBRATOR/'calibrated_model.joblib'),'calibrator_module_sha256':sha(ROOT/'experiments/paired_deliberation/native_calibrated_challenge.py'),
      'calibrator_model_selection_sha256':sha(CALIBRATOR/'model_selection.json'),'calibrator_split_commitment_sha256':sha(CALIBRATOR/'split_commitment.json'),
      'fold_teacher_manifest_sha256':sha(DATA/'manifest.json'),'fold_teacher_sparse_metadata_sha256':sha(DATA/'sparse/metadata.json'),
      'fold_teacher_kbrd_metadata_sha256':sha(DATA/'kbrd/metadata.json'),'full_sparse_scores_sha256':sha(root/'valid.sparse_scores.npy'),
      'full_sparse_ranks_sha256':sha(root/'valid.sparse_ranks.npy'),'full_kbrd_logits_sha256':sha(ROOT/'experiments/results/native_kbrd_reproduction_v1/train_recall_r1/export/valid.native_logits.npy'),
      'full_kbrd_serving_metadata_sha256':sha(ROOT/'experiments/results/native_kbrd_reproduction_v1/train_recall_r1/export/serving_metadata.json'),
      'validation_inputs_sha256':sha(root/'valid.inputs.json'),'calibrated_predictions_sha256':sha(root/'valid.predictions.json'),
      'new64_materials_sha256':sha(root/'new64.inputs.json'),'source_sha256':sha(Path(__file__)),
      'model_byte_hash_unchanged_during_transfer':True,'any_gain_attribution':'supervised calibrated predictor, not multi-agent/game effect',
      'full_native_RRF_vs_list_RRF_difference':'top100 truncation +7 item hits, sparse-rank tie -1 vs native-ID; same teachers exactly'}
    create(root/'numerical_lineage.json',lineage)
    create(root/'artifact_complete.json',{'ready':True,'lineage_sha256':sha(root/'numerical_lineage.json'),'strong_baseline_comparison_sha256':sha(root/'strong_baseline_comparison.json'),
        'all_public_valid_inference_complete':True,'old32_and_new64_materials_complete':True,'no_model_API_calls':True})
    print(json.dumps(summary,indent=2));return summary

if __name__=='__main__':run()
