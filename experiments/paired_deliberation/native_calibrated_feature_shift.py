"""No-label feature-shift audit: fold80 teacher calibration-fit -> fulltrain serving."""
import json
from pathlib import Path
import numpy as np
from experiments.consensus_recommendation.common import ROOT,create,sha
from experiments.paired_deliberation.native_calibrated_challenge import RESULT as CALIBRATOR,DATA,FEATURES,pool_indices,candidate_features
from experiments.paired_deliberation.native_calibrated_transfer import RESULT as TRANSFER


def run():
    ids=json.loads((DATA/'hold20.ids.json').read_text());inputs=json.loads((DATA/'hold20.inputs.json').read_text());items=json.loads((DATA/'native_item_ids.json').read_text())
    selected=set(json.loads((CALIBRATOR/'split_commitment.json').read_text())['fit']['ids'])
    arrays={'sparse_scores':np.load(DATA/'sparse/hold20.native_scores.npy',mmap_mode='r'),'sparse_ranks':np.load(DATA/'sparse/hold20.native_ranks.npy',mmap_mode='r'),
        'kbrd_logits':np.load(DATA/'kbrd/hold20.native_logits.npy',mmap_mode='r'),'kbrd_ranks':np.load(DATA/'kbrd/hold20.native_ranks.npy',mmap_mode='r')}
    total=np.zeros(len(FEATURES));squared=np.zeros(len(FEATURES));n=0
    for i,cid in enumerate(ids):
        if cid not in selected:continue
        ix=pool_indices(arrays['sparse_ranks'][i],arrays['kbrd_ranks'][i]);x=candidate_features(inputs[i],items,ix,**{k:np.asarray(v[i]) for k,v in arrays.items()}).astype(float)
        total+=x.sum(axis=0);squared+=(x*x).sum(axis=0);n+=len(x)
    mean=total/n;std=np.sqrt(np.maximum(0,squared/n-mean*mean));serving=json.loads((TRANSFER/'serving_metadata.json').read_text())
    other_mean=np.array(serving['serving_feature_mean']);other_std=np.array(serving['serving_feature_std']);delta=(other_mean-mean)/np.maximum(std,1e-8)
    records=[{'feature':f,'calibration_fit_mean':float(mean[j]),'calibration_fit_std':float(std[j]),'fulltrain_public_valid_mean':float(other_mean[j]),
        'fulltrain_public_valid_std':float(other_std[j]),'standardized_mean_difference':float(delta[j])} for j,f in enumerate(FEATURES)]
    audit={'source_sha256':sha(Path(__file__)),'calibration_fit_queries':len(selected),'calibration_fit_candidate_rows':n,'serving_queries':len(json.loads((TRANSFER/'valid.ids.json').read_text())),
        'no_outcome_labels_read':True,'no_parameters_changed':True,'features':records,'max_abs_standardized_mean_difference':float(np.max(np.abs(delta))),
        'features_abs_smd_gt_point25':[r['feature'] for r in records if abs(r['standardized_mean_difference'])>.25],
        'interpretation':'descriptive covariate shift only; rank/z/relative scaling limits teacher-size effects, no outcome-based recalibration; no guarantee of uncertainty calibration'}
    create(TRANSFER/'feature_shift_audit.json',audit);print(json.dumps(audit,indent=2));return audit
if __name__=='__main__':run()
