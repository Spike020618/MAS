"""Label-blind reviewer diagnostic selection; no response-label files or API.

This freezes candidate pairs and fit-derived quality assumptions, not complete
review requests. Full evidence packets need a prefix-only evidence loader first:
NativeEvidence currently opens the complete validation conversation file.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from experiments.consensus_recommendation.common import ROOT, create, sha, fingerprint
from experiments.paired_deliberation.native_review_calibration_data import blind_pool, conv

GAIN=ROOT/'experiments/results/native_gain_calibration_v1'
RANKER=ROOT/'experiments/results/native_calibrated_challenge_v3'
FIT_DATA=ROOT/'experiments/results/native_review_calibration_v2'
FIT_RESULT=ROOT/'experiments/results/native_reviewer_calibration_v1'
OUT=ROOT/'experiments/results/native_review_diagnostic_v1'
SEED='native-review-diagnostic-conversation-order-20261028:'


def select_blind(ids, diagnostic_ids, candidate_ids, offsets, mean, members, count=64):
    """One query per hash-selected conversation; within-query lowest margin/SD.

    No label/count input exists. Query selection precedes candidate selection and
    follows hash(case_id) within a conversation, avoiding label/quality cherry-pick.
    """
    if count <= 0: raise ValueError('positive count required')
    index={str(cid):i for i,cid in enumerate(ids)}
    if len(index)!=len(ids): raise ValueError('duplicate ids')
    by_conv={}
    for cid in diagnostic_ids:
        if cid not in index: raise ValueError('diagnostic ID missing from predictions')
        by_conv.setdefault(conv(cid),[]).append(cid)
    order=sorted(by_conv,key=lambda c:hashlib.sha256((SEED+c).encode()).hexdigest())
    selected=[]
    for conversation in order:
        cid=min(by_conv[conversation],key=lambda c:hashlib.sha256((SEED+'query:'+c).encode()).hexdigest())
        q=index[cid];sl=slice(int(offsets[q]),int(offsets[q+1]));pool=np.asarray(candidate_ids[sl]);mu=np.asarray(mean[sl]);ens=np.asarray(members[:,sl])
        if len(pool)!=len(mu) or ens.shape[1]!=len(pool):raise ValueError('prediction shape mismatch')
        if not np.isfinite(mu).all() or not np.isfinite(ens).all():raise ValueError('nonfinite forecast')
        ranking=np.argsort(-mu,kind='stable');ais=ranking[3:10];bis=ranking[10:]
        if not len(ais) or not len(bis):continue
        delta=ens[:,bis][:,None,:]-ens[:,ais][:,:,None];sd=delta.std(axis=0)
        gap=np.abs(mu[bis][None,:]-mu[ais][:,None]);normalized=gap/(sd+.001)
        ai,bi=min(((i,j)for i in range(len(ais))for j in range(len(bis))),key=lambda ab:(float(normalized[ab]),-float(sd[ab]),float(gap[ab]),10+ab[1],3+ab[0]))
        a,b=str(pool[ais[ai]]),str(pool[bis[bi]])
        rank=[str(pool[j])for j in ranking]
        target={'a':a,'b':b,'a_position':3+ai,'b_position':10+bi,'predicted_a_count':float(mu[ais[ai]]),'predicted_b_count':float(mu[bis[bi]]),'delta_member_sd':float(sd[ai,bi]),'standardized_margin':float(normalized[ai,bi]),'absolute_margin':float(gap[ai,bi]),'protected_prefix':3,'base_ranking_hash':fingerprint(rank)}
        selected.append({'id':cid,'case_id':cid,'task_id':'review-diag:'+cid+':'+a+':'+b,'pair':[a,b],'target':target,'blind_pool_native_ids':rank,'query_index':q,'query_labels_available':False})
        if len(selected)==count:break
    if len(selected)!=count:raise ValueError('not enough eligible diagnostic conversations')
    return selected


def run(output=OUT,count=64):
    output=Path(output)
    split=json.loads((FIT_DATA/'review_split_commitment.json').read_text())
    diag=split['partition']['review-cal-diag']['ids'];fit=split['partition']['review-cal-fit']['ids']
    if {conv(c)for c in diag}&{conv(c)for c in fit}:raise ValueError('conversation partition overlap')
    # Explicit allowlist of files read: no raw dialogue, targets or labels.
    ids=json.loads((GAIN/'hold20.ids.json').read_text());items=json.loads((GAIN/'native_item_ids.json').read_text())
    cand=np.load(RANKER/'candidate_ids.npy',mmap_mode='r');offset=np.load(RANKER/'candidate_offsets.npy',mmap_mode='r')
    mean=np.load(RANKER/'hold20.expected_count.npy',mmap_mode='r');members=np.load(RANKER/'hold20.member_expected_count.npy',mmap_mode='r')
    sr=np.load(GAIN/'sparse/hold20.native_ranks.npy',mmap_mode='r');kr=np.load(GAIN/'kbrd/hold20.native_ranks.npy',mmap_mode='r')
    pairs=select_blind(ids,diag,cand,offset,mean,members,count)
    for packet in pairs:
        q=packet['query_index'];stored=np.asarray(cand[offset[q]:offset[q+1]]).tolist()
        if stored!=blind_pool(items,sr[q],kr[q]):raise ValueError('pool not blind teacher union')
    summary=json.loads((FIT_RESULT/'summary.json').read_text())
    rates={name:row['effective_parameters']for name,row in summary['models'].items()}
    create(output/'frozen_fit_rates.json',{'summary_sha256':sha(FIT_RESULT/'summary.json'),'rates':rates,'scope':'selected balanced fit strata; conditional effective actions, not deployment accuracy or semantic truth'})
    create(output/'selected_pairs.blind.json',pairs)
    plan={'version':'native-review-diagnostic-v1-blind-selection','seed':SEED,'count':count,'conversation_count':len({conv(p['case_id'])for p in pairs}),'diagnostic_available_queries':len(diag),'diagnostic_available_conversations':len({conv(c)for c in diag}),'selection':'hash conversation, hash query, minimum standardized forecast margin pair; no observed winner stratification','candidate_pool':'verified blind Sparse100 union KBRD100, native item IDs unchanged','target_labels_read':False,'raw_dialogues_read':False,'model_api_calls':0,'complete_request_packets':False,'blocked_stage':'Need prefix-only evidence loader: existing NativeEvidence reads full valid_data.json including future movie annotations. Do not call it for this diagnostic claim.','next_step':'Build prefix-only scoped evidence and complete inputs; seal predictions before separately extracting labels. Do not reuse fit-stratum class quotas.','frozen_rates_sha256':sha(output/'frozen_fit_rates.json'),'selected_pairs_sha256':sha(output/'selected_pairs.blind.json'),'source_sha256':sha(Path(__file__)),'split_sha256':sha(FIT_DATA/'review_split_commitment.json')}
    create(output/'plan.json',plan);print(json.dumps(plan,indent=2));return plan

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(OUT));p.add_argument('--count',type=int,default=64);a=p.parse_args();run(Path(a.output),a.count)
