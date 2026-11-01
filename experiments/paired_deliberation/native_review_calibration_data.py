"""Train-fold-only evidence and label-stratified review calibration tasks.

No API calls. Reuses original-train outer ranker diagnostic only: half becomes
review-fit, half remains unused review-diagnostic. Observed target-count winner
is a training label, never semantic truth or a prompt field.
"""
import argparse,collections,hashlib,json,math,os
from pathlib import Path
import numpy as np
from experiments.consensus_recommendation.common import ROOT,create,sha,fingerprint
from experiments.paired_deliberation.native_entity_evidence import NativeEvidence

GAIN=ROOT/'experiments/results/native_gain_calibration_v1'
RANKER=ROOT/'experiments/results/native_calibrated_challenge_v3'
OFFICIAL=ROOT/'experiments/data/raw/crslab_redial_native/released'
OUT=ROOT/'experiments/results/native_review_calibration_v2'
SEED='native-review-calibration-conversation-split-20261028:'
QUOTAS={'b':24,'a':24,'tie':16}


def conv(case_id):return str(case_id).rsplit(':',1)[0]

def review_partition(ids):
    fit=[];diagnostic=[]
    for case_id in ids:
        bit=int(hashlib.sha256((SEED+conv(case_id)).encode()).hexdigest(),16)%2
        (fit if bit==0 else diagnostic).append(case_id)
    return {'review-cal-fit':fit,'review-cal-diag':diagnostic}


def prepare_fold_directory(output,gain=GAIN,official=OFFICIAL):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fit={str(x) for x in json.loads((gain/'fold80.conversations.json').read_text())};hold={str(x) for x in json.loads((gain/'hold20.conversations.json').read_text())}
    if fit&hold:raise ValueError('fold80 and hold20 conversations overlap')
    raw=json.loads((official/'train_data.json').read_text())
    train=[c for c in raw if str(c['conv_id']) in fit];valid=[c for c in raw if str(c['conv_id']) in hold]
    if {str(c['conv_id'])for c in train}!=fit or {str(c['conv_id'])for c in valid}!=hold:raise ValueError('missing fold conversation')
    create(output/'train_data.json',train);create(output/'valid_data.json',valid)
    for name in ('entity2id.json','movie_ids.json','dbpedia_subkg.json','token2id.json','concept2id.json','conceptnet_subkg.txt'):
        link=output/name;target=(official/name).resolve()
        if link.is_symlink():
            if link.resolve()!=target:raise ValueError('metadata link mismatch')
        elif link.exists():raise ValueError('unexpected copied metadata at '+str(link))
        else:link.symlink_to(target)
    return {'fold80_conversations':len(train),'hold20_conversations':len(valid),'overlap':0,
            'fit_cids_hash':fingerprint(sorted(fit)),'hold_cids_hash':fingerprint(sorted(hold)),
            'train_data_sha256':sha(output/'train_data.json'),'valid_data_sha256':sha(output/'valid_data.json'),
            'metadata_symlink_targets':{p.name:str(p.resolve()) for p in output.iterdir() if p.is_symlink()}},train,valid


def read_fit_targets(raw,fit_ids,entities):
    """Read movies only in requested fit responses; diagnostic dialogue untouched."""
    fit_ids=set(fit_ids);allowed={conv(c)for c in fit_ids};targets=collections.defaultdict(list)
    for conversation in raw:
        cid=str(conversation['conv_id'])
        if cid not in allowed:continue
        turn=-1;last=None
        for u in conversation['dialog']:
            role=u['role']
            if role!=last:turn+=1;last=role
            case_id=cid+':'+str(turn)
            if case_id in fit_ids:
                if role!='Recommender':raise ValueError('fit label outside recommender response')
                targets[case_id].extend(entities[m] for m in u['movies'] if m in entities)
    if set(targets)!=fit_ids or any(not t for t in targets.values()):raise ValueError('fit response labels incomplete')
    return dict(targets)


def blind_pool(item_ids,sparse_ranks,kbrd_ranks):
    ix=np.flatnonzero((sparse_ranks<=100)|(kbrd_ranks<=100));score=.5/(60+sparse_ranks[ix])+.5/(60+kbrd_ranks[ix])
    ix=ix[np.lexsort((sparse_ranks[ix],-score))]
    return np.asarray(item_ids,dtype=np.int32)[ix].tolist()


def select_pairs(ids,fit_ids,observed,candidate_ids,offsets,mean,members):
    best={winner:{} for winner in QUOTAS};availability=collections.Counter();lookup={case_id:i for i,case_id in enumerate(ids)}
    for case_id in fit_ids:
        q=lookup[case_id];sl=slice(offsets[q],offsets[q+1]);pool=np.asarray(candidate_ids[sl]);mu=np.asarray(mean[sl]);ens=np.asarray(members[:,sl]);order=np.argsort(-mu,kind='stable')
        ais=order[3:10];bis=order[10:]
        if not len(ais) or not len(bis):continue
        counts=collections.Counter(observed[case_id]);ca=np.array([counts[int(pool[j])] for j in ais]);cb=np.array([counts[int(pool[j])] for j in bis])
        delta=ens[:,bis][:,None,:]-ens[:,ais][:,:,None];sd=delta.std(axis=0);gap=np.abs(mu[bis][None,:]-mu[ais][:,None]);normalized=gap/(sd+.001)
        for winner in QUOTAS:
            mask=cb[None,:]>ca[:,None] if winner=='b' else ca[:,None]>cb[None,:] if winner=='a' else cb[None,:]==ca[:,None]
            ai,bi=np.nonzero(mask)
            if not len(ai):continue
            # Predeclared: smallest standardized margin, largest ensemble delta SD,
            # smallest absolute margin, challenger rank, incumbent rank, query hash.
            choices=sorted(zip(ai.tolist(),bi.tolist()),key=lambda ab:(float(normalized[ab]),-float(sd[ab]),float(gap[ab]),int(np.flatnonzero(order==bis[ab[1]])[0]),int(np.flatnonzero(order==ais[ab[0]])[0])))
            aidx,bidx=choices[0];a=int(pool[ais[aidx]]);b=int(pool[bis[bidx]])
            a_pos=int(np.flatnonzero(order==ais[aidx])[0]);b_pos=int(np.flatnonzero(order==bis[bidx])[0])
            priority=(float(normalized[aidx,bidx]),-float(sd[aidx,bidx]),float(gap[aidx,bidx]),b_pos,a_pos,hashlib.sha256((SEED+case_id).encode()).hexdigest())
            packet={'id':case_id,'a':str(a),'b':str(b),'a_position':a_pos,'b_position':b_pos,'predicted_a_count':float(mu[ais[aidx]]),
                'predicted_b_count':float(mu[bis[bidx]]),'delta_member_sd':float(sd[aidx,bidx]),'standardized_margin':float(normalized[aidx,bidx]),
                'absolute_margin':float(gap[aidx,bidx]),'protected_prefix':3,'ranking':[str(pool[j]) for j in order],
                'priority':priority,'winner':winner,'observed_a_count':int(counts[a]),'observed_b_count':int(counts[b]),'query_index':q}
            old=best[winner].get(conv(case_id))
            if old is None or priority<tuple(old['priority']):best[winner][conv(case_id)]=packet
    selected=[];used=set();shortage={}
    for winner in ('b','a','tie'):
        choices=sorted(best[winner].values(),key=lambda r:tuple(r['priority']));availability[winner]=len(choices);take=[]
        for packet in choices:
            if conv(packet['id']) in used:continue
            take.append(packet);used.add(conv(packet['id']))
            if len(take)==QUOTAS[winner]:break
        selected+=take
        if len(take)<QUOTAS[winner]:shortage[winner]={'requested':QUOTAS[winner],'available_after_disjoint_selection':len(take)}
    return selected,dict(availability),shortage


def forecast_cards(case_id,a,b,q,items,sparse_scores,sparse_ranks,kbrd_logits,kbrd_ranks):
    lookup={int(m):j for j,m in enumerate(items)};pair=[a,b];cards={}
    for name,score_array,rank_array,scope in [('sparse',sparse_scores,sparse_ranks,'train-fold sparse score, not probability'),('kbrd',kbrd_logits,kbrd_ranks,'train-fold KBRD logit, not calibrated count')]:
        values=[{'movie_id':m,'native_rank':int(rank_array[q,lookup[int(m)]]),'score':float(score_array[q,lookup[int(m)]])}for m in pair]
        sid='forecast:'+name+':'+case_id
        cards[sid]={'source_id':sid,'kind':'model_forecast','allowed_use':'prediction_basis','origin_case_id':'fold80_predictor_'+name,
            'movie_ids':pair,'text':scope+'; predictions for this CURRENT prefix only: '+json.dumps(values,separators=(',',':'))+'. Forecasts are not observed targets or verified preferences.',
            'current_user_fact_authority':False,'semantic_entailment_certified':False,'fit_scope':'official-train-fold80-only'}
    return cards


def run(output=OUT):
    output=Path(output);fold,train,hold=prepare_fold_directory(output/'fold-serving-data')
    rank_split=json.loads((RANKER/'split_commitment.json').read_text());outer=rank_split['diagnostic']['ids']
    if len(outer)!=1157:raise ValueError('expected existing outer diagnostic cohort 1157')
    partition=review_partition(outer)
    commitment={'version':'review-reliability-calibration-data-v2-calibrated-forecast','seed':SEED,'split_unit':'conversation SHA256 parity',
        'partition':{key:{'ids':rows,'queries':len(rows),'conversations':len({conv(c)for c in rows})}for key,rows in partition.items()},
        'ranker_outer_diagnostic_cohort_reuse':'review-cal-fit is NOW training material; cannot keep calling all1157 untouched final diagnostic',
        'ranker_calibration_fit_intersection':len(set(rank_split['fit']['ids'])&set(outer)),
        'label_selection_after_split_commitment':True,'no_model_api_calls':True,'quotas':QUOTAS,
        'pair_rule':'a rank4..10, b outside10, blind sparse100-union-KBRD100 <=200; low abs(mean delta)/(member SD+.001), high delta SD ties',
        'conversation_cap':'one selected task per original conversation; quota selection order b,a,tie; no query duplication'}
    create(output/'review_split_commitment.json',commitment)
    fit_ids=partition['review-cal-fit'];diag_ids=partition['review-cal-diag']
    if {conv(c)for c in fit_ids}&{conv(c)for c in diag_ids}:raise ValueError('review partition overlap')
    entities=json.loads((OFFICIAL/'entity2id.json').read_text());observed=read_fit_targets(hold,fit_ids,entities)
    ids=json.loads((GAIN/'hold20.ids.json').read_text());items=json.loads((GAIN/'native_item_ids.json').read_text());rows={r['id']:r for r in json.loads((GAIN/'hold20.inputs.json').read_text())}
    candidate_ids=np.load(RANKER/'candidate_ids.npy',mmap_mode='r');offsets=np.load(RANKER/'candidate_offsets.npy',mmap_mode='r');mean=np.load(RANKER/'hold20.expected_count.npy',mmap_mode='r');members=np.load(RANKER/'hold20.member_expected_count.npy',mmap_mode='r')
    arrays={'sparse_scores':np.load(GAIN/'sparse/hold20.native_scores.npy',mmap_mode='r'),'sparse_ranks':np.load(GAIN/'sparse/hold20.native_ranks.npy',mmap_mode='r'),
        'kbrd_logits':np.load(GAIN/'kbrd/hold20.native_logits.npy',mmap_mode='r'),'kbrd_ranks':np.load(GAIN/'kbrd/hold20.native_ranks.npy',mmap_mode='r')}
    # Pool verification and selection inputs have no labels; stratification below
    # deliberately reads only review-fit labels as supervised training metadata.
    id_lookup={c:i for i,c in enumerate(ids)}
    for case_id in fit_ids:
        q=id_lookup[case_id];stored=np.asarray(candidate_ids[offsets[q]:offsets[q+1]]).tolist();blind=blind_pool(items,arrays['sparse_ranks'][q],arrays['kbrd_ranks'][q])
        if stored!=blind:raise ValueError('candidate pool differs from blind teacher union')
    selected,available,shortage=select_pairs(ids,fit_ids,observed,candidate_ids,offsets,mean,members)
    create(output/'selected_pairs.training_only.json',[{k:v for k,v in row.items() if k!='ranking'}for row in selected])
    builder=NativeEvidence(output/'fold-serving-data');train_cids={str(c['conv_id'])for c in train};hold_cids={str(c['conv_id'])for c in hold}
    if {r['conversation_id']for r in builder.training} - train_cids:raise ValueError('evidence fitted outside fold80')
    if {r['conversation_id']for r in builder.training}&hold_cids:raise ValueError('hold20 used in evidence training')
    inputs=[];labels=[]
    for packet in selected:
        cid=packet['id'];q=packet['query_index'];a,b=packet['a'],packet['b'];task_id='review-fit:'+cid+':'+a+':'+b
        row=rows[cid];serving={'id':cid,'text':row.get('original_text',row['text']),
            'candidates':[{'movie_id':m,'title':builder.names[int(m)]}for m in packet['ranking']]}
        evidence=builder.augment(serving,history_limit=3,kg_limit=3)
        offered={sid:card for sid,card in evidence['sources'].items() if card['kind'] in ('dialogue','historical_prediction_basis') or card.get('movie_id') in (a,b)}
        offered.update(forecast_cards(cid,a,b,q,items,**arrays))
        sl=slice(offsets[q],offsets[q+1]);local_ids=list(map(str,np.asarray(candidate_ids[sl]).tolist()))
        for mid in (a,b):
            j=local_ids.index(mid);member_counts=np.asarray(members[:,sl])[:,j].astype(float).tolist()
            forecast={'movie_id':mid,'predicted_target_count':float(mean[sl][j]),'three_member_stability_sd':float(np.std(member_counts)),'member_predicted_counts':member_counts}
            sid='forecast:calibrated:'+cid+':'+mid
            offered[sid]={'source_id':sid,'kind':'model_forecast','allowed_use':'prediction_basis','origin_case_id':'outer_ranker_calibration_fit_only',
                'movie_id':mid,'text':'Calibrated expected next-recommender target occurrence count, not probability or observed truth: '+json.dumps(forecast,separators=(',',':'))+'. The current query was excluded from ranker fitting.',
                'current_user_fact_authority':False,'semantic_entailment_certified':False,'fit_scope':'ranker calibration-fit, disjoint from original outer diagnostic queries'}
        target={k:packet[k]for k in ('a','b','a_position','b_position','predicted_a_count','predicted_b_count','delta_member_sd','standardized_margin','absolute_margin','protected_prefix')}
        target['base_ranking_hash']=fingerprint(packet['ranking'])
        inputs.append({'id':cid,'task_id':task_id,'case_id':cid,'text':evidence['text'],'original_text':serving['text'],
            'pair':[a,b],'target':target,'pair_candidates':[m for m in evidence['candidates']if m['movie_id']in(a,b)],
            'blind_pool_native_ids':packet['ranking'],'blind_pool_hash':fingerprint(sorted(packet['ranking'])),
            'candidate_evidence':{m:evidence['candidate_evidence'][m]for m in (a,b)},'sources':offered,
            'query_labels_available':False,'prediction_scope':'soft next-recommendation forecast only; no truth certification'})
        labels.append({'id':cid,'task_id':task_id,'pair':[a,b],'winner':packet['winner'],'observed_a_count':packet['observed_a_count'],'observed_b_count':packet['observed_b_count'],
            'label_scope':'supervised review training only; observed count comparison, not semantic/factual truth'})
    # Serving order is label-blind after stratified training selection.
    order=sorted(range(len(inputs)),key=lambda i:hashlib.sha256(('native-review-serving-order:'+inputs[i]['task_id']).encode()).hexdigest())
    inputs=[inputs[i] for i in order];labels=[labels[i] for i in order]
    create(output/'pair64.inputs.json',inputs);create(output/'pair64.labels.json',labels)
    winner_ids={r['task_id']for r in labels if r['winner']!='tie'}
    create(output/'fit48.inputs.json',[r for r in inputs if r['task_id']in winner_ids]);create(output/'fit48.labels.json',[r for r in labels if r['winner']!='tie'])
    create(output/'tie16.inputs.json',[r for r in inputs if r['task_id']not in winner_ids]);create(output/'tie16.labels.json',[r for r in labels if r['winner']=='tie'])
    audit={'source_sha256':sha(Path(__file__)),'fold':fold,'review_fit_queries':len(fit_ids),'review_fit_conversations':len({conv(c)for c in fit_ids}),
        'review_diag_queries_untouched_for_selection':len(diag_ids),'review_diag_conversations':len({conv(c)for c in diag_ids}),
        'available_unique_conversations_by_class':available,'requested_quotas':QUOTAS,'selected_counts':dict(collections.Counter(r['winner']for r in labels)),
        'actual_pair_count':len(inputs),'actual_q_fit_winner_count':len(winner_ids),'shortage':shortage,
        'serving_order':'label-blind task_id SHA256, no class block or observed winner in visible IDs',
        'calibrated_forecast_cards_per_task':2,'q_b_q_a_symmetry_assumed':False,
        'fold80_evidence_training_contexts':len(builder.training),'fold80_history_conversation_overlap_with_hold20':0,
        'pool_verified_blind_top100_union':True,'query_labels_in_prompt':False,'labels_used_only_review_cal_fit':True,
        'hold20_labels_json_opened':False,'public_valid_test_labels_read':False,'model_api_calls':0,
        'research_history_caveat':'outer ranker diagnostic is partly repurposed for review training; full1157 is no longer untouched diagnostic',
        'class_balance_warning':'24/24/16 is supervised stratified training, not natural winner prevalence or a representative accuracy test',
        'metadata_symlinks_are_public_catalog_and_KG_not_query_outcomes':True,
        'source_hashes':{'ranker_split':sha(RANKER/'split_commitment.json'),'ranker_fitting_complete':sha(RANKER/'fitting_complete.json'),
            'ranker_selected_model':sha(RANKER/'selected_model.json'),'gain_fold80_cids':sha(GAIN/'fold80.conversations.json'),
            'gain_hold20_cids':sha(GAIN/'hold20.conversations.json'),'review_split_commitment':sha(output/'review_split_commitment.json')},
        'artifact_hashes':{name:sha(output/name)for name in ('pair64.inputs.json','pair64.labels.json','fit48.inputs.json','fit48.labels.json','tie16.inputs.json','tie16.labels.json')}}
    create(output/'audit.json',audit);print(json.dumps(audit,indent=2));return audit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(OUT));a=p.parse_args();run(Path(a.output))
