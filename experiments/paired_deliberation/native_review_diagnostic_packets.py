"""Build real reviewer requests from visible-prefix inputs and fold80 evidence only.

No diagnostic raw conversations, target annotations, labels or model APIs are read.
Role-scoped movie annotations are reconstructed only from training-resolved refs;
all visible context annotation IDs are separately disclosed without inventing roles.
"""
import argparse,json,re
from pathlib import Path
import numpy as np
from experiments.consensus_recommendation.common import ROOT,create,sha,fingerprint
from experiments.paired_deliberation.native_entity_evidence import NativeEvidence,REF
from experiments.paired_deliberation.native_review_calibration_data import forecast_cards

GAIN=ROOT/'experiments/results/native_gain_calibration_v1'
RANKER=ROOT/'experiments/results/native_calibrated_challenge_v3'
DIAG=ROOT/'experiments/results/native_review_diagnostic_v1'
FOLD=ROOT/'experiments/results/native_review_calibration_v2/fold-serving-data'
OUT=ROOT/'experiments/results/native_review_diagnostic_packets_v1'


def parse_visible(text,links,entities):
    parts=re.split(r'(?m)^(Seeker|Recommender): ',text)
    if parts[0] or len(parts)<3:raise ValueError('visible prefix role boundaries required')
    rows=[]
    for i in range(1,len(parts),2):
        body=parts[i+1].removesuffix('\n') if i+2<len(parts) else parts[i+1]
        movies=sorted({links[ref] for ref in REF.findall(body) if ref in links and links[ref] in entities})
        rows.append({'role':parts[i],'text':[body],'movies':movies})
    if '\n'.join(u['role']+': '+' '.join(u['text'])for u in rows)!=text:raise ValueError('lossy visible-prefix parse')
    return rows


class PrefixOnlyEvidence(NativeEvidence):
    def __init__(self,data_dir,visible_rows):
        super().__init__(data_dir)
        if self.valid_raw:raise ValueError('safe serving directory must contain no validation dialogue')
        self.prefixes={r['id']:parse_visible(r.get('original_text',r['text']),self.links,self.entities)for r in visible_rows}
    def visible_prefix(self,case_id):
        if case_id not in self.prefixes:raise ValueError('prefix not explicitly supplied')
        return self.prefixes[case_id]


def prepare_safe_directory(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    # Never link/open the original valid_data.json. Train and public catalog only.
    create(output/'valid_data.json',[])
    for name in ('train_data.json','entity2id.json','movie_ids.json','dbpedia_subkg.json'):
        p=output/name;target=(FOLD/name).resolve()
        if p.is_symlink():
            if p.resolve()!=target:raise ValueError('safe metadata link mismatch')
        elif p.exists():raise ValueError('unexpected safe data file')
        else:p.symlink_to(target)
    return output


def run(output=OUT):
    output=Path(output)
    pairs=json.loads((DIAG/'selected_pairs.blind.json').read_text())
    all_rows={r['id']:r for r in json.loads((GAIN/'hold20.inputs.json').read_text())}
    selected_rows=[all_rows[p['case_id']]for p in pairs]
    builder=PrefixOnlyEvidence(prepare_safe_directory(output/'prefix-only-serving-data'),selected_rows)
    fold80=set(json.loads((GAIN/'fold80.conversations.json').read_text()))
    if {str(c['conv_id'])for c in builder.train_raw}!=set(map(str,fold80)):raise ValueError('unexpected evidence training scope')
    if {r['conversation_id']for r in builder.training}&{p['case_id'].rsplit(':',1)[0]for p in pairs}:raise ValueError('query evidence fit overlap')
    ids=json.loads((GAIN/'hold20.ids.json').read_text());items=json.loads((GAIN/'native_item_ids.json').read_text())
    cand=np.load(RANKER/'candidate_ids.npy',mmap_mode='r');offsets=np.load(RANKER/'candidate_offsets.npy',mmap_mode='r')
    mean=np.load(RANKER/'hold20.expected_count.npy',mmap_mode='r');members=np.load(RANKER/'hold20.member_expected_count.npy',mmap_mode='r')
    arrays={'sparse_scores':np.load(GAIN/'sparse/hold20.native_scores.npy',mmap_mode='r'),'sparse_ranks':np.load(GAIN/'sparse/hold20.native_ranks.npy',mmap_mode='r'),'kbrd_logits':np.load(GAIN/'kbrd/hold20.native_logits.npy',mmap_mode='r'),'kbrd_ranks':np.load(GAIN/'kbrd/hold20.native_ranks.npy',mmap_mode='r')}
    inputs=[];states={}
    for p in pairs:
        cid=p['case_id'];q=p['query_index'];a,b=p['pair'];row=all_rows[cid]
        if ids[q]!=cid:raise ValueError('query index mismatch')
        ranking=p['blind_pool_native_ids'];target=p['target']
        if fingerprint(ranking)!=target['base_ranking_hash'] or ranking[target['a_position']]!=a or ranking[target['b_position']]!=b:raise ValueError('target base mismatch')
        serving={'id':cid,'text':row.get('original_text',row['text']),'candidates':[{'movie_id':m,'title':builder.names[int(m)]}for m in ranking]}
        evidence=builder.augment(serving,history_limit=3,kg_limit=3)
        offered={sid:card for sid,card in evidence['sources'].items() if card['kind'] in ('dialogue','historical_prediction_basis') or card.get('movie_id')in(a,b)}
        # Input-side aggregate context IDs have no per-role attribution; do not
        # manufacture a seeker utterance/movie preference from those annotations.
        context=[str(m)for m in row.get('context_movies',[])]
        sid='dialogue:visible-annotation-context'
        offered[sid]={'source_id':sid,'kind':'dialogue','role':'visible_context_annotation','origin_case_id':cid,'allowed_use':'dialogue_context','movie_ids':context,'text':'Visible prefix movie annotations (role unspecified): '+json.dumps([{'movie_id':m,'title':builder.names.get(int(m),m)}for m in context])+'. A mention does not certify liking or a current-user preference.','current_user_fact_authority':False,'semantic_entailment_certified':False}
        offered.update(forecast_cards(cid,a,b,q,items,**arrays))
        sl=slice(int(offsets[q]),int(offsets[q+1]));local=list(map(str,np.asarray(cand[sl]).tolist()))
        for mid in (a,b):
            j=local.index(mid);values=np.asarray(members[:,sl])[:,j].astype(float).tolist()
            card={'movie_id':mid,'predicted_target_count':float(mean[sl][j]),'three_member_stability_sd':float(np.std(values)),'member_predicted_counts':values}
            sid='forecast:calibrated:'+cid+':'+mid
            offered[sid]={'source_id':sid,'kind':'model_forecast','allowed_use':'prediction_basis','origin_case_id':'outer_ranker_calibration_fit_only','movie_id':mid,'text':'Calibrated expected next-recommender target occurrence count, not probability or observed truth: '+json.dumps(card,separators=(',',':'))+'. Current query excluded from ranker fitting.','current_user_fact_authority':False,'semantic_entailment_certified':False,'fit_scope':'ranker calibration-fit, disjoint original outer diagnostic queries'}
        inputs.append({'id':cid,'case_id':cid,'task_id':p['task_id'],'text':evidence['text'],'original_text':serving['text'],'pair':[a,b],'target':target,'pair_candidates':[m for m in serving['candidates']if m['movie_id']in(a,b)],'blind_pool_native_ids':ranking,'blind_pool_hash':fingerprint(sorted(ranking)),'candidate_evidence':{m:evidence['candidate_evidence'][m]for m in(a,b)},'sources':offered,'query_labels_available':False,'prediction_scope':'soft next-recommendation forecast; no truth certification'})
        states[p['task_id']]=ranking
    create(output/'pair64.inputs.json',inputs);create(output/'pair_states.json',states)
    create(output/'frozen_fit_rates.json',json.loads((DIAG/'frozen_fit_rates.json').read_text()))
    plan={'version':'native-review-diagnostic-prefix-only-packets-v1','count':len(inputs),'conversation_count':len({p['case_id'].rsplit(':',1)[0]for p in pairs}),'complete_request_packets':True,'model_api_calls':0,'query_target_labels_read':False,'query_raw_dialogues_read':False,'safe_valid_data':[],'serving_inputs_sha256':sha(GAIN/'hold20.inputs.json'),'selected_pairs_sha256':sha(DIAG/'selected_pairs.blind.json'),'frozen_rates_sha256':sha(output/'frozen_fit_rates.json'),'inputs_sha256':sha(output/'pair64.inputs.json'),'states_sha256':sha(output/'pair_states.json'),'source_sha256':sha(Path(__file__)),'evidence_scope':'fold80 training only; CURRENT visible text from sealed input prefixes; no diagnostic raw dialogues','difference_from_fit_packets':'Role movie annotations only training-link reconstructed; original visible context IDs are aggregate role-unspecified card. No future metadata opened. Same packets supplied to all reviewers.'}
    create(output/'plan.json',plan);print(json.dumps(plan,indent=2));return plan

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(OUT));a=parser.parse_args();run(Path(a.output))
