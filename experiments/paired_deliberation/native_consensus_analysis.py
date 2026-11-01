"""Exploratory paired dialogue-cluster bootstrap, after generation sealing."""
import json,copy
from pathlib import Path
from itertools import product
from collections import defaultdict
import numpy as np
from experiments.consensus_recommendation.common import create,fingerprint,sha
from experiments.paired_deliberation.rapid_negotiation import scores,local_update
from experiments.paired_deliberation.native_consensus_experiment import readout
from experiments.paired_deliberation.native_consensus_experiment import RESULT,OUT,METHODS,audit
from experiments.paired_deliberation.benchmark_evaluation import native_item_metrics
from experiments.paired_deliberation.budget_v6 import AttemptClient


def cluster_difference(labels,left,right,replicates=10000,seed=271027):
    groups=defaultdict(lambda:np.zeros(3,dtype=float))
    for row,a,b in zip(labels,left,right):
        group=groups[row['id'].rsplit(':',1)[0]]
        for target in row['targets']:
            ma=native_item_metrics(a,target);mb=native_item_metrics(b,target)
            group[0]+=ma['recall@10']-mb['recall@10'];group[1]+=ma['ndcg@10']-mb['ndcg@10'];group[2]+=1
    values=np.array(list(groups.values()));n=len(values)
    rng=np.random.default_rng(seed);indices=rng.integers(0,n,size=(replicates,n));totals=values[indices].sum(axis=1)
    delta=values[:,:2].sum(axis=0)/values[:,2].sum()
    ci=np.quantile(totals[:,:2]/totals[:,2,None],[.025,.975],axis=0)
    return {'dialogue_clusters':n,'item_targets':int(values[:,2].sum()),'replicates':replicates,'seed':seed,
            'recall@10_delta':float(delta[0]),'recall@10_cluster95_interval':ci[:,0].tolist(),
            'ndcg@10_delta':float(delta[1]),'ndcg@10_cluster95_interval':ci[:,1].tolist(),
            'interpretation':'exploratory paired percentile bootstrap; repeated development sample, no multiplicity adjustment or confirmatory claim'}


def run(root=RESULT):
    plan=json.loads((root/'plan.json').read_text());rows=[json.loads((root/'cases'/('case_'+str(i)+'.json')).read_text()) for i in range(len(plan['ids']))]
    sealed=json.loads((root/'generation_complete.json').read_text())
    assert sealed['case_hashes']==[fingerprint(r) for r in rows]
    if [r['id'] for r in rows]!=plan['ids']:raise ValueError('case ID order mismatch')
    if sha(OUT/'single_llm.inputs.json')!=plan['inputs_sha256']:raise ValueError('serving source changed')
    if sha(OUT/'single_llm/predictions.json')!=plan['parent_predictions_sha256']:raise ValueError('parent baseline changed')
    for name,h in plan['source_sha256'].items():
        if sha(root/'source_snapshot'/name)!=h:raise ValueError('source snapshot mismatch')
    labels=json.loads((OUT/'single_llm.labels.json').read_text())
    if [r['id'] for r in labels]!=plan['ids']:raise ValueError('label ID mismatch')
    rankings={m:[r['methods'][m]['ranking'] for r in rows] for m in METHODS}
    inputs=json.loads((root/'inputs.json').read_text())
    if fingerprint(inputs)!=fingerprint(json.loads((OUT/'single_llm.inputs.json').read_text())):raise ValueError('input copy mismatch')
    rankings['retrieval']=[[int(m['movie_id']) for m in r['candidates']] for r in inputs]
    parent=json.loads((OUT/'single_llm/predictions.json').read_text());rankings['single1']=[r['ranking'] for r in parent]
    pairs=[('game_review','fixed_review'),('game_review','independent6'),('game_review','single6'),
           ('game_review','independent3'),('independent6','retrieval'),('single6','single1'),('independent3','retrieval')]
    out={'analysis_source_sha256':sha(Path(__file__)),'comparisons':{a+'_minus_'+b:cluster_difference(labels,rankings[a],rankings[b]) for a,b in pairs},
         'physical':audit(root),'cases':len(rows),'disjoint_pair_locality_checked':True,
         'retrospective_diagnostics':{},'scope':'same32 development contexts, all failures primary zero'}
    budgets={}
    for method in METHODS:
        usage={'prompt_tokens':0,'completion_tokens':0};attempts=0;unknown=0
        for i,row in enumerate(rows):
            paths=[OUT/'single_llm/slots'/('native_'+str(i)+'.raw.json')]
            suffixes=['single'] if method=='single6' else ['initial']
            if method=='independent6':suffixes.append('independent')
            if method in ('fixed_review','game_review'):suffixes.append(method)
            for suffix in suffixes:
                paths.extend((root/'slots').glob('c'+str(i)+'_'+suffix+'_*.raw.json'))
                pending=list((root/'slots').glob('c'+str(i)+'_'+suffix+'_*.intent.json'))
                attempts+=len(pending);unknown+=len(pending)
            for path in paths:
                wrapper=json.loads(path.read_text());raw=wrapper['raw'];attempts+=1
                if raw['status_code']==200:
                    body=json.loads(raw['body']);u=body.get('usage',{})
                    if any(k not in u for k in usage):unknown+=1
                    for k in usage:usage[k]+=u.get(k,0)
                else:unknown+=1
        budgets[method]={'logical_physical_attempts_including_invalid':attempts,'known_usage':usage,'unknown_usage_responses':unknown,
           'configured_logical_cost_cny':(usage['prompt_tokens']*.15+usage['completion_tokens']*1.5)/1e6,
           'scope':'shared prerequisites charged once per arm for fair method cost, not summed as actual provider bill'}
    out['audited_method_budgets']=budgets
    out['physical']['new_physical_attempts_including_unknown']=out['physical']['new_physical_requests']+out['physical']['pending_intents']
    out['physical']['unknown_intents']=[p.name for p in (root/'slots').glob('*.intent.json')]
    parent_cost=json.loads((OUT/'single_llm/summary.json').read_text())['known_response_cost_cny']
    stopped=json.loads((root.parent/'native_consensus_v1/stopped_early.json').read_text())
    bridge=audit(root.parent/'native_consensus_v2')
    out['cumulative_collection_cost']={'parent32_configured_cny':parent_cost,'one_source_engineering_batch_known_cny':bridge['configured_price_cny'],'stopped_first_contract_known_cny':stopped['configured_price_cny'],
        'revised_new_known_cny':out['physical']['configured_price_cny'],
        'sum_known_cny':parent_cost+stopped['configured_price_cny']+bridge['configured_price_cny']+out['physical']['configured_price_cny'],
        'original_unknown_intents':stopped['pending_intents_preserved'],'current_unknown_intents':[p.name for p in (root/'slots').glob('*.intent.json')],'unknown_billing_not_zero':True}
    reserve=0.0
    for name in stopped['pending_intents_preserved']:
        req=json.loads((root.parent/'native_consensus_v1/slots'/name).read_text())['request']
        reserve+=(AttemptClient.upper(req)*.15+req['max_tokens']*1.5)/1e6
    for p in (root/'slots').glob('*.intent.json'):
        req=json.loads(p.read_text())['request'];reserve+=(AttemptClient.upper(req)*.15+req['max_tokens']*1.5)/1e6
    out['cumulative_collection_cost']['unknown_intent_configured_reserve_cny']=reserve
    games=[r['methods']['game_review'].get('game') for r in rows]
    games=[g for g in games if g is not None]
    out['central_proxy_comparison']={
        'contexts':len(games),
        'identical_action_profiles':sum(g['actions']==g['central_optimum_actions'] for g in games),
        'identical_selected_query_sets':sum(set(g['actions'])-{None}==set(g['central_optimum_actions'])-{None} for g in games),
        'zero_proxy_value_gap_contexts':sum(abs(g['central_potential_gap'])<=1e-10 for g in games),
        'meaning':'computed allocation diagnostic, not extra centralized LLM arm; equal proxy utility is not identical realized ranking'}
    for method in ('fixed_review','game_review'):
        commits=[];abstain=0;changed=0;initial_agreed=0;initial_diverse=0
        for r in rows:
            initial=r['initial']
            if len(initial)==3:
                initial_diverse+=len({tuple(c['ranking']) for c in initial})>1
                initial_agreed+=len({c['ranking'][0] for c in initial})==1
            arm=r['methods'][method];changed+=arm.get('review_changes',False)
            for e in arm.get('events',[]):
                if e['valid']:
                    bound=e['bound_selection'];abstain+=bound['abstained']
                    if not bound['abstained']:commits.extend(bound['bound_sources'])
        out['retrospective_diagnostics'][method]={'non_abstained_source_choices':len(commits),'abstained_reviews':abstain,'changed_contexts':changed,
             'selected_source_uses':{use:sum(c['allowed_use']==use for c in commits) for use in sorted({c['allowed_use'] for c in commits})},
             'initial_contexts_with_nonidentical_rankings':initial_diverse,'initial_contexts_with_unanimous_top1':initial_agreed,
             'source_identity_not_entailment':True}
    ceilings={'total_item_targets':sum(len(r['targets']) for r in labels),'pool100_covered_targets':0,'initial3_union_covered_targets':0,
              'oracle_scope':'retrospective labels-only diagnostic, not runnable prediction or baseline',
              'selected_pair_action_oracle_hits@10':{m:0 for m in ('fixed_review','game_review')}}
    for i,(r,label,inp) in enumerate(zip(rows,labels,inputs)):
        pool=[m['movie_id'] for m in inp['candidates']]
        ceilings['pool100_covered_targets']+=sum(str(t) in pool for t in label['targets'])
        union={int(m) for call in r['initial'] for m in call['ranking']}
        ceilings['initial3_union_covered_targets']+=sum(t in union for t in label['targets'])
        if len(r['initial'])!=3:continue
        vectors=[scores(call['ranking'],pool) for call in r['initial']]
        for method in ('fixed_review','game_review'):
            commit=json.loads((root/'commitments'/('c'+str(i)+'_'+method+'.json')).read_text())
            targets=[commit['targets'][q] for player,q in commit['allocation']]
            best=0
            for choices in product((0,1),repeat=len(targets)):
                state=copy.deepcopy(vectors)
                for target,choice in zip(targets,choices):state=local_update(state,target,target['b'] if choice else target['a'],fingerprint(state))
                ranking=readout(state,pool)
                best=max(best,sum(t in ranking[:10] for t in label['targets']))
            ceilings['selected_pair_action_oracle_hits@10'][method]+=best
    out['retrospective_ceilings']=ceilings
    create(root/'paired_analysis.json',out);print(json.dumps(out,indent=2));return out
if __name__=='__main__':run()
