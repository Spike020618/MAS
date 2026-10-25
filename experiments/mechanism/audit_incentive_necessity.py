"""Offline incentive-necessity audit; no API, learned policy, or gold-based BR.

Candidate report deviations hold recorded subjective beliefs fixed and expose
other sealed reports. This is a proposed report-stage scenario, NOT observed
strategic LLM behavior or a posterior-update model.
"""
from fractions import Fraction
import hashlib
from itertools import product
import json
from pathlib import Path

from experiments.mechanism import probe_resource_allocation as p
from experiments.mechanism import probe_probability_routing as probability
from experiments.mechanism.audit_probability_compatibility import compatible_decode
from experiments.mechanism.replay_low_cost_router import vote
from experiments.mechanism.run_upgrade_pilot import freeze,read_rows
from experiments.mechanism.audit_contract_game_alignment import audit as legacy_audit

OUT=p.ROOT/'experiments/results/audit/incentive_necessity_2026-10-23'
RULES=['individual_absolute','team_brier_difference','team_accuracy_difference']


def simplex(q):
    return (isinstance(q,list) and len(q)>=2 and all(type(v) in [int,float] and 0<=v<=1 for v in q)
            and sum(Fraction(str(v)) for v in q)==1)


def score(q,y):return 1-.5*sum((v-int(j==y))**2 for j,v in enumerate(q))


def expected_score(belief,report):
    return sum(q*score(report,y) for y,q in enumerate(belief))


def mean_report(reports):return [sum(r[j] for r in reports)/len(reports) for j in range(len(reports[0]))]


def projection(v):
    """Euclidean projection to simplex; continuous best response for mean Brier."""
    ordered=sorted(v,reverse=True);running=0;theta=0
    for j,value in enumerate(ordered,1):
        running+=value
        candidate=(running-1)/j
        if value>candidate:theta=candidate
    result=[max(value-theta,0.) for value in v]
    if abs(sum(result)-1)>1e-8:raise ValueError('projection failed')
    return result


def prediction(q,choices,preferred=None):
    winners=[i for i,v in enumerate(q) if v>=max(q)-1e-12]
    if preferred in choices and choices.index(preferred) in winners:return preferred
    return choices[winners[0]]


def expected_component(rule,belief,reports,votes,agent,candidate,conclusion,choices):
    updated=list(reports);updated[agent]=candidate
    if rule=='individual_absolute':return expected_score(belief,candidate)
    if rule=='team_brier_difference':return expected_score(belief,mean_report(updated))
    if rule=='team_accuracy_difference':
        cv=list(votes);cv[agent]=conclusion;result=vote(cv,choices)
        return belief[choices.index(result)] if result in choices else 0.
    raise ValueError('unknown payment rule')


def candidates(belief,reports,votes,agent,choices):
    # All choices depend only on beliefs/visible reports, never an outcome label.
    result=[('honest',belief,votes[agent])]
    for j,choice in enumerate(choices):result.append(('one_hot:'+choice,[float(i==j) for i in range(len(choices))],choice))
    for j,report in enumerate(reports):
        if j!=agent:result.append(('copy:'+str(j),report,votes[j]))
    uniform=[1/len(choices)]*len(choices)
    result.append(('uniform',uniform,prediction(uniform,choices,votes[agent])))
    target=[len(reports)*belief[j]-sum(r[j] for i,r in enumerate(reports) if i!=agent) for j in range(len(choices))]
    br=projection(target)
    result.append(('mean_brier_continuous_BR',br,prediction(br,choices,votes[agent])))
    return result


def best_response(rule,belief,reports,votes,agent,choices):
    options=candidates(belief,reports,votes,agent,choices)
    honest=expected_component(rule,belief,reports,votes,agent,belief,votes[agent],choices)
    best=options[0];value=honest
    for candidate in options[1:]:
        current=expected_component(rule,belief,reports,votes,agent,candidate[1],candidate[2],choices)
        if current>value+1e-10:best=candidate;value=current
    return {'strategy':best[0],'report':best[1],'C':best[2],'expected_component_gain':value-honest}


def load_batch(name,directory,labels_path):
    rows=read_rows(directory/'responses.jsonl')
    if len(rows)!=144 or len({r['key'] for r in rows})!=144:raise ValueError('complete distinct 144-response source required')
    result=json.loads((directory/'result.json').read_text())
    if result.get('response_hash')!=p.digest(rows):raise ValueError('source response commitment changed')
    for row in rows:
        if row['response_hash']!=p.digest(row['response']) or row['request_hash']!=p.digest(row['request']):
            raise ValueError('raw response/request changed')
    cases=json.loads((directory/'public_cases.json').read_text());index={r['key']:r for r in rows}
    # Outcomes are evaluator-only; never passed to best_response or candidates.
    labels={r['id']:r['label'] for r in json.loads(labels_path.read_text())}
    parsed=[]
    for case in cases:
        reports={};costs={};votes={}
        for backend in ['zhipu_flash','deepseek']:
            rs=[];cs=[];vs=[]
            for agent in range(3):
                row=index[probability.key(case,agent,backend)];body=p.body(row)
                decoded=compatible_decode(body['choices'][0]['message']['content'],case['choices'])
                rs.append([decoded['probabilities'][c] for c in case['choices']] if decoded['probabilities'] is not None else None)
                cs.append(body['usage']['total_tokens']);vs.append(decoded['answer'] or 'abstain')
            reports[backend]=rs;costs[backend]=cs;votes[backend]=vs
        parsed.append({'id':case['id'],'choices':case['choices'],'reports':reports,'tokens':costs,'votes':votes,
            'label':labels[case['id']],'batch':name})
    return parsed,{'responses_sha256':hashlib.sha256((directory/'responses.jsonl').read_bytes()).hexdigest(),
        'labels_sha256':hashlib.sha256(labels_path.read_bytes()).hexdigest(),'n':24}


def reporting_audit(rows):
    output={};details=[];eligibility={}
    for row in rows:
        choices=row['choices'];y=choices.index(row['label'])
        for backend,reports in row['reports'].items():
            eligible=all(q is not None and simplex(q) for q in reports)
            tag=row['batch']+':'+backend
            coverage=eligibility.setdefault(tag,{'panels_total':0,'panels_exact_simplex':0,'excluded_no_imputation':0})
            coverage['panels_total']+=1;coverage['panels_exact_simplex']+=int(eligible);coverage['excluded_no_imputation']+=int(not eligible)
            if not eligible:continue
            votes=row['votes'][backend];initial_correct=int(vote(votes,choices)==row['label']);initial_score=score(mean_report(reports),y)
            for rule in RULES:
                summary=output.setdefault(tag+':'+rule,{'panels':0,'agent_slots':0,'profitable_deviations':0,
                    'profitable_and_majority_harm':0,'profitable_and_team_brier_harm':0,
                    'majority_corrections':0,'max_expected_component_gain':0.,'max_individual_absolute_regret':0.})
                summary['panels']+=1
                for agent in range(3):
                    # No y enters this decision. Gold used only after the candidate is fixed.
                    br=best_response(rule,reports[agent],reports,votes,agent,choices)
                    summary['agent_slots']+=1;gain=br['expected_component_gain']
                    summary['max_expected_component_gain']=max(summary['max_expected_component_gain'],gain)
                    if rule=='individual_absolute':summary['max_individual_absolute_regret']=max(summary['max_individual_absolute_regret'],gain)
                    if gain<=1e-10:continue
                    updated=list(reports);updated[agent]=br['report'];cv=list(votes);cv[agent]=br['C']
                    changed_correct=int(vote(cv,choices)==row['label']);changed_score=score(mean_report(updated),y)
                    harm=initial_correct>changed_correct;brier_harm=changed_score<initial_score-1e-10
                    summary['profitable_deviations']+=1;summary['profitable_and_majority_harm']+=int(harm)
                    summary['profitable_and_team_brier_harm']+=int(brier_harm)
                    summary['majority_corrections']+=int(changed_correct>initial_correct)
                    details.append({'id':row['id'],'batch':row['batch'],'backend':backend,'rule':rule,'agent':agent,
                        'strategy':br['strategy'],'expected_component_gain':gain,'before_C':votes[agent],'after_C':br['C'],
                        'majority_harm':bool(harm),'team_brier_harm':bool(brier_harm),
                        'before_majority_correct':initial_correct,'after_majority_correct':changed_correct})
    return {'coverage':eligibility,'summaries':output,'details':details,
        'interpretation':'hypothetical unilateral report-stage deviations with fixed subjective q and visible peers; NOT observed LLM cheating or realized equilibrium',
        'payment_scaling':'individual gain times B/3; team difference gain times B/6, costs unchanged for report-only deviations; sign independent of B>0',
        'limits':['original independent model runs did not reveal peer reports; exposure is a new audited scenario',
            'q is self-reported subjective belief, not a proved posterior, calibrated truth or belief after seeing peers',
            'one deviation at a time, no repeated best-response or endogenous peer reaction simulation',
            'eligible exact-simplex subset only, no silent normalization; counts are correlated agent slots',
            'absolute own score removes this fixed-q distortion incentive but has separable report payoffs, not new coordinated mechanism']}


def effort_audit(rows):
    records=[];role_stats={}
    for row in rows:
        choices=row['choices'];y=choices.index(row['label'])
        for agent in range(3):
            low=row['reports']['zhipu_flash'][agent];high=row['reports']['deepseek'][agent]
            if low is None or high is None or not simplex(low) or not simplex(high):continue
            delta=score(high,y)-score(low,y);tokens=row['tokens']['deepseek'][agent]
            records.append({'batch':row['batch'],'id':row['id'],'agent':agent,'realized_score_delta':delta,
                'incremental_high_tokens':tokens,
                'realized_budget_break_even_token_equivalent':3*tokens/delta if delta>1e-12 else None})
    # Calibration/test roles fixed by source batch, not by outcomes; no reward chosen.
    for agent in range(3):
        train=[r for r in records if r['batch']=='opened24' and r['agent']==agent]
        test=[r for r in records if r['batch']=='screened24' and r['agent']==agent]
        def stats(rs):
            return {'n':len(rs),'mean_realized_score_delta':sum(r['realized_score_delta'] for r in rs)/len(rs) if rs else None,
                'mean_high_tokens':sum(r['incremental_high_tokens'] for r in rs)/len(rs) if rs else None,
                'positive_score_deltas':sum(r['realized_score_delta']>1e-12 for r in rs),
                'nonpositive_score_deltas':sum(r['realized_score_delta']<=1e-12 for r in rs)}
        a,b=stats(train),stats(test);delta=a['mean_realized_score_delta']
        role_stats[str(agent)]={'opened_calibration':a,'screened_now_open_audit':b,
            'calibration_expected_budget_break_even':3*a['mean_high_tokens']/delta if delta is not None and delta>0 else None}
    return {'n_paired_exact_simplex_slots':len(records),'role_stats':role_stats,'records':records,
        'equation':'Delta u = (B/3)*Delta S - additional_high_tokens; initial low tokens sunk but charged to deployment',
        'cost_unit':'logged additional tokens; B is token-equivalent virtual credit, NOT API money',
        'no_selected_budget':True,'no_optimal_contract_or_ne_claim':True,
        'limits':['per-question realized break-even uses evaluator outcome and is not available for online action selection',
            'mean-by-role estimates do not establish conditional value of information or optimal per-question actions',
            'a positive average score delta permits a stylized incentive threshold but does not prove truthful effort choice',
            'no shared-budget allocation or auction implemented; fixed B/3 scoring shares are separable',
            'screened24 has now been opened by previous validation; this audit is not new independent evidence']}


def sandbagging_fixture():
    honest=[.8,.2];fake=[.5,.5];final=[.9,.1]
    direct=expected_score(honest,final)-expected_score(honest,honest)
    manipulated=expected_score(honest,final)-expected_score(honest,fake)
    return {'constructed_not_model_behavior':True,'belief':honest,'final_report_fixed':final,
        'expected_delta_score_truthful_initial':direct,'expected_delta_score_fake_initial':manipulated,
        'profit_from_lowering_initial_report':manipulated-direct,
        'absolute_final_score_unchanged':True,'conclusion':'raw before-after improvement reward invites baseline sandbagging'}


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    first=probability.OUT
    second=p.ROOT/'experiments/results/resource_allocation/routing_holdout_validation_v1'
    a,pa=load_batch('opened24',first,p.ROOT/'experiments/results/upgrade_game/p2_data_v2/development_private.json')
    b,pb=load_batch('screened24',second,p.ROOT/'experiments/results/resource_allocation/routing_holdout_data_v1/private_labels.json')
    if {r['id'] for r in a}&{r['id'] for r in b}:raise ValueError('batches overlap')
    freeze(OUT/'audit_scope.json',{'version':'incentive-necessity-v1','no_api_calls':True,
        'code_hash':p.digest(Path(__file__).read_text()),'sources':{'opened24':pa,'screened24':pb},
        'not_preregistration':'both source batches already opened; this is necessary-condition audit, not confirmation',
        'reference_rules':RULES,'no_reward_tuning':True,'no_automatic_paid_experiment':True})
    old=legacy_audit()
    if not old['passed_execution_replay']:raise ValueError('legacy settlement audit failed')
    result={'scope':'offline necessary-condition audit only, NOT new effective mechanism or strategic LLM experiment',
        'api_calls_added':0,'source_question_count':48,'legacy_contract':old,
        'reporting':reporting_audit(a+b),'effort':effort_audit(a+b),
        'before_after_reward_counterexample':sandbagging_fixture(),
        'decision':'do not add payments merely to obtain a game; identify a feasible nonseparable incentive problem before model calls',
        'open_gaps':['actual strategic behavior not observed','token-to-credit value not externally specified',
            'q calibration/posterior assumption unproved','shared-budget incentive compatibility unresolved',
            'no evidence these payment candidates improve true group quality-cost over simple routing']}
    freeze(OUT/'result.json',result)
    print(json.dumps({'api_calls_added':0,'reporting_summaries':result['reporting']['summaries'],
        'reporting_coverage':result['reporting']['coverage'],'role_stats':result['effort']['role_stats'],
        'sandbagging_fixture':result['before_after_reward_counterexample']},ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':run()
