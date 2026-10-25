"""All saved K/U profiles, distinct from observed autonomous game play."""
from itertools import product
import json

from experiments.mechanism import audit_incentive_necessity as audit
from experiments.mechanism import probe_resource_allocation as p
from experiments.mechanism.run_upgrade_pilot import freeze
from experiments.mechanism.replay_low_cost_router import vote


def no_pay_edges(vote_paths,high_costs,choices,label):
    """Own high-token cost model, with low acquisition already sunk."""
    if len(high_costs)!=3 or any(c<=0 for c in high_costs):raise ValueError('three positive own costs required')
    result=[]
    for profile in product((0,1),repeat=3):
        before=[vote_paths[a][i] for i,a in enumerate(profile)]
        for i,a in enumerate(profile):
            if not a:continue
            alternative=list(profile);alternative[i]=0
            after=[vote_paths[b][j] for j,b in enumerate(alternative)]
            result.append({'profile':list(profile),'agent':i,'profit_tokens_if_agent_bears_cost':high_costs[i],
                'before_correct':int(vote(before,choices)==label),'after_correct':int(vote(after,choices)==label)})
    return result


def run():
    first=audit.probability.OUT
    second=p.ROOT/'experiments/results/resource_allocation/routing_holdout_validation_v1'
    a,pa=audit.load_batch('opened24',first,p.ROOT/'experiments/results/upgrade_game/p2_data_v2/development_private.json')
    b,pb=audit.load_batch('screened24',second,p.ROOT/'experiments/results/resource_allocation/routing_holdout_data_v1/private_labels.json')
    resources={};mixed={};examples=[];coverage={}
    for row in a+b:
        batch=row['batch'];paths=[row['votes'][b] for b in ['zhipu_flash','deepseek']]
        costs=row['tokens']['deepseek']
        edges=no_pay_edges(paths,costs,row['choices'],row['label'])
        resource=resources.setdefault(batch,{'questions':0,'profiles':0,'profitable_keep_deviations':0,
            'profitable_keep_and_majority_harm':0,'all_high_keep_harm':0,
            'keep_baseline_correct':0,'all_high_correct':0})
        resource['questions']+=1;resource['profiles']+=8;resource['profitable_keep_deviations']+=len(edges)
        resource['keep_baseline_correct']+=int(vote(paths[0],row['choices'])==row['label'])
        resource['all_high_correct']+=int(vote(paths[1],row['choices'])==row['label'])
        for edge in edges:
            harm=edge['before_correct']>edge['after_correct']
            resource['profitable_keep_and_majority_harm']+=int(harm)
            resource['all_high_keep_harm']+=int(harm and edge['profile']==[1,1,1])
            if harm:examples.append({'id':row['id'],'batch':batch,**edge})
        counts=coverage.setdefault(batch,{'profiles':0,'exact_simplex_profiles':0,'excluded':0})
        for profile in product((0,1),repeat=3):
            counts['profiles']+=1
            reports=[row['reports']['deepseek' if v else 'zhipu_flash'][i] for i,v in enumerate(profile)]
            if not all(r is not None and audit.simplex(r) for r in reports):counts['excluded']+=1;continue
            counts['exact_simplex_profiles']+=1
            votes=[paths[v][i] for i,v in enumerate(profile)];choices=row['choices'];y=choices.index(row['label'])
            before_correct=int(vote(votes,choices)==row['label']);before_score=audit.score(audit.mean_report(reports),y)
            for rule in audit.RULES:
                stat=mixed.setdefault(batch+':'+rule,{'slots':0,'profitable':0,'majority_harms':0,'brier_harms':0,'majority_corrections':0})
                for agent in range(3):
                    stat['slots']+=1;br=audit.best_response(rule,reports[agent],reports,votes,agent,choices)
                    if br['expected_component_gain']<=1e-10:continue
                    rs=list(reports);rs[agent]=br['report'];cv=list(votes);cv[agent]=br['C']
                    correct=int(vote(cv,choices)==row['label']);score=audit.score(audit.mean_report(rs),y)
                    stat['profitable']+=1;stat['majority_harms']+=int(before_correct>correct)
                    stat['brier_harms']+=int(score<before_score-1e-10);stat['majority_corrections']+=int(correct>before_correct)
    output={'scope':'exhaustive saved-profile counterfactual audit, not repeated experimental samples or observed strategic actions',
        'api_calls_added':0,'resource_scenario':'new assumption: independent providers bear own incremental tokens and no payment; project currently bears actual API costs',
        'cost_and_credit_not_money':True,'no_selected_payment_budget':True,
        'resource_results':resources,'resource_harm_examples':examples,'reporting_profile_coverage':coverage,
        'reporting_profile_results':mixed,
        'sources':{'opened24':pa,'screened24':pb},
        'limits':['all-high-to-keep branches reuse independent request-compatible potential reports, not actual provider deviations',
            'own-cost assumption is not an existing financial transfer or revealed preference',
            'within-question profiles and agent deviations are dependent, not independent n',
            'report-stage subjective-belief and visible-peer assumptions match necessity audit, not original runtime visibility',
            'score harm and majority harm are different metrics; do not substitute one for another']}
    freeze(audit.OUT/'resource_profile_audit.json',output)
    print(json.dumps({k:v for k,v in output.items() if k not in ['resource_harm_examples','sources']},ensure_ascii=False,indent=2))
    return output


if __name__=='__main__':run()
