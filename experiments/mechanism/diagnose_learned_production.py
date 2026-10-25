"""Post-seal statistics, deviations and controller identity; never refits."""
from math import comb
from random import Random
import json
from experiments.core.mechanism import learned_production_game as g
from experiments.mechanism import run_learned_production as run
from experiments.mechanism.run_upgrade_pilot import freeze


def paired(binary_a,binary_b):
    b=sum(x and not y for x,y in zip(binary_a,binary_b))
    c=sum(y and not x for x,y in zip(binary_a,binary_b)); n=b+c
    return {'corrections':b,'harms':c,'exact_mcnemar_p':min(1.,2*sum(comb(n,i) for i in range(min(b,c)+1))/2**n) if n else 1.}


def interval(values):
    rng=Random(20261025);n=len(values)
    draws=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(10000))
    return {'mean':sum(values)/n,'bootstrap_percentile_95':[draws[249],draws[9749]],
            'scope':'descriptive paired question bootstrap, very small sample'}


def main():
    out=run.OUT
    result=json.loads((out/'result.json').read_text());rows=result['rows']
    comparisons={}
    for baseline in ('centralized_erm','fixed00','fixed01','fixed10','fixed11'):
        comparisons[baseline]={'correctness':paired([bool(r['methods']['game']['correct']) for r in rows],
                                                    [bool(r['methods'][baseline]['correct']) for r in rows]),
                              'objective_difference':interval([r['methods']['game']['objective']-r['methods'][baseline]['objective'] for r in rows])}
    keys=sorted(comparisons,key=lambda k:comparisons[k]['correctness']['exact_mcnemar_p'])
    previous=0.
    for i,k in enumerate(keys):
        previous=max(previous,min(1.,(len(keys)-i)*comparisons[k]['correctness']['exact_mcnemar_p']))
        comparisons[k]['correctness']['holm_p']=previous
    outputs=json.loads((out/'test_sealed_outputs.json').read_text())
    decisions=json.loads((out/'test_sealed_readouts.json').read_text())
    labels=json.loads((out/'test_labels.json').read_text())
    policy=json.loads((out/'frozen_policy.json').read_text())
    groups={};regrets={}
    for cid,s in outputs.items():
        state=g.context({k:s[k] for k in ('fast0','fast1','fast2')})
        state=state if state in policy['contexts'] else 'global'
        groups.setdefault(state,[]).append(cid)
    for state,ids in groups.items():
        profile=tuple(policy['contexts'][state]['game']['profile']);cell=[]
        for player in range(2):
            other=list(profile);other[player]=1-other[player]
            gains=[]
            for cid in ids:
                slots=outputs[cid];correct=int(g.outcome(slots,profile)==labels[cid])
                alternative=int(g.outcome(slots,other)==labels[cid])
                cost=float(g.COST_PER_TOKEN)*slots['think'+str(player+1)]['tokens']
                gains.append(alternative-correct-cost*(other[player]-profile[player]))
            cell.append({'player':player,'mean_alternative_gain':sum(gains)/len(gains),
                         'positive_deviation':max(0.,sum(gains)/len(gains))})
        regrets[state]={'n':len(ids),'deviations':cell}
    capability={}
    for split in ('train','test'):
        ss=json.loads((out/(split+'_sealed_outputs.json')).read_text())
        yy=json.loads((out/(split+'_labels.json')).read_text())
        details=[]
        for cid,s in ss.items():
            correct={''.join(map(str,p)):g.outcome(s,p)==yy[cid] for p in g.PROFILES}
            details.append({'id':cid,'profile_correct':correct,
                'single_fast0_correct':s['fast0']['answer']==yy[cid],
                'single_think1_correct':s['think1']['answer']==yy[cid],
                'initial_unanimous':g.context({k:s[k] for k in ('fast0','fast1','fast2')})=='unanimous',
                'upgrade_supplies_missing_answer':not any(s['fast'+str(i)]['answer']==yy[cid] for i in range(3))
                    and any(s['think'+str(i)]['answer']==yy[cid] for i in (1,2))})
        capability[split]={'n':len(details),
            'profile_correct':{p:sum(r['profile_correct'][p] for r in details) for p in ('00','01','10','11')},
            'single_fast0_correct':sum(r['single_fast0_correct'] for r in details),
            'single_think1_correct':sum(r['single_think1_correct'] for r in details),
            'upgrades_supply_initially_missing_answer_cases':sum(r['upgrade_supplies_missing_answer'] for r in details),
            'only_joint_upgrade_corrects':sum(not r['profile_correct']['00'] and not r['profile_correct']['01']
                                            and not r['profile_correct']['10'] and r['profile_correct']['11'] for r in details),
            'all_upgrade_vs_no_upgrade':paired([r['profile_correct']['11'] for r in details],
                                              [r['profile_correct']['00'] for r in details]),
            'rows':details}
    diagnostic={'comparisons':comparisons,'test_empirical_deviations':regrets,'resource_capability':capability,
        'game_vs_erm_same_profile_cases':sum(r['profiles']['game']==r['profiles']['centralized_erm'] for r in decisions.values()),
        'game_selected_upgrade_cases':sum(sum(r['profiles']['game'])>0 for r in decisions.values()),
        'test_context_counts':{k:len(v) for k,v in groups.items()},
        'limits':['regret on held-out realized outputs is descriptive, not a population equilibrium certificate',
                  'no adaptation after test labels; no cost tuning; no extra API calls',
                  'policy equality would negate independent game-value claim even if both beat fixed controls']}
    freeze(out/'diagnosis.json',diagnostic)
    print(json.dumps(diagnostic,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
