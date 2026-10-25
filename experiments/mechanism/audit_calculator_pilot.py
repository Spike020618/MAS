"""Recompute acceptance and finite-game certificates from frozen pilot logs."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.mechanism.solve_finite_stackelberg import solve
from experiments.mechanism.pilot_calculator_contract import balances
from experiments.mechanism.pilot_verified_contract import settle, expected_payoffs


def main():
    p=argparse.ArgumentParser()
    p.add_argument('directory',type=Path)
    args=p.parse_args()
    directory=args.directory
    data=json.loads((directory/'summary.json').read_text())
    calls=[json.loads(x) for x in (directory/'calls.jsonl').read_text().splitlines()]
    rows=data['outcomes']
    errors=[]
    keys=[x['key'] for x in calls]
    if len(set(keys))!=len(keys): errors.append('duplicate API call keys')
    for call in calls:
        digest=hashlib.sha256(json.dumps(call['messages'],ensure_ascii=False).encode()).hexdigest()
        if digest!=call['prompt_hash']: errors.append('prompt digest mismatch')
        if call['trace'].get('status')!=200: errors.append('non-200 API call')
        for field in ['prompt_tokens','completion_tokens','latency_s']:
            if not isinstance(call['trace'].get(field),(int,float)) or call['trace'][field]<0:
                errors.append('invalid API cost/latency')
    for row in rows:
        if not row.get('valid'): errors.append('invalid trial'); continue
        expected=balances(row['reward'],row['cost'])
        if row['expected_balances']!=expected: errors.append('incorrect expected balances')
        settlement=settle(row['report']['C'],row['truth'],row['action'],row['reward'],20,row['cost'])
        if settlement!=row['settlement']: errors.append('incorrect settlement')
    dev=set(data['protocol']['dev_costs'])
    val=set(data['protocol']['validation_costs'])
    if dev & val: errors.append('development/validation parameter overlap')
    protocol=data['protocol']
    n_expected=(len(protocol['dev_costs'])*len(protocol['reward_multipliers'])*len(protocol['modes'])
                +len(protocol['validation_costs'])*len(protocol['reward_multipliers'])*len(protocol['modes'])*len(protocol['validation_personas']))
    if len(rows)!=n_expected or len(calls)!=2*n_expected:
        errors.append('unexpected experimental size')
    certificates=[]
    for cost in sorted(val):
        candidates=sorted({r['reward'] for r in rows if r['phase']=='validation' and r['cost']==cost})
        game={'actions':['low','verify'], 'payoff_source':'analytic fair-bit assumptions; known synthetic probabilities',
              'contracts':[{'id':f'r{r:g}','payoffs':expected_payoffs(r,cost)} for r in candidates]}
        solution=solve(game)
        selected=solution['solutions']['optimistic']
        selected_reward=next(r for r in candidates if f'r{r:g}'==selected['contract_id'])
        observed=[r for r in rows if r['phase']=='validation' and r['mode']=='calculator'
                  and r['cost']==cost and r['reward']==selected_reward]
        agreement=all(r['action']==selected['action'] for r in observed) and len(observed)==2
        certificates.append({'cost':cost,'game':game,'solution':solution,
                             'selected_contract_observed_action_agreement':agreement,
                             'interpretation':'finite synthetic pure-commitment model; not real-QA equilibrium'})
    metrics={}
    for mode in ['unaided','calculator']:
        subset=[r for r in rows if r['phase']=='validation' and r['mode']==mode]
        matches=sum(r.get('valid',False) and r['action']==r['predicted_best'] for r in subset)
        regret=sum(r.get('expected_regret',0) for r in subset)/len(subset)
        pairs={}
        for row in subset: pairs.setdefault((row['cost'],row['persona']),[]).append(row)
        switched=sum(len(v)==2 and all(r.get('valid',False) and r['action']==r['predicted_best'] for r in v)
                     for v in pairs.values())
        verified=[r for r in subset if r.get('action')=='verify']
        metrics[mode]={'n':len(subset),'best_response_count':matches,'best_response_rate':matches/len(subset),
                       'n_pairs':len(pairs),'correct_switch_pairs':switched,'switch_rate':switched/len(pairs),
                       'mean_expected_regret':regret,'verified_reports':len(verified),
                       'verified_correct':sum(r['settlement']['correct'] for r in verified)}
    verified_ok=metrics['calculator']['verified_reports']>0 and metrics['calculator']['verified_reports']==metrics['calculator']['verified_correct']
    gate=not errors and metrics['calculator']['best_response_rate']>=0.9 and metrics['calculator']['switch_rate']>=0.8 and verified_ok
    report={'data_integrity_pass':not errors,'errors':errors,'assisted_gate_pass':gate,
            'metrics':metrics,'finite_synthetic_game_certificates':certificates,
            'api_calls':len(calls),
            'total_tokens':sum(x['trace']['prompt_tokens']+x['trace']['completion_tokens'] for x in calls),
            'api_latency_sum_s':sum(x['trace']['latency_s'] for x in calls),
            'hashes':{name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
                      for name in ['protocol.json','private_cases.json','calls.jsonl','summary.json']}}
    (directory/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['hashes','finite_synthetic_game_certificates']},indent=2))


if __name__=='__main__': main()
