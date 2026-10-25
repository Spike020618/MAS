"""Independent rational-arithmetic check of proxy-game scope, zero APIs.

No theorem or payment modification. Old benefit and new proxy equilibria are
audited separately. Oracle counts are offline ceilings, never policies.
"""
from fractions import Fraction
from hashlib import sha256
from itertools import product
import json

from experiments.mechanism.attribute_legacy_gain import ROOT, OUT
from experiments.mechanism.audit_contract_game_alignment import audit as old_audit
from experiments.mechanism.repair_upgrade_v3 import load
from experiments.core.mechanism.upgrade_value_v3 import contract_for
from experiments.core.mechanism.upgrade_game import digest
from experiments.core.mechanism.upgrade_protocol_v3 import decode_content, snapshot, request


def exact_check(contract, selected):
    profiles = tuple(product((0,1),repeat=contract.n))
    values = dict(zip(profiles,map(lambda x:Fraction(str(x)),contract.values)))
    costs = [[Fraction(str(x)) for x in pair] for pair in contract.costs]
    budget = Fraction(str(contract.budget)); beta = budget/(2*contract.n)
    def payment(a,i):
        ref = list(a);ref[i]=0
        return beta*(1+values[a]-values[tuple(ref)])
    def utility(a,i):return payment(a,i)-costs[i][a[i]]
    def potential(a):return beta*values[a]-sum(costs[i][a[i]] for i in range(contract.n))
    edges = 0
    for a in profiles:
        payments = [payment(a,i) for i in range(contract.n)]
        if min(payments)<0 or max(payments)>budget/contract.n or sum(payments)>budget:
            raise ValueError('proxy budget bound fails')
        for i in range(contract.n):
            b=list(a);b[i]=1-b[i];b=tuple(b)
            if utility(b,i)-utility(a,i)!=potential(b)-potential(a):
                raise ValueError('exact potential identity fails')
            edges += 1
    a = tuple(selected)
    gains=[]
    for i in range(contract.n):
        b=list(a);b[i]=1-b[i]
        gains.append(utility(tuple(b),i)-utility(a,i))
    return {'profiles':len(profiles),'directed_unilateral_edges':edges,
            'max_selected_regret':float(max(Fraction(0),*gains)),
            'global_proxy_objective_gap':float(max(potential(p) for p in profiles)-potential(a)),
            'weak_keep_participation_shortfall':float(max(Fraction(0),*(beta-costs[i][0]-utility(a,i) for i in range(contract.n)))),
            'arithmetic':'Fractions of logged decimal table/cost strings; no settlement() calls'}


def semantic_scope_fixture(model):
    choices=['yes','no']
    a = [decode_content(json.dumps({'A':'unsupported assumption','E':['unsupported assertion'],'I':'unsupported inference','C':'yes'}),choices) for _ in range(3)]
    b = [decode_content(json.dumps({'A':'different assumption','E':['different evidence packet'],'I':'different inference','C':'yes'}),choices) for _ in range(3)]
    ca,cb = contract_for(model,a),contract_for(model,b)
    return {'constructed_not_model_behavior':True,
            'same_C_patterns_different_AEI_same_contract':ca.commitment()==cb.commitment(),
            'same_actions_same_proxy_payments':ca.settlement((1,0,1))==cb.settlement((1,0,1)),
            'upgrade_prompt_changes_with_AEI':request(snapshot('Fixed public question?',choices,a),0)!=request(snapshot('Fixed public question?',choices,b),0),
            'scope':'AEI can affect report-generation inputs but is not independently verified or valued by this proxy contract'}


def main():
    root=ROOT/'experiments/results/upgrade_game'
    saved=root/'v3_offline_repair_bounds'
    manifest=json.loads((saved/'manifest.json').read_text())
    mismatches=[]
    for path,expected in manifest['code_sha256'].items():
        if sha256((ROOT/path).read_bytes()).hexdigest()!=expected:mismatches.append(path)
    for stage,expected in manifest['source_response_sha256'].items():
        path=root/('p2_'+stage+'_v2')/'responses.jsonl'
        if sha256(path.read_bytes()).hexdigest()!=expected:mismatches.append(str(path))
    if mismatches:raise ValueError('protected source or code mismatch: '+str(mismatches))
    model=json.loads((saved/'model.json').read_text()); result=json.loads((saved/'result.json').read_text())
    if digest(model)!=manifest['model_hash'] or digest(result)!=manifest['result_hash']:
        raise ValueError('saved v3 result/model commitment mismatch')
    rows,source_audit=load(root/'p2_development_v2',root/'p2_data_v2/development_private.json')
    matched=[r for r in rows if r['all_replayable']]
    decisions={r['id']:r['recommendation'] for r in json.loads((saved/'recommendations.json').read_text())}
    checks=[]
    for row in matched:
        decision=decisions[row['id']]
        if digest({k:v for k,v in decision.items() if k!='decision_hash'})!=decision['decision_hash']:
            raise ValueError('recommendation digest mismatch')
        contract=contract_for(model,row['initial'])
        if contract.commitment()!=decision['contract_hash']:raise ValueError('proxy contract mismatch')
        checks.append({'id':row['id'],**exact_check(contract,decision['actions'])})
    oracle = {'n':len(matched),'initial_correct':sum(r['initial_correct'] for r in matched),
        'best_realized_profile_oracle_correct':sum(max(r['scores']) for r in matched),
        'initial_wrong_correctable_by_any_saved_profile':[r['id'] for r in matched if not r['initial_correct'] and max(r['scores'])],
        'profile_variable_cases':[{'id':r['id'],'initial_correct':r['initial_correct'],'scores':r['scores']}
                                  for r in matched if len(set(r['scores']))>1],
        'scope':'offline full 8-profile outcome ceiling on fixed compatible potential reports; NOT expected true value, not deployment or independent test'}
    old=old_audit()
    if not old['passed_execution_replay']:raise ValueError('old settlement replay fails')
    audit={'no_api_calls':True,'old_contract_audit':old,'new_proxy_fraction_checks':checks,
        'new_proxy_summary':{'contracts':len(checks),'directed_edges':sum(x['directed_unilateral_edges'] for x in checks),
            'max_regret':max(x['max_selected_regret'] for x in checks),
            'max_global_gap':max(x['global_proxy_objective_gap'] for x in checks),
            'max_participation_shortfall':max(x['weak_keep_participation_shortfall'] for x in checks)},
        'semantic_scope_fixture':semantic_scope_fixture(model),'realized_profile_oracle':oracle,
        'protected_code_and_response_hashes_pass':True,'source_audit':source_audit,
        'limits':['old static benefit does not validate new proxy contract',
                  'equilibrium is conditional on frozen initial reports and public proxy table, not a truthful-reporting game',
                  'no leader contract optimization or Stackelberg solution verified',
                  'proxy estimation MSE is not a uniform conditional true-value error bound',
                  'current finite-table global selection is equivalent to centralized proxy enumeration',
                  'oracle uses offline labels and cannot be used by action selection',
                  'new-sample/other-model/fresh-regeneration performance is not bounded by this fixed-report oracle'],
        'source_hashes':{str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in
            [saved/'manifest.json',saved/'model.json',saved/'result.json',saved/'recommendations.json']}}
    if max(audit['new_proxy_summary'][k] for k in ['max_regret','max_global_gap','max_participation_shortfall'])>1e-12:
        raise ValueError('proxy equilibrium tolerance failure')
    (OUT/'theory_bridge.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:audit[k] for k in ['new_proxy_summary','semantic_scope_fixture','realized_profile_oracle']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
