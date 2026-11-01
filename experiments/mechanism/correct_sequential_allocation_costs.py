"""Pre-score cost correction; no model calls or collection changes."""
import argparse
import copy
from pathlib import Path
from experiments.mechanism import run_sequential_allocation_screen as r
from experiments.mechanism.run_upgrade_pilot import freeze, utc

DESIGN=r.ROOT/'docs/research/sequential_allocation_v1/评分前成本更正.md'
COMMITMENT=r.OUT/'cost_correction_commitment.json'


def prepare():
    if COMMITMENT.exists():verify();return
    if (r.OUT/'result.json').exists():raise RuntimeError('cannot preregister after scores')
    freeze(COMMITMENT,{'utc':utc(),'scoring_file_present':False,
        'sealed_cases_at_commitment':len(list(r.OUT.glob('*/sealed_outputs.json'))),
        'sha256':{str(p.relative_to(r.ROOT)):r.p.base.h(p) for p in (Path(__file__),DESIGN)}})


def verify():
    for path,value in r.p.base.read(COMMITMENT)['sha256'].items():
        if r.p.base.h(r.ROOT/path)!=value:raise ValueError('pre-score correction changed:'+path)


def analyze():
    verify();r.verify();original=r.p.base.read(r.OUT/'result.json');result=copy.deepcopy(original)
    splits=r.p.base.read(r.OUT/'splits.json');details=result['per_case']
    for mode in r.MODES:
        for split,cell in result['cells'][mode]['splits'].items():
            ids=[c for c in details if split=='all' or splits[c]==split]
            policy=cell['simple_policies']['no_new_documents'];stops=policy['stages']
            policy['tokens']=sum(details[c][mode][s]['fixed_cost']['tokens'] for c,s in zip(ids,stops))
            policy['calls']=sum(details[c][mode][s]['fixed_cost']['calls'] for c,s in zip(ids,stops))
            policy['cost_scope']='query feedback only; no unused intermediate answers'
            budget=cell['fixed_stages'][2]
            feasible=[{'name':'fixed'+str(s),**v} for s,v in enumerate(cell['fixed_stages']) if v['tokens']<=budget['tokens'] and v['calls']<=budget['calls']]
            feasible+=[{'name':name,**v} for name,v in cell['simple_policies'].items() if v['tokens']<=budget['tokens'] and v['calls']<=budget['calls']]
            simple=max(feasible,key=lambda v:(v['em_correct'],-v['tokens'],-v['calls']))
            cell['best_budget_feasible_simple']={'name':simple['name'],'em_correct':simple['em_correct'],'tokens':simple['tokens'],'calls':simple['calls']}
            oracle=cell['oracle']['adaptive_cost']['best_with_fixed_s2_token_and_call_budget']
            cell['oracle_gap_to_best_feasible_simple']=None if oracle is None else oracle['correct']-simple['em_correct']
    result['cost_correction']={'commitment_sha256':r.p.base.h(COMMITMENT),'original_result_sha256':r.p.base.h(r.OUT/'result.json'),
        'only_no_new_documents_cost_changed':True,'not_a_new_experiment':True}
    result['gate']['confirmation_oracle_gap_to_best_feasible_simple']=result['cells']['enabled']['splits']['confirmation']['oracle_gap_to_best_feasible_simple']
    freeze(r.OUT/'corrected_analysis.json',result)
    print(__import__('json').dumps({'gate':result['gate'],'confirmation':{m:{'simple':result['cells'][m]['splits']['confirmation']['best_budget_feasible_simple'],
        'oracle_gap':result['cells'][m]['splits']['confirmation']['oracle_gap_to_best_feasible_simple']} for m in r.MODES}},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','analyze','verify'))
    {'prepare':prepare,'analyze':analyze,'verify':verify}[p.parse_args().stage]()
