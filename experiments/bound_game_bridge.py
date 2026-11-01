"""Predictions -> feasible allocation -> actual review -> sealed decision evaluation.

No oracle scores enter allocation. This is a controlled diagnostic, not a natural
benchmark or a claim of physically decentralized deployment.
"""
import argparse
import copy
import itertools
import json
import random
from pathlib import Path
from experiments import bound_opportunity_probe as bound
from experiments.consensus_mechanism import core as c
from experiments.consensus_mechanism.engine import estimate
from experiments.consensus_mechanism.allocation import SetAllocator
from experiments.consensus_mechanism.runner import locked
from experiments.consensus_mechanism.runtime import Runtime
from experiments.consensus_mechanism.data import read
from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha
from experiments.paired_deliberation.native_optimization_runtime import audit

MODES=('fixed','random','greedy','central','game')

def choose(task,state,reports,bundle,mode,calls,sources):
    actions=c.actions(task,state,allow_joint=True)
    if not actions: return [], {'assignment':[None]*3,'potential':0.,'reason':'no_active_action'}
    values=estimate(task,state,actions,reports,bundle)
    solver=SetAllocator(values,calls,sources,money=.02)
    if mode=='random':
        domain=[s for s in itertools.product([None]+list(range(len(values))),repeat=3)
                if solver.feasible(s)]
        nonempty=[s for s in domain if any(q is not None for q in s)]
        assignment=random.Random(int(fingerprint([task['id'],fingerprint(state),calls])[:16],16)).choice(nonempty or domain)
        allocation={'mode':'random','assignment':assignment,'potential':solver.potential(assignment)}
    else: allocation=solver.solve(mode)
    allocation.pop('solver_seconds', None)  # nondeterministic timing is not part of the frozen decision
    selected=[{**values[q],'reviewer':i,'marginal':solver.utility(allocation['assignment'],i)}
              for i,q in enumerate(allocation['assignment']) if q is not None]
    selected.sort(key=lambda row:(-row['marginal'],row['id'],row['reviewer']))
    return selected,allocation

def execute(row,api,bundle,mode,index,budget=2):
    task,state,reports=row['task'],copy.deepcopy(row['state']),row['reports']
    events=[];traces=[];left=budget;sources=budget
    while left>0 and sources>0:
        selected,allocation=choose(task,state,reports,bundle,mode,left,sources)
        traces.append({'state_hash':fingerprint(state),'remaining_calls':left,
                       'remaining_sources':sources,'allocation':allocation,'candidates_selected':selected})
        if not selected: break
        action=selected[0]
        before=fingerprint(state); c.expose(task,state,action['source_ids'])
        # All strategies use the same linked actor with the selected member's
        # report context; no policy-specific prompt or oracle-assisted action.
        reviewer=action['reviewer']
        req=bound.review_request(task,state,reports,action,reviewer)
        expected=fingerprint(state)
        event={'valid':False,'accepted':False,'action':action,'before_state_hash':before,
               'request_hash':fingerprint(req),'reviewer':reviewer,'before':copy.deepcopy(state['decision'])}
        try:
            record=api.call(f'{index}_{mode}_{budget-left}',req)
            event.update(report=record['report'],usage=record['usage'],returned_model=record['model'])
            if fingerprint(state)!=expected:raise ValueError('state changed during review')
            refs,basis=bound.validate_linked(task,state,record['report'],action)
            state,accepted=c.commit(task,state,record['report'],'review',100,refs)
            event.update(valid=True,accepted=accepted,report=record['report'],usage=record['usage'],
                         returned_model=record['model'],revision_basis=basis,citations=refs)
        except (ValueError,KeyError,TypeError,RuntimeError,OSError)as exc:event['error']=str(exc)
        event['after']=copy.deepcopy(state['decision']);events.append(event)
        left-=1;sources-=len(action['source_ids'])
    return {'task_id':task['id'],'initial_decision':row['state']['decision'],
            'decision':state['decision'],'events':events,'allocation_trace':traces,
            'calls':len(events),'source_reads':budget-sources}

def evaluate(rows,cases):
    summary={'tasks':len(rows),'methods':{},'scope':'Controlled fixed-state bridge; pretrained contribution estimator; no heldout natural-domain or decentralized deployment claim.'}
    for mode in MODES:
        before=[c.quality(row['state']['decision'],row['evaluation'])for row in rows]
        after=[c.quality(case['methods'][mode]['decision'],row['evaluation'])for row,case in zip(rows,cases)]
        events=[e for case in cases for e in case['methods'][mode]['events']]
        summary['methods'][mode]={'quality_before':sum(before)/len(before),'quality_after':sum(after)/len(after),
          'improved':sum(b>a for a,b in zip(before,after)),'harmed':sum(b<a for a,b in zip(before,after)),
          'logical_calls':len(events),'valid_reviews':sum(e['valid']for e in events),
          'accepted':sum(e['accepted']for e in events),
          'selected_source_sets':[[e['action']['source_ids']for e in case['methods'][mode]['events']]for case in cases]}
    summary['game_vs_central_different_choices']=sum(
      summary['methods']['game']['selected_source_sets'][i]!=summary['methods']['central']['selected_source_sets'][i]
      for i in range(len(rows)))
    summary['diagnosis']={'selection_can_differ':len({fingerprint(v['selected_source_sets'])for v in summary['methods'].values()})>1,
                         'any_actual_improvement':any(v['improved']for v in summary['methods'].values()),
                         'game_gain_over_greedy':summary['methods']['game']['quality_after']-summary['methods']['greedy']['quality_after']}
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','report'])
    p.add_argument('--root',type=Path,default=ROOT/'experiments/results/bound_game_bridge_v3')
    p.add_argument('--manifest',type=Path,default=ROOT/'experiments/results/bound_opportunity_v3/manifest.private.json')
    p.add_argument('--model',type=Path,default=ROOT/'experiments/results/consensus_mechanism_paper_v2/model/model.joblib')
    p.add_argument('--run',action='store_true');p.add_argument('--max-cny',type=float,default=.3)
    a=p.parse_args()
    if not 0<a.max_cny<=.5:p.error('hard cap 0..0.5 CNY')
    rows=read(a.manifest)
    for row in rows:bound.validate_binding(row)
    # The value model is frozen training-side, never fitted from these labels.
    import joblib
    bundle=joblib.load(a.model)
    source_files=[Path(__file__),Path(bound.__file__)]+list((ROOT/'experiments/consensus_mechanism').glob('*.py'))+list((ROOT/'experiments/paired_deliberation').glob('*.py'))+[ROOT/'experiments/consensus_recommendation/common.py',ROOT/'experiments/evidence_consensus/schema.py']
    plan={'version':'bound-game-bridge-2-typed-verdict','manifest_hash':fingerprint(rows),'model_hash':sha(a.model),
          'physical_cap':len(rows)*len(MODES)*2,'max_cny':a.max_cny,'reuse_root':None,'methods':list(MODES),
          'source_sha256':{str(f.relative_to(ROOT)):sha(f)for f in sorted(set(source_files))},
          'selection_labels_available':False,'construction_labels_known':True}
    # Precompute first allocations BEFORE generation or evaluation. Explicitly
    # persist every estimator input, prediction, feasible assignment and trace.
    forecasts={row['task']['id']:{mode:choose(row['task'],row['state'],row['reports'],bundle,mode,2,2)
                                 for mode in MODES}for row in rows}
    with locked(a.root):
        create(a.root/'plan.json',plan);create(a.root/'manifest.private.json',rows)
        create(a.root/'selection.before_generation.json',forecasts)
        dest=a.root/'estimator.joblib'
        if dest.exists()and sha(dest)!=sha(a.model):raise ValueError('estimator drift')
        dest.write_bytes(a.model.read_bytes())
        for name in plan['source_sha256']:
            dst=a.root/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists()and sha(dst)!=plan['source_sha256'][name]:raise ValueError('source drift')
            dst.write_bytes((ROOT/name).read_bytes())
        if a.stage=='prepare'or(a.stage=='run'and not a.run):
            print(json.dumps({'tasks':len(rows),'methods':MODES,'max_calls':plan['physical_cap'],'network':False}));return
        if a.stage=='run':
            from dotenv import load_dotenv
            load_dotenv(ROOT/'.env.local',override=False);api=Runtime(a.root,plan,'qwen_flash')
            for i,row in enumerate(rows):
                arms={}
                for mode in MODES:
                    path=a.root/'arms'/f'{i}_{mode}.json'
                    if not path.exists():create(path,execute(row,api,bundle,mode,i))
                    arms[mode]=read(path)
                create(a.root/'cases'/f'case_{i}.json',{'task_id':row['task']['id'],'methods':arms})
            cases=[read(a.root/'cases'/f'case_{i}.json')for i in range(len(rows))]
            create(a.root/'generation_complete.json',{'manifest_hash':fingerprint(rows),'hashes':[fingerprint(x)for x in cases]})
        cases=[read(a.root/'cases'/f'case_{i}.json')for i in range(len(rows))]
        if read(a.root/'generation_complete.json')!={'manifest_hash':fingerprint(rows),'hashes':[fingerprint(x)for x in cases]}:raise ValueError('seal mismatch')
        result=evaluate(rows,cases);result['physical']=audit(a.root,'qwen_flash');create(a.root/'summary.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
