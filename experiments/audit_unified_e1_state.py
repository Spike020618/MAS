"""Independent sealed-state replay; no gold-label reads or network calls."""
import argparse, copy, json
from pathlib import Path
from experiments.consensus_recommendation.common import ROOT, create, fingerprint, sha
from experiments.evidence_consensus.schema import validate_decision, candidate_ids
from experiments.evidence_consensus.state import initialize_decision, register_report, apply_patch
from experiments.evidence_consensus.actions import fetch_source, verify_source
from experiments.evidence_consensus.runner import common_state, aggregate, linked_patch


def audit(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text());seal=json.loads((root/'generation_complete.json').read_text())
    checks=[]
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest or sha(root/'source_snapshot'/name)!=digest:
            raise ValueError('executed/live source changed: '+name)
    rows=json.loads((root/'inputs.json').read_text())
    if len(rows)!=plan['count'] or fingerprint(rows)!=plan['bounded_tasks_hash']:raise ValueError('frozen input mismatch')
    cases=[json.loads((root/'cases'/('case_'+str(i)+'.json')).read_text())for i in range(len(rows))]
    if seal['hashes']!=[fingerprint(c)for c in cases]or seal['task_ids']!=[t['id']for t in rows]:raise ValueError('generation seal mismatch')
    totals={'tasks':len(rows),'method_states':0,'review_calls':0,'accepted_patches':0,'rejected_patches':0,'accepted_state_replay_failures':0,'unknown_budget_receipts':0,'unknown_zero_cny_bound':0,'swap_direction_violations':0,'claim_only_operations':0}
    for task,case in zip(rows,cases):
        if case['task_id']!=task['id']:raise ValueError('case/task identity mismatch')
        for method,arm in case['methods'].items():
            events=arm['events'];initial=[e for e in events if 'source_action'not in e]
            reports=[e['report']for e in initial if e['valid']]
            state=common_state(task)
            if method=='single':
                for e in initial:
                    if e['valid']:
                        report=e['report'];state['decision']=validate_decision(task,report['decision'])
                        if task['kind']=='rec':state['decision'] += [m for m in candidate_ids(task)if m not in state['decision']]
                        try:state=register_report(state,report)
                        except ValueError:pass
                state=initialize_decision(state,copy.deepcopy(state['decision']))
            else:
                state=initialize_decision(state,aggregate(task,reports))
                for report in reports:
                    try:state=register_report(state,report)
                    except ValueError:pass
            for e in events:
                if 'source_action'not in e:continue
                totals['review_calls']+=1
                if e['expected_state_version']!=state['version']:raise ValueError('request state version mismatch')
                sid=e['source_action']['source_id'];fetch=fetch_source(state,sid);verify_source(state,sid)
                if fetch!=e['source_action']:raise ValueError('source expansion replay mismatch')
                if not e['valid']:continue
                report=e['report'];before=copy.deepcopy(state)
                try:
                    next_state=register_report(state,report);patch=report['C']['proposed_patch']
                    allowed={'abstain','set_answer'}if task['kind']=='qa'else {'abstain','swap_candidate'}
                    if patch.get('operation')not in allowed:raise ValueError('review must use scoped task patch')
                    if not linked_patch(report):raise ValueError('unlinked_patch_evidence')
                    result=apply_patch(next_state,patch)
                    if result['accepted']!=e.get('patch_accepted'):raise ValueError('recorded acceptance mismatch')
                    if e.get('before_decision')!=before['decision']or e.get('after_decision')!=result['state']['decision']:
                        raise ValueError('recorded decision transition mismatch')
                    if result['accepted']:
                        totals['accepted_patches']+=1
                        if patch['operation']=='swap_candidate':
                            rank=before['decision'];a,b=patch.get('a'),patch.get('b')
                            if not(rank.index(a)<10 and rank.index(b)>=10):totals['swap_direction_violations']+=1
                            positions=[j for j,(x,y)in enumerate(zip(rank,result['state']['decision']))if x!=y]
                            if sorted(positions)!=sorted([rank.index(a),rank.index(b)]):raise ValueError('non-target rank positions changed')
                        elif patch['operation']=='update_claim':totals['claim_only_operations']+=1
                    else:totals['rejected_patches']+=1
                    state=result['state']
                except (ValueError,KeyError,TypeError) as exc:
                    if e.get('patch_accepted'):raise ValueError('accepted patch failed independent replay')from exc
                    if e.get('patch_reason')!=str(exc):raise ValueError('rejection reason changed')from exc
                    totals['rejected_patches']+=1
            if state!=arm['state']or state['decision']!=arm['decision']:raise ValueError('final readout differs from replayed state')
            budget=arm['budget']
            for key in budget['unknown_requests']:
                totals['unknown_budget_receipts']+=1
                if budget['receipts'][key]['charged']['cny']<=0:totals['unknown_zero_cny_bound']+=1
            for field,limit in budget['limits'].items():
                if budget['used_or_reserved'][field]>limit+1e-10:raise ValueError('task budget exceeded')
            totals['method_states']+=1
    out={'scope':'independent executed-source state replay, no label reads or API calls','source_audit_sha256':sha(Path(__file__)),'plan_sha256':sha(root/'plan.json'),'generation_seal_sha256':sha(root/'generation_complete.json'),'totals':totals,'all_state_readouts_match':True,'semantic_truth_certified':False,
         'structural_limits':['update_claim is not admitted by E1 runner and has no separate final readout mechanism','swap Top10-to-outside direction instructed but not hard-enforced by core','proposal scores use untrained metadata relevance; not semantic information value','unknown response upper-bound reservation is accounting conservatism, not platform bill']}
    create(root/'independent_state_replay_audit.json',out);print(json.dumps(out,ensure_ascii=False,indent=2));return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'experiments/results/unified_e1_v3');a=p.parse_args();audit(a.output)
