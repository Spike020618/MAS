"""Bind source scopes and execute finite-game-selected review paths on saved initials."""
import argparse
import copy
import json
from pathlib import Path

from experiments.consensus_recommendation.common import ROOT, OUTPUT, create, fingerprint, read_rows, sha
from experiments.paired_deliberation.rapid_negotiation import API, aggregate, local_update, materials, scores, select_target, physical_audit
from experiments.paired_deliberation.scoped_evidence import cards, request, validate
from experiments.paired_deliberation.review_allocation_game import ReviewGame
from experiments.paired_deliberation.evaluate_training_signals import metrics


def run(root, source, count):
    root, source = Path(root), Path(source)
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env.local', override=False)
    previous = json.loads((source/'plan.json').read_text())
    selected = previous['cases'][:count]
    items = {r['id']: r for r in json.loads((source/'inputs.json').read_text())}
    plan = {'version': 'scoped-review-allocation-game-v1', 'cases': selected,
        'parent_root': str(source.resolve()), 'parent_plan_sha256': sha(source/'plan.json'),
        'reuse_root': None, 'players': 3, 'max_candidates': 4, 'per_player_review_allowance': 1,
        'max_new_calls_per_case': 3, 'cost_rule': '.1 * (UTF8 input upper / 64000 + max_output / 4096)',
        'value_rule': 'public boundary priority normalized by maximum priority in this case; NOT accuracy value',
        'game': 'marginal coverage minus committed own effort cost; asynchronous strict best responses',
        'semantic_authority': 'runtime source identity and immutable evidence channels, never model-generated current facts',
        'phase': 'scope safety and live game integration, not fair effect comparison',
        'generation_labels_available': False, 'source_sha256': {n: sha(ROOT/'experiments/paired_deliberation'/n)
            for n in ('run_scoped_game.py','review_allocation_game.py','scoped_evidence.py','rapid_negotiation.py')}}
    create(root/'plan.json', plan); api = API(root, plan); results = []
    for cid in selected:
        item = items[cid]; prefix = cid.replace(':','_')
        parent = json.loads((source/'cases'/(prefix+'.json')).read_text())
        pool = [m['movie_id'] for m in item['evidence']['candidates_in_shared_retrieval_order']]
        vectors = [scores(r['ranking'], pool) for r in parent['initial']]
        base_rank = aggregate(vectors,pool)[0][:5]
        entry = {'id': cid, 'parent_initial_hash': fingerprint(parent['initial']), 'baseline': base_rank,
                 'events': [], 'parent_initial_valid': all(r['valid'] for r in parent['initial'])}
        if not entry['parent_initial_valid']:
            entry.update(ranking=base_rank, status='parent_initial_failed_no_dispatch')
            results.append(entry); create(root/'cases'/(prefix+'.json'),entry); continue
        source_map,_ = materials(item); bound_cards = cards(cid,source_map)
        processed = set(); targets = []
        while len(targets)<4:
            t = select_target(vectors,pool,processed)
            if t is None: break
            processed.add(frozenset((t['a'],t['b']))); targets.append(t)
        peak = max((t['priority'] for t in targets),default=1)
        targets = [{**t,'value':t['priority']/peak} for t in targets]
        reqs = [request(cid,bound_cards,t) for t in targets]
        costs = [[.1*(sum(len(m['content'].encode('utf8'))+512 for m in req['messages'])/64000 + req['max_tokens']/4096)
                  for req in reqs] for _ in range(3)]
        game = ReviewGame(targets,costs); outcome = game.solve()
        commitment = {'targets':targets,'costs':costs,'strategy_sets':game.actions,'initial_state_hash':fingerprint(vectors)}
        entry.update(game=outcome,commitment=commitment,commitment_hash=fingerprint(commitment))
        # The complete payoff table and strategy outcome exist BEFORE any corresponding API execution.
        create(root/'commitments'/(prefix+'.json'),{'game':outcome,**commitment,'commitment_hash':entry['commitment_hash']})
        for reviewer,q in enumerate(outcome['actions']):
            if q is None: continue
            t = targets[q]; req = reqs[q]; before = fingerprint(vectors)
            slot = prefix+'_player'+str(reviewer)+'_query'+str(q)
            event = {'reviewer':reviewer,'query':q,'target':t,'state_hash_before':before}
            try:
                obj,attempts = api.complete('qwen_flash',slot,req,item['evidence'])
                bound = validate(obj,req)
                vectors = local_update(vectors,t,bound['preferred'],before)
                event.update(status='committed',response=obj,bound_selection=bound,attempts=attempts,
                             state_hash_after=fingerprint(vectors),unrelated_scores_preserved=True)
            except (ValueError,KeyError,TypeError,RuntimeError) as exc:
                event.update(status='rejected',error=str(exc))
            entry['events'].append(event)
        entry.update(ranking=aggregate(vectors,pool)[0][:5],status='completed',
            final_changed=aggregate(vectors,pool)[0][:5]!=base_rank,
            current_user_fact_promotions=0)
        create(root/'cases'/(prefix+'.json'),entry);results.append(entry)
        print(cid+' game_actions='+str(outcome['actions'])+' regret='+str(outcome['max_unilateral_gain'])+
              ' committed='+str(sum(e['status']=='committed' for e in entry['events'])),flush=True)
    create(root/'generation_complete.json',{'case_hashes':{r['id']:fingerprint(r) for r in results},'generation_labels_available':False})
    labels = {r['case_id']:r['relevant_movie_ids'] for r in read_rows(OUTPUT/'data/dev.labels.jsonl')}
    scored = [(metrics(r['ranking'],labels[r['id']]),metrics(r['baseline'],labels[r['id']])) for r in results]
    scored = [p for p in scored if p[0] is not None]
    games = [r['game'] for r in results if 'game' in r];events=[e for r in results for e in r['events']]
    summary = {'cases':len(results),'annotated':len(scored),'game_selected_hits':sum(p[0]['Hit@5'] for p in scored),
        'independent_hits':sum(p[1]['Hit@5'] for p in scored),
        'game_selected_NDCG@5':sum(p[0]['NDCG@5'] for p in scored)/len(scored) if scored else None,
        'independent_NDCG@5':sum(p[1]['NDCG@5'] for p in scored)/len(scored) if scored else None,
        'max_proxy_game_regret':max((g['max_unilateral_gain'] for g in games),default=0),
        'games_with_nontrivial_strategy_changes':sum(bool(g['trace']) for g in games),
        'max_central_proxy_gap':max((g['central_potential_gap'] for g in games),default=0),
        'selected_actions_executed':len(events),'source_scoped_commits':sum(e['status']=='committed' for e in events),
        'rejected_selections':sum(e['status']=='rejected' for e in events),
        'final_rankings_changed':sum(r.get('final_changed',False) for r in results),
        'current_user_fact_promotions':0,'physical':physical_audit(root)[0],
        'scope':'live integration of a REAL software-controlled finite proxy game; no semantic correctness or recommendation superiority proved',
        'limits':['standard coverage potential game, not a novel general game theorem','proxy equilibrium is not true-quality equilibrium',
                  'up to three review calls unlike parent two-call methods; no causal quality comparison',
                  'source channel validation does not establish all natural language entailments',
                  'small already-used development sample; no heldout files read']}
    create(root/'summary.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--source',required=True);p.add_argument('--count',type=int,default=4)
    a=p.parse_args()
    if not 1<=a.count<=8: raise ValueError('integration screen only')
    print(json.dumps(run(a.output,a.source,a.count),ensure_ascii=False,indent=2))
