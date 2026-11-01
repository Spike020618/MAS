"""Label-free deterministic replay of allocations, source bindings and readouts."""
import copy,json
from pathlib import Path
from experiments.consensus_recommendation.common import create,fingerprint,sha
from experiments.paired_deliberation.native_consensus_experiment import RESULT,review_request,validate_review,readout
from experiments.paired_deliberation.rapid_negotiation import scores,local_update
from experiments.paired_deliberation.review_allocation_game import ReviewGame
from experiments.paired_deliberation.transport_d2 import parse_d2_raw


def verify_case(root,index,row,result):
    assert result['id']==row['id'] and result['generation_labels_available'] is False
    if len(result['initial'])!=3:return {'case':index,'prerequisites_complete':False,'valid_review_events_replayed':0}
    pool=[m['movie_id'] for m in row['candidates']];initial=[c['ranking'] for c in result['initial']]
    vectors=[scores(r,pool) for r in initial];ind=result['methods']['independent3']
    assert ind['valid'] and readout(vectors,pool)==ind['ranking']
    six=result['methods']['independent6']
    if six['valid']:assert readout([scores(c['ranking'],pool) for c in six['calls']],pool)==six['ranking']
    single=result['methods']['single6']
    if single['valid']:
        final=single['calls'][-1]['ranking'];assert single['ranking']==list(map(int,final+[m for m in pool if m not in final]))
    replayed=0;aliases=0
    for method in ('fixed_review','game_review'):
        arm=result['methods'][method];commit=json.loads((root/'commitments'/('c'+str(index)+'_'+method+'.json')).read_text())
        assert fingerprint(commit)==arm['commitment_hash'] and fingerprint(vectors)==commit['initial_hash']
        reqs=[[review_request(row,initial,t,i) for t in commit['targets']] for i in range(3)]
        assert fingerprint(reqs)==commit['requests_hash']
        if commit['game'] is not None:
            solved=ReviewGame(commit['targets'],commit['costs']).solve()
            assert fingerprint(solved)==fingerprint(commit['game'])
            assert commit['allocation']==[[i,q] for i,q in enumerate(solved['actions']) if q is not None]
        used=[]
        for t in commit['targets']:used.extend([t['a'],t['b']])
        assert len(used)==len(set(used))
        state=copy.deepcopy(vectors)
        assert len(arm['events'])==len(commit['allocation'])
        for event,(player,q) in zip(arm['events'],commit['allocation']):
            target=commit['targets'][q];assert player!=target['owner'] and event['target']==target
            raw_path=root/'slots'/(event['slot']+'.raw.json')
            if not event['valid']:continue
            raw=json.loads(raw_path.read_text());assert fingerprint(raw['raw'])==raw['integrity']
            assert raw['request']==reqs[player][q]
            record=parse_d2_raw(raw['raw'],raw['request']);bound=validate_review(record['report'],raw['request'],row)
            assert fingerprint(bound)==fingerprint(event['bound_selection'])
            assert not bound['entailment_verified'] and not bound['current_preference_certified']
            if not bound['abstained']:state=local_update(state,target,bound['preferred'],fingerprint(state))
            replayed+=1;aliases+=len(bound.get('representation_operations',[]))
        assert readout(state,pool)==arm['diagnostic_preserved_state_ranking']
        if arm['valid']:assert arm['ranking']==readout(state,pool)
        else:assert arm['ranking']==[]
        for before,after in zip(vectors,state):assert sorted(before.values())==sorted(after.values())
    return {'case':index,'prerequisites_complete':True,'valid_review_events_replayed':replayed,'lossless_aliases_replayed':aliases}


def run(root=RESULT,partial=False):
    inputs=json.loads((root/'inputs.json').read_text());results=[]
    for i,row in enumerate(inputs):
        path=root/'cases'/('case_'+str(i)+'.json')
        if not path.exists():
            if partial:continue
            raise ValueError('incomplete generations')
        results.append(verify_case(root,i,row,json.loads(path.read_text())))
    audit={'source_sha256':sha(Path(__file__)),'no_labels_loaded':True,'verified_cases':len(results),
           'valid_review_events_replayed':sum(r['valid_review_events_replayed'] for r in results),
           'lossless_aliases_replayed':sum(r.get('lossless_aliases_replayed',0) for r in results),'cases':results}
    if not partial:create(root/'label_free_replay_audit.json',audit)
    print(json.dumps(audit,indent=2));return audit
if __name__=='__main__':run()
