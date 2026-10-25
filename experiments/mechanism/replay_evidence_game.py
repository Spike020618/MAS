"""Replay fixed source reports into a coupled evidence-payoff game.

Exploratory fixed-report payoff tables, not a live autonomous equilibrium test.
Gold labels score leader outcome only; settlement depends on trace/support logs.
"""
import argparse
import hashlib
import json
import sys
from itertools import product
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.core.mechanism.evidence_settlement import settle_evidence
from experiments.mechanism.solve_multifollower import solve_multifollower
from experiments.mechanism.pilot_external_evidence import quote_provenance


def main():
    p=argparse.ArgumentParser()
    p.add_argument('directory',type=Path)
    args=p.parse_args()
    directory=args.directory
    rows=json.loads((directory/'outcomes.json').read_text())
    retrieval={r['key']:r for r in map(json.loads,(directory/'retrieval.jsonl').read_text().splitlines())}
    question_keys=sorted({(r['dataset'],r['question_id']) for r in rows})
    results=[]
    for dataset,qid in question_keys:
        observations={(r['repeat'],r['condition']):r for r in rows if r['dataset']==dataset and r['question_id']==qid}
        sources=retrieval[f'{dataset}/{qid}']['sources']
        source_by_id={s['id']:s for s in sources}
        source_hashes={s['id']:hashlib.sha256((s['url']+'\n'+s['excerpt_sha256']).encode()).hexdigest() for s in sources}
        contracts=[]
        for budget in [0.0,0.5,1.0,2.0,4.0]:
            table=[]
            for profile in product(['direct','retrieve'],repeat=2):
                reports=[];selected=[]
                for agent,action in enumerate(profile):
                    r=observations[agent,'direct' if action=='direct' else 'retrieval_check']
                    selected.append(r)
                    citations=[]
                    if action=='retrieve':
                        status=(r.get('support_judgment') or {}).get('support','insufficient')
                        for citation in r['citations']:
                            if citation.get('source_id') not in source_by_id:continue
                            citations.append({'source_hash':source_hashes[citation['source_id']],
                                'exact_quote_valid':quote_provenance([citation],sources)['all_valid'],
                                'support_status':status})
                    reports.append({'agent_id':str(agent),'cost':r['decision_tokens']/1000,
                        'authorized_source_hashes':list(source_hashes.values()) if action=='retrieve' else [],
                        'citations':citations})
                settlement=settle_evidence(reports,budget)
                # With two valid binary reports, fair random tie-breaking yields
                # mean individual correctness; this is an expected aggregation score.
                quality=sum(r['correct'] for r in selected)/2
                token_cost=sum(r['decision_tokens'] for r in selected)/1000
                leader=quality-0.05*settlement['paid']-0.01*token_cost
                table.append({'profile':list(profile),'followers':[settlement['agents'][str(i)]['utility'] for i in [0,1]],
                              'leader':leader,'expected_team_quality':quality,'settlement':settlement})
            contracts.append({'id':f'budget{budget:g}','payoffs':table})
        game={'agents':['0','1'],'actions':{'0':['direct','retrieve'],'1':['direct','retrieve']},'contracts':contracts}
        results.append({'dataset':dataset,'question_id':qid,'game':game,'solution':solve_multifollower(game)})
    output={'scope':'fixed-report exploratory replay, not live follower behavior or out-of-sample contract performance',
            'credit_unit':'1000 generation-path tokens; virtual cost proxy, not money',
            'cost_exclusions':'query generation, HTTP and external support assessment excluded from payoff proxy; report these separately',
            'support_limit':'one report-level LLM support judgment applied to its quotes, not independently fact-checked source-level entailment',
            'leader_quality_uses_gold_offline_only':True,
            'contract_selection_is_question_specific_oracle_replay':True,
            'deployment_warning':'cannot deploy per-question contracts chosen with true outcomes; must calibrate across separate development tasks',
            'results':results}
    (directory/'coupled_game_replay.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'questions':len(results),'pure_nash_solved':True,
        'scope':output['scope'], 'source_based_settlement_only':True},indent=2))


if __name__=='__main__':main()
