"""No new API calls or lambda search: exact-response obstruction + quarantine."""
from hashlib import sha256
import json
from pathlib import Path

from experiments.core.mechanism.signaling_safety_audit import rank_lock,quarantine
from experiments.mechanism.run_upgrade_pilot import freeze

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'experiments/results/resource_allocation/signaling_pilot_v1'
OUT=ROOT/'experiments/results/audit/signaling_failure_repair_v2'


def main():
    sealed=json.loads((SOURCE/'sealed_readouts.json').read_text())
    choices={c['id']:c['choices'] for c in json.loads((SOURCE/'public_cases.json').read_text())}
    # First audit only inference objects. Labels are attached later for diagnosis.
    audits={}
    for qid,row in sealed.items():
        complete=bool(row['details']);d=row['details'];answers=row['answers']
        audits[qid]={'quarantine':quarantine(answers['game'],answers['verification'],inputs_complete=complete,
                    converged=d['game_result']['converged_to_tolerance'] if complete else False),
                    'lock':rank_lock(choices[qid],d['game_reference']['generator'][0],.5) if complete else None,
                    'game_answers':answers,
                    'endogenous_checker_answer':d['game_result']['readout']['discriminator']['answer'] if complete else None}
    OUT.mkdir(parents=True,exist_ok=True)
    freeze(OUT/'label_blind_audit.json',audits)
    labels={r['id']:r['gold'] for r in json.loads((SOURCE/'result.json').read_text())['records']}
    lock_cases=[qid for qid,r in audits.items() if r['lock'] and r['lock']['exact_response_rank_locked']]
    checker_corruption=[qid for qid,r in audits.items() if r['game_answers']['verification']==labels[qid]
                        and r['endogenous_checker_answer'] is not None and r['endogenous_checker_answer']!=labels[qid]]
    summary={'status':'structural_failure_localized_not_effective_game_repaired','new_api_calls':0,
        'n':len(audits),'exact_response_rank_locked_cases':lock_cases,
        'locked_anchor_wrong_cases':[qid for qid in lock_cases if audits[qid]['lock']['anchor_top']!=labels[qid]],
        'initial_checker_correct_but_game_checker_wrong':checker_corruption,
        'quarantined_game_outputs':sum(not r['quarantine']['game_admissible_as_unchanged_checker_readout'] for r in audits.values()),
        'fallback_identically_direct_verification':all(r['quarantine']['operational_fallback_answer']==r['game_answers']['verification'] for r in audits.values()),
        'warning':'fallback copies baseline, not a game-caused accuracy gain; no parameter search or new efficacy test',
        'supersedes':'signaling_failure_repair_v1: diagnostic used sorted JSON score keys instead of registered candidate vector order; original pilot unchanged',
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in
                         [SOURCE/'public_cases.json',SOURCE/'sealed_readouts.json',SOURCE/'result.json']}}
    freeze(OUT/'result.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
