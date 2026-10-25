"""Post-unsealing offline diagnosis; never modifies prompts/parameters or calls API."""
import json
from experiments.mechanism import run_signaling_pilot as run
from experiments.mechanism.run_upgrade_pilot import freeze


def main():
    cases=json.loads((run.OUT/'public_cases.json').read_text())
    sealed=json.loads((run.OUT/'sealed_outputs.json').read_text())
    readouts=json.loads((run.OUT/'sealed_readouts.json').read_text())
    result=json.loads((run.OUT/'result.json').read_text())
    labels={r['id']:r['gold'] for r in result['records']}
    records=[]
    for c in cases:
        r=readouts[c['id']];s=sealed[c['id']];answers=r['answers'];game=r['details'].get('game_result')
        bad=[key for key,v in s.items() if not v['usable']]
        pos,neg=s['g_positive'],s['g_negative']
        same=(max(pos['probabilities'],key=pos['probabilities'].get)==
              max(neg['probabilities'],key=neg['probabilities'].get)) if pos['usable'] and neg['usable'] else None
        records.append({'id':c['id'],'answers':answers,'gold':labels[c['id']],
            'fully_usable_game':bool(game),'unusable_slots':bad,'positive_negative_same_top_code':same,
            'game_residual':game['residual']['max_gap'] if game else None,
            'game_sweeps':game['sweeps'] if game else None,
            'game_vs_verification':('harm_unavailable_game_input' if not game else 'harm_with_usable_input')
                if answers['verification']==labels[c['id']] and answers['game']!=labels[c['id']] else 'other'})
    summary={'n':12,'new_api_calls':0,'scope':'posthoc diagnostic, not an independent replication or tuning set',
        'game_same_answer_as_generation':sum(r['answers']['game']==r['answers']['generation'] for r in records),
        'game_same_answer_as_contrastive_product':sum(r['answers']['game']==r['answers']['contrastive_product'] for r in records),
        'game_verification_harms_by_cause':{reason:sum(r['game_vs_verification']==reason for r in records) for reason in
            ('harm_unavailable_game_input','harm_with_usable_input')},
        'fully_usable_game_residual_max':max(r['game_residual'] for r in records if r['game_residual'] is not None),
        'positive_negative_valid_pairs':sum(r['positive_negative_same_top_code'] is not None for r in records),
        'positive_negative_same_top':sum(r['positive_negative_same_top_code'] is True for r in records),
        'reversed_code_valid_pairs':{arm:sum(arm in r['reversed_code_diagnostics'] for r in readouts.values()) for arm in ('positive','negative')},
        'records':records}
    freeze(run.OUT/'diagnosis.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='records'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
