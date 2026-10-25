"""Check whether the observed correctness payoffs actually couple actions."""
import json
from experiments.mechanism import run_learned_production as run
from experiments.core.mechanism import learned_production_game as g
from experiments.mechanism.run_upgrade_pilot import freeze


def main():
    result={'scope':'post-hoc realized correctness-payoff interaction audit, not a general impossibility theorem',
            'new_api_calls':0,'splits':{}}
    for split in ('train','test'):
        outputs=json.loads((run.OUT/(split+'_sealed_outputs.json')).read_text())
        labels=json.loads((run.OUT/(split+'_labels.json')).read_text());rows=[]
        for cid,slots in outputs.items():
            correctness={''.join(map(str,p)):int(g.outcome(slots,p)==labels[cid]) for p in g.PROFILES}
            interaction=correctness['11']-correctness['10']-correctness['01']+correctness['00']
            rows.append({'id':cid,'correctness':correctness,'joint_interaction':interaction,
                         'any_correctness_change':len(set(correctness.values()))>1})
        result['splits'][split]={'n':len(rows),'nonzero_joint_interactions':sum(r['joint_interaction']!=0 for r in rows),
            'any_correctness_change_cases':sum(r['any_correctness_change'] for r in rows),'rows':rows}
    freeze(run.OUT/'strategic_interaction.json',result)
    print(json.dumps({k:{x:y for x,y in v.items() if x!='rows'} for k,v in result['splits'].items()},indent=2))


if __name__=='__main__':main()
