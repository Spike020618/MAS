"""Post-hoc capability controls: disclose ALL five single-report paths."""
import json
from experiments.mechanism import run_learned_production as run
from experiments.mechanism.run_upgrade_pilot import freeze


def main():
    result={'scope':'post-hoc capability audit; all roles disclosed, not test-selected deployable controller',
            'new_api_calls':0,'splits':{}}
    for split in ('train','test'):
        outputs=json.loads((run.OUT/(split+'_sealed_outputs.json')).read_text())
        labels=json.loads((run.OUT/(split+'_labels.json')).read_text())
        result['splits'][split]={slot:{'n':len(outputs),
            'correct':sum(s[slot]['answer']==labels[cid] for cid,s in outputs.items()),
            'tokens':sum(s[slot]['tokens'] for s in outputs.values()),'calls':len(outputs),
            'sum_http_seconds':sum(s[slot]['wall_seconds'] for s in outputs.values())}
            for slot in ('fast0','fast1','fast2','think1','think2')}
    freeze(run.OUT/'single_report_controls.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
