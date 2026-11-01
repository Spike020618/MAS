"""Post-seal evidence-use and reversal diagnostics, without reading labels.

A repeated decision or old source is a diagnostic signal, not proof that an
update is wrong: new inference from the same source can be useful. No APIs.
"""
import argparse,json,collections
from pathlib import Path
from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha


def audit(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text());seal=json.loads((root/'generation_complete.json').read_text());cases=[json.loads((root/'cases'/('case_'+str(i)+'.json')).read_text())for i in range(plan['count'])]
    assert seal['hashes']==[fingerprint(c)for c in cases]
    groups={};records=[]
    for case in cases:
        for method,arm in case['methods'].items():
            counter=groups.setdefault(case['kind']+':'+method,collections.Counter())
            reviews=[e for e in arm['events']if 'source_action'in e]
            first=next((e['before_decision']for e in reviews if 'before_decision'in e),arm['decision'])
            seen_decisions={fingerprint(first)};cited=set()
            for e in reviews:
                counter['reviews']+=1
                if not e.get('patch_accepted'):continue
                counter['accepted']+=1;p=e['report']['C']['proposed_patch'];evidence=set(p['evidence_ids']);after=fingerprint(e['after_decision'])
                uses_current=e['source_action']['source_id']in evidence
                repeated=bool(cited)and evidence<=cited
                reversal=after in seen_decisions
                counter['accepted_citing_current_fetched_source']+=uses_current
                counter['accepted_without_new_cited_source_since_previous_acceptance']+=repeated
                counter['accepted_returning_to_previous_decision']+=reversal
                records.append({'task_id':case['task_id'],'kind':case['kind'],'method':method,'slot':e['slot'],'source_fetched':e['source_action']['source_id'],'sources_cited':sorted(evidence),'uses_current_fetched_source':uses_current,'no_new_cited_source_since_previous_acceptance':repeated,'returns_to_previously_seen_decision':reversal,'before_decision_hash':fingerprint(e['before_decision']),'after_decision_hash':after})
                cited.update(evidence);seen_decisions.add(after)
    result={'scope':'label-free executed-trace diagnosis; no claim repeated source/reversal is necessarily wrong; logical-arm counts may share exact physical response','plan_sha256':sha(root/'plan.json'),'generation_seal_sha256':sha(root/'generation_complete.json'),'groups':{k:dict(v)for k,v in groups.items()},'accepted_events':records,'labels_read':False,'model_api_calls':0}
    create(root/'independent_contribution_trace_audit.json',result);print(json.dumps({'groups':result['groups'],'accepted_events':len(records)},ensure_ascii=False,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'experiments/results/unified_e1_v4');a=p.parse_args();audit(a.output)
