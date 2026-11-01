"""Independent online-update behavior diagnosis; no API and no label access."""
import argparse,collections,json
from pathlib import Path
from experiments.consensus_recommendation.common import create,fingerprint,sha


def audit(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text());seal=json.loads((root/'generation_complete.json').read_text())
    rows=[json.loads((root/'cases'/('case_'+str(i)+'.json')).read_text())for i in range(plan['count'])]
    if seal['hashes']!=[fingerprint(r)for r in rows]:raise ValueError('case generation identity mismatch')
    counts=collections.defaultdict(collections.Counter);details=[]
    for row in rows:
        for method,arm in row['methods'].items():
            key=row['kind']+':'+method;out=counts[key];seen=set();cycle=0
            for event in arm['events']:
                if 'source_action'not in event:continue
                out['review_calls']+=1;out['fresh_source_fetches']+=event['source_action']['new_sources']
                if not event['valid']:out['invalid_model_reports']+=1;continue
                patch=event['report']['C']['proposed_patch'];op=patch.get('operation');out['explicit_abstentions']+=op=='abstain'
                out['patches_with_operands']+=op=='abstain'or(op=='set_answer'and isinstance(patch.get('value'),str))or(op=='swap_candidate'and isinstance(patch.get('a'),str)and isinstance(patch.get('b'),str))
                if not event.get('patch_accepted'):out['not_applied_'+event.get('patch_reason','unknown')]+=1;continue
                out['accepted_updates']+=1
                before=fingerprint(event['before_decision']);after=fingerprint(event['after_decision']);seen.add(before)
                revisits=after in seen;cycle+=revisits;out['decision_state_revisits']+=revisits;seen.add(after)
                source=event['source_action']['source_id'];cited=source in patch.get('evidence_ids',[])
                out['accepted_updates_citing_just_fetched_source']+=cited
                out['accepted_updates_not_citing_just_fetched_source']+=not cited
                details.append({'task_id':row['task_id'],'method':method,'slot':event['slot'],'operation':op,'state_revisit':revisits,'just_fetched_source_cited':cited,'fetched_source':source,'patch_evidence_ids':patch.get('evidence_ids',[])})
            out['arms_with_state_revisit']+=cycle>0
    result={'scope':'Post-generation engineering/behavior diagnosis only; accepted update means operational validity, not task correction or semantic truth. No labels or APIs read. Repeated evidence and state cycles may reveal missing contribution/stop control.','generation_complete_sha256':sha(root/'generation_complete.json'),'source_sha256':sha(Path(__file__)),'by_kind_method':{k:dict(v)for k,v in sorted(counts.items())},'accepted_update_details':details}
    create(root/'behavioral_update_diagnostic.json',result);print(json.dumps({k:v for k,v in result.items()if k!='accepted_update_details'},ensure_ascii=False,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();audit(args.output)
