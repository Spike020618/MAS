"""Explicit enum alias adapter; never changes native answers, patches or evidence.

Some supplied-evidence models use the task's NEI answer as the claim relation.
For tasks explicitly declaring NEI as an uncertainty answer, it is equivalent to
an unclear relation. Retain raw reports and record each conversion separately.
"""
import copy
from experiments.consensus_mechanism import core as c

def canonicalize(task, report):
    report=copy.deepcopy(report);operations=[]
    if (isinstance(report,dict) and report.get('verdict')=='not_enough_info'
            and 'not_enough_info' in task.get('metadata',{}).get('uncertainty_answers',[])):
        report['verdict']='unclear'
        operations.append({'field':'verdict','from':'not_enough_info','to':'unclear',
                           'rule':'declared uncertainty answer maps to unknown claim relation'})
    return report,operations

class CanonicalReviewAPI:
    """Optional transport-independent wrapper for NEW releases only."""
    def __init__(self,api,task):self.api=api;self.task=task;self.operations={}
    def call(self,slot,request):
        record=copy.deepcopy(self.api.call(slot,request))
        record['report'],ops=canonicalize(self.task,record['report'])
        self.operations[slot]=ops
        return record

def audit_panel(root,output):
    from experiments import bound_opportunity_probe as b
    from experiments.consensus_mechanism.data import read
    from experiments.consensus_recommendation.common import create,fingerprint,sha
    rows=read(root/'manifest.private.json')
    original=[read(root/'cases'/f'case_{i}.json')for i in range(len(rows))]
    if read(root/'generation_complete.json')!={'manifest_hash':fingerprint(rows),'hashes':[fingerprint(x)for x in original]}:raise ValueError('seal mismatch')
    cases=copy.deepcopy(original);converted=0
    for row,case in zip(rows,cases):
        for name,arm in case['arms'].items():
            if not name.endswith(':linked'):continue
            req_hash=arm['event']['request_hash']
            wrapper=read(root/'slots'/('request_'+req_hash[:32]+'.json'))
            record=wrapper['record']; report,ops=canonicalize(row['task'],record['report'])
            state=copy.deepcopy(row['state']);c.expose(row['task'],state,arm['action']['source_ids'])
            event={'valid':False,'accepted':False,'raw_request_hash':req_hash,'schema_alias_operations':ops}
            converted+=bool(ops)
            try:
                refs,basis=b.validate_linked(row['task'],state,report,arm['action'])
                after,accepted=c.commit(row['task'],state,report,'review',100,refs)
                event.update(valid=True,accepted=accepted,revision_basis=basis)
                arm['decision']=after['decision']
            except (ValueError,KeyError,TypeError)as exc:event['error']=str(exc)
            arm['event']=event
    result=b.summarize(rows,cases)
    result.update(schema_alias_conversions=converted,mode='offline_schema_replay',
      original_root=str(root),adapter_sha256=sha(__file__),
      promotion_gate=False,scope='Post-hoc schema replay, not a fresh confirmatory run. Original strict results remain unchanged; release is not unlocked.')
    create(output/'cases.json',cases);create(output/'summary.json',result)
    return result

if __name__=='__main__':
    import json
    from pathlib import Path
    from experiments.consensus_recommendation.common import ROOT
    print(json.dumps(audit_panel(ROOT/'experiments/results/bound_opportunity_v3',ROOT/'experiments/results/bound_contract_audit_v1'),ensure_ascii=False,indent=2))
