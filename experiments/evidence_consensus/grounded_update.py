"""Opt-in successor to E1: reject decision cycles without newly cited evidence.

This conservative operational policy is not semantic certification. Old-source
new reasoning can be valid, so refusals are logged rather than called false.
"""
import copy
from experiments.consensus_recommendation.common import fingerprint
from .state import apply_patch


def grounded_patch(state, report, required_source=None):
    patch=report.get('C',{}).get('proposed_patch',{})
    def reject(reason):return {'accepted':False,'reason':reason,'state':copy.deepcopy(state),'semantic_truth_certified':False}
    if patch.get('operation')=='abstain':return apply_patch(state,patch)
    cited=set(patch.get('evidence_ids',[]))
    if not cited<=set(report.get('E',[])):return reject('patch_evidence_not_in_report')
    linked={e.get('source_id')for e in report.get('I',[])if e.get('relation')in ('support','refute')}
    if not cited or not cited&linked:return reject('unlinked_patch_evidence')
    if required_source is not None and required_source not in cited:return reject('assigned_source_not_used')
    result=apply_patch(state,patch)
    if not result['accepted']:return result
    history=state.get('grounded_history',[])
    target=fingerprint(result['state']['decision'])
    seen={fingerprint(state['decision'])}|{h['before']for h in history}|{h['after']for h in history}
    prior=set().union(*(set(h['evidence_ids'])for h in history)) if history else set()
    if target in seen and not cited-prior:return reject('decision_cycle_without_new_cited_evidence')
    result['state']['grounded_history']=history+[{'before':fingerprint(state['decision']),'after':target,'evidence_ids':sorted(cited)}]
    return result
