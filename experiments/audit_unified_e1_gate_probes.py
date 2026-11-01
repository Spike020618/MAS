"""Counterfactual gate probes on sealed E1 states; no labels or API."""
import argparse,json
from pathlib import Path
from experiments.consensus_recommendation.common import ROOT,create,fingerprint
from experiments.evidence_consensus.state import apply_patch


def audit(root):
    root=Path(root);seal=json.loads((root/'generation_complete.json').read_text())
    rows=[json.loads((root/'cases'/('case_'+str(i)+'.json')).read_text())for i in range(len(seal['task_ids']))]
    if seal['hashes']!=[fingerprint(x)for x in rows]:raise ValueError('sealed state mismatch')
    out={'scope':'counterfactual probes against sealed method states; no gold labels or API','stale_version_probes':0,'stale_version_rejected':0,'unfetched_source_probes':0,'unfetched_source_rejected':0,'unknown_receipts':[]}
    for row in sorted(rows,key=lambda x:x['task_id']):
        for method,arm in sorted(row['methods'].items()):
            s=arm['state'];patch={'expected_version':s['version']+1,'evidence_ids':list(s['visible_sources'])[:1],'preserve_ids':[]}
            if s['kind']=='qa':patch.update(operation='set_answer',value='yes'if s['decision']!='yes'else'no')
            else:patch.update(operation='swap_candidate',a=s['decision'][0],b=s['decision'][-1])
            a=apply_patch(s,patch);out['stale_version_probes']+=1
            out['stale_version_rejected']+=not a['accepted']and a['state']==s
            missing=next((sid for sid in s['catalog']if sid not in s['visible_sources']),None)
            if missing:
                patch.update(expected_version=s['version'],evidence_ids=[missing]);a=apply_patch(s,patch);out['unfetched_source_probes']+=1
                out['unfetched_source_rejected']+=not a['accepted']and a['state']==s
            for key in arm['budget']['unknown_requests']:
                receipt=arm['budget']['receipts'][key]
                out['unknown_receipts'].append({'task_id':row['task_id'],'method':method,'key':key,'charged':receipt['charged'],'reported':receipt['reported']})
    # Stable task order for the original exploratory receipt artifact.
    out['unknown_receipts'].sort(key=lambda x:(x['task_id'],x['method'],x['key']))
    if out['stale_version_probes']!=out['stale_version_rejected']or out['unfetched_source_probes']!=out['unfetched_source_rejected']:raise ValueError('gate failure')
    create(root/'independent_state_gate_probes.sorted.json',out);print(json.dumps(out,ensure_ascii=False,indent=2));return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'experiments/results/unified_e1_v3');a=p.parse_args();audit(a.output)
