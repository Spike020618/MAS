"""Evaluate only after run seal; no model calls."""
import collections,json,math
from pathlib import Path
from experiments.consensus_recommendation.common import create,fingerprint


def evaluate(root, label_reader):
    root=Path(root);plan=json.loads((root/'plan.json').read_text());seal=json.loads((root/'generation_complete.json').read_text())
    cases=[json.loads((root/'cases'/('case_'+str(i)+'.json')).read_text())for i in range(plan['count'])]
    if seal['hashes']!=[fingerprint(c)for c in cases]:raise ValueError('generation seal mismatch')
    tasks=json.loads((root/'inputs.json').read_text());labels=label_reader(tasks,generation_complete=root/'generation_complete.json');create(root/'labels.after_generation.json',labels)
    out={'scope':'E1 real online integration only; contribution heuristic uncalibrated; not a confirmatory method-effect experiment','methods':{}}
    for method in plan['methods']:
        qa_n=qa_ok=rec_n=rec_hits=0;accepted=rejected=calls=0
        for c in cases:
            arm=c['methods'][method];pred=arm['decision'];gold=labels[c['task_id']]
            if c['kind']=='qa':qa_n+=1;qa_ok+=str(pred).strip().lower()==str(gold['answer']).strip().lower()
            else:
                target=list(map(str,gold['targets']));rec_n+=len(target);rec_hits+=sum(t in list(map(str,pred))[:10]for t in target)
            accepted+=sum(e.get('patch_accepted',False)for e in arm['events']);rejected+=sum(e.get('patch_rejected',False)for e in arm['events']);calls+=arm['logical_calls']
        out['methods'][method]={'qa_correct':qa_ok,'qa_tasks':qa_n,'qa_accuracy':qa_ok/qa_n if qa_n else None,'rec_hits_at_10':rec_hits,'rec_targets':rec_n,'rec_recall_at_10':rec_hits/rec_n if rec_n else None,'accepted_patches':accepted,'rejected_patches':rejected,'logical_calls':calls}
    create(root/'summary.json',out);return out
