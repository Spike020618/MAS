"""Prepare 12 native QA + 12 native recommendation E1 tasks, without labels/API."""
import argparse
from pathlib import Path
from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha
from experiments.evidence_consensus.task_adapter import qa_tasks,rec_tasks,source_index

OUT=ROOT/'experiments/results/evidence_consensus_e1_tasks_v1'


def prepare(output=OUT):
    output=Path(output)
    qa,skipped=qa_tasks(12);rec=rec_tasks(12);tasks=qa+rec
    if len({t['id']for t in tasks})!=24:raise ValueError('unique 24 task IDs required')
    for task in tasks:
        if not task['source_catalog']:raise ValueError('actual material source required')
        if any(card['scope']['task_id']!=task['id']for card in task['source_catalog'].values()):raise ValueError('unbound task source')
        if task['metadata']['query_labels_available'] is not False:raise ValueError('preparation must not carry labels')
    create(output/'tasks.json',tasks)
    create(output/'source_indexes.json',{t['id']:source_index(t)for t in tasks})
    plan={'version':'unified-evidence-consensus-e1-native24-v1','count':24,'qa':12,'rec':12,
          'task_ids':[t['id']for t in tasks],'tasks_fingerprint':fingerprint(tasks),
          'selection':'native public QA cases with cached independent web materials; first12 sealed prefix-only Rec rows',
          'same_source_permissions_for_all_methods':True,
          'source_expansion':'catalog metadata may be discovered; actual source text is obtained by a logged fetch action',
          'query_label_files_read':False,'gold_supporting_facts_used_as_retrieval':False,'model_api_calls':0,
          'history':'all reused development sources, engineering closed-loop test only; not independent method confirmation',
          'qa_skipped_for_missing_material':skipped,
          'source_sha256':{name:sha(ROOT/'experiments/evidence_consensus'/name)for name in ('task_adapter.py','prepare_e1.py')},
          'tasks_sha256':sha(output/'tasks.json'),
          'evaluation_gate':'task_adapter.load_evaluation_labels requires generation_complete task_ids matching every task before reading native labels'}
    create(output/'plan.json',plan)
    print(f'Prepared24: QA={len(qa)}, Rec={len(rec)}, sources={sum(len(t["source_catalog"])for t in tasks)}, no API/labels')
    return tasks,plan

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();prepare(a.output)
