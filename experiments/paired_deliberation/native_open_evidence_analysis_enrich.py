"""Attach post-seal action ceilings to independent analysis, never model data."""
import argparse,json
from pathlib import Path
from experiments.consensus_recommendation.common import sha
from experiments.paired_deliberation.native_open_evidence_analysis import load_sealed

def enrich(root):
    root=Path(root);load_sealed(root)
    selection=root/'selection_audit.json';analysis=root/'analysis.json'
    if not selection.exists() or not analysis.exists():raise ValueError('both independent analyses required')
    data=json.loads(analysis.read_text());diag=json.loads(selection.read_text())
    data['selection_diagnostic']={key:diag[key] for key in ('coverage','per_method','rejected_out_of_pair_diagnostic','shared_omission_examples','oracle_rule','interpretation')}
    data['selection_diagnostic_provenance']={'selection_file_sha256':sha(selection),'enrichment_source_sha256':sha(Path(__file__)),
       'primary_metrics_unchanged':True,'generation_summary_case_files_unchanged':True,'labels_only_after_generation_seal':True}
    analysis.write_text(json.dumps(data,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    return {'root':str(root),'coverage':diag['coverage']}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',action='append',required=True);a=p.parse_args()
    for root in a.root:print(json.dumps(enrich(root)))
