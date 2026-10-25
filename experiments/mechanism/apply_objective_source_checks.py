"""Offline developmental source-check repair; no additional API calls.

Numerical checks may override draft/LLM-review answers only when applicable.
Every other question retains the v3 conservative guard output. Not final test.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.core.mechanism.source_checks import check_tea_comparison
from experiments.mechanism.analyze_real_qa_payoffs import paired_summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('guard_directory',type=Path)
    p.add_argument('retrieval_directory',type=Path)
    p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args()
    guard=args.guard_directory;retrieval=args.retrieval_directory;out=args.out_dir
    out.mkdir(parents=True,exist_ok=True)
    sources=json.loads((guard/'sources.json').read_text())
    cases=json.loads((retrieval/'private_cases.json').read_text())
    questions={c['dataset']+'/'+c['question_id']:c['question'] for c in cases}
    outcomes=json.loads((guard/'outcomes.json').read_text())
    results=[];checks={}
    # Checks see question/source only, not baseline answers or target labels.
    for key,question in questions.items():
        checks[key]=check_tea_comparison(question,sources[key])
    for row in outcomes:
        check=checks[row['dataset']+'/'+row['question_id']]
        final=check['stance'] if check['status']=='checked_against_source' else row['stance']
        results.append({**row,'condition':'objective_source_guard', 'source_check':check,
                        'stance':final,'correct':final==row['ground_truth'],
                        'decision_policy':'typed source computation where applicable; otherwise v3 guard'})
    paired=[]
    for row in results:
        paired.extend([{**row,'action':'direct','correct':row['baseline_correct']},
                       {**row,'action':'self_check','correct':row['correct']}])
    result={'scope':'post-hoc developmental task-specific repair; no independent test claim',
        'n':len(results),'baseline_correct':sum(r['baseline_correct'] for r in results),
        'final_correct':sum(r['correct'] for r in results),
        'corrected':sum(r['correct'] and not r['baseline_correct'] for r in results),
        'harmed':sum(not r['correct'] and r['baseline_correct'] for r in results),
        'objective_checked_questions':sum(c['status']=='checked_against_source' for c in checks.values()),
        'paired_analysis':paired_summary(paired),
        'checker_sha256':hashlib.sha256((ROOT/'experiments/core/mechanism/source_checks.py').read_bytes()).hexdigest(),
        'new_api_calls':0,'llm_autonomy_claim':False,
        'caveats':['source-relative arithmetic, not guaranteed source truth','typed tea checker has narrow coverage',
                   'same questions inspected during development','conservative fallback retains unverified baseline, not verified truth']}
    (out/'source_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    (out/'outcomes.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
