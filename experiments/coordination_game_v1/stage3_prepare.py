"""Zero-API stage 3 frozen cohort: 60 undeveloped HotpotQA questions for the comparison matrix.

Same structural selection as the P1 diagnostic cohort (public validation, hash order, >=6
context sources, recorded-exposure exclusion) with a larger count; the P1 twelve are excluded
automatically because their roots are indexed by the exposure audit."""
import argparse,hashlib,json,re
from pathlib import Path
from .data import project,partition
from .exposure import audit
from .schema import digest

ROOT=Path(__file__).resolve().parents[2]

def load_public(path):
    import pyarrow.parquet as pq
    rows=pq.read_table(path,columns=['id','question','context']).to_pylist()
    return [{'_id':r['id'],'question':r['question'],'context':list(zip(r['context']['title'],r['context']['sentences']))}for r in rows]

def p1_exclusion():
    """The P1 diagnostic roots store task lists in tasks.json/plan.json, which the generic
    inputs.json audit does not index; exclude their ids and question keys explicitly.
    Also excludes prior stage-3 cohorts so power extensions draw fresh questions."""
    ids=set();questions=set()
    for p in [ROOT/'experiments/results/coordination_game_p1_prepare_v1/tasks.json',
              ROOT/'experiments/results/coordination_game_p1_v3/plan.json',
              ROOT/'experiments/results/coordination_game_p1_v4/plan.json',
              ROOT/'experiments/results/coordination_game_s3_prepare_v1/tasks.json',
              ROOT/'experiments/results/coordination_game_s3_power_prepare_v1/tasks.json']:
        if not p.exists():continue
        data=json.loads(p.read_text())
        items=data if isinstance(data,list)else data.get('task_ids',[])
        if isinstance(items,list):
            for x in items:
                s=x.get('id')if isinstance(x,dict)else x
                ids.add(str(s))
                if isinstance(x,dict)and isinstance(x.get('question'),str):
                    from .data import question_key
                    questions.add(question_key(x['question']))
    return ids,questions

def prepare(output,count=60,source=None,results=None):
    source=Path(source or ROOT/'experiments/data/public/hotpotqa/validation.parquet');output=Path(output)
    if output.exists():raise ValueError('prepare output already exists; never overwrite')
    public=load_public(source);exposure=audit(results or ROOT/'experiments/results')
    native_ids={r['_id']for r in public};mapped=set()
    for id in exposure['ids']:
        for match in re.findall(r'[a-f0-9]{24}',id):
            if match in native_ids:mapped.add(match)
    p1_ids,p1_questions=p1_exclusion()
    split=partition(public,set(exposure['ids'])|mapped|p1_ids,set(exposure['question_keys'])|p1_questions)
    # Structural requirement only (not difficulty): three member-initial sources plus three
    # action candidates for the budget-scarce comparison matrix.
    split['excluded']=[e for e in split['excluded']]+[{'id':r['id'],'reason':'insufficient_sources'}for r in split['eligible']if len(r['sources'])<6]
    split['eligible']=[r for r in split['eligible']if len(r['sources'])>=6]
    chosen=split['eligible'][:count]
    if len(chosen)!=count:raise ValueError(f'insufficient eligible public records: {len(split["eligible"])}<{count}')
    ids={x['id']for x in chosen}
    tasks=[]
    for task in chosen:
        sources=task['sources'];order=sorted([s['id']for s in sources],key=lambda sid:digest([task['id'],sid]))
        tasks.append({**task,'member_initial_sources':{str(i):order[i::3][:1]for i in range(3)},
                      'source_order':order,'candidates':order[3:6]})
    # Labels read only after the selection freezes in memory.
    import pyarrow.parquet as pq
    labels=pq.read_table(source,columns=['id','answer','supporting_facts']).to_pylist()
    vault={r['id']:{'answer':r['answer'],'supporting_facts':r['supporting_facts']}for r in labels if r['id']in ids}
    output.mkdir(parents=True);(output/'private').mkdir(mode=0o700)
    def write(name,data):
        p=output/name;p.write_text(json.dumps(data,ensure_ascii=False,indent=2));return hashlib.sha256(p.read_bytes()).hexdigest()
    tasks_hash=write('tasks.json',tasks);vault_hash=write('private/scoring.json',vault);(output/'private/scoring.json').chmod(0o600)
    write('exposure.json',exposure)
    manifest={'schema':'coordination-stage3-data.v1','source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
       'source_split':'public_validation','cohort_role':'frozen_stage3_comparison','fresh_test_certified':False,
       'selection':'public ID hash order after recorded exact-ID and normalized-question exclusions (P1 twelve excluded via indexed roots); structural minimum of six context sources; no answer/support/difficulty filtering',
       'source_records':len(public),'eligible_records':len(split['eligible']),'excluded_records':len(split['excluded']),
       'tasks':count,'task_ids':[t['id']for t in tasks],'tasks_sha256':tasks_hash,'private_vault_sha256':vault_hash,
       'budget_reads_per_question':2,'paid_requests':0,
       'limitations':['Only locally recorded historical inputs are indexed; unrecorded manual exposure cannot be certified absent.',
        'Public validation cohort is development only; a fresh partition is required before any confirmatory claim.',
        'Shared encyclopedia passages retained; no source-disjoint generalization claim.']}
    write('manifest.json',manifest);write('excluded.json',split['excluded'])
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'experiments/results/coordination_game_s3_prepare_v1'))
    p.add_argument('--count',type=int,default=60);a=p.parse_args()
    print(json.dumps(prepare(a.out,a.count),ensure_ascii=False,indent=2))
