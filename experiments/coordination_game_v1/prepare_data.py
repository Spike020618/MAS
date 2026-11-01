"""Zero-API P1 development cohort. Selection never loads answer columns."""
import argparse,hashlib,json,re
from pathlib import Path
from .data import project,partition,question_key
from .exposure import audit
from .schema import digest

ROOT=Path(__file__).resolve().parents[2]
def load_public(path):
    import pyarrow.parquet as pq
    rows=pq.read_table(path,columns=['id','question','context']).to_pylist()
    return [{'_id':r['id'],'question':r['question'],'context':list(zip(r['context']['title'],r['context']['sentences']))}for r in rows]

def prepare(output,source=None,count=12,repeats=2,results=None):
    source=Path(source or ROOT/'experiments/data/public/hotpotqa/validation.parquet');output=Path(output)
    if output.exists():raise ValueError('prepare output already exists; never overwrite')
    if count!=12 or repeats!=2:raise ValueError('first P1 cohort is fixed at 12 x 2')
    public=load_public(source);exposure=audit(results or ROOT/'experiments/results')
    native_ids={r['_id']for r in public};mapped=set()
    for id in exposure['ids']:
        for match in re.findall(r'[a-f0-9]{24}',id):
            if match in native_ids:mapped.add(match)
    split=partition(public,set(exposure['ids'])|mapped,exposure['question_keys'])
    # Structural requirement (not difficulty filtering): the matched 16-branch design needs
    # 3 member-initial sources plus 3 action sources per question.
    structural=[r for r in split['eligible']if len(r['sources'])>=6]
    split['excluded']=[e for e in split['excluded']]+[{'id':r['id'],'reason':'insufficient_sources'}for r in split['eligible']if len(r['sources'])<6]
    split['eligible']=structural
    # Deterministic hash order, no difficulty/answer/support-fact filtering.
    chosen=split['eligible'][:count]
    if len(chosen)!=count:raise ValueError('insufficient eligible public records')
    ids={x['id']for x in chosen}
    tasks=[]
    for task in chosen:
        sources=task['sources'];order=sorted([s['id']for s in sources],key=lambda sid:digest([task['id'],sid]))
        tasks.append({**task,'member_initial_sources':{str(i):order[i::3][:1]for i in range(3)},'source_order':order,'repeats':[0,1]})
    # Only after selection freeze in memory, read labels for selected rows into private vault.
    import pyarrow.parquet as pq
    labels=pq.read_table(source,columns=['id','answer','supporting_facts']).to_pylist()
    vault={r['id']:{'answer':r['answer'],'supporting_facts':r['supporting_facts']}for r in labels if r['id']in ids}
    output.mkdir(parents=True);(output/'private').mkdir(mode=0o700)
    def write(name,data):
        p=output/name;p.write_text(json.dumps(data,ensure_ascii=False,indent=2));return hashlib.sha256(p.read_bytes()).hexdigest()
    tasks_hash=write('tasks.json',tasks);vault_hash=write('private/scoring.json',vault);(output/'private/scoring.json').chmod(0o600)
    write('exposure.json',exposure)
    manifest={'schema':'coordination-p1-data.v1','source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_split':'public_validation','cohort_role':'development_diagnostic','fresh_test_certified':False,
       'selection':'public ID hash order after recorded exact-ID and normalized-question exclusions; structural minimum of six context sources for the matched branch design; no answer/support/difficulty filtering',
       'source_records':len(public),'eligible_records':len(split['eligible']),'excluded_records':len(split['excluded']),'mapped_native_exposed_ids':len(mapped),'tasks':count,'repeats':repeats,'task_ids':[t['id']for t in tasks],
       'tasks_sha256':tasks_hash,'private_vault_sha256':vault_hash,'near_duplicate_review_required':True,'paid_requests':0,
       'limitations':['Only locally recorded historical inputs are indexed; unrecorded manual exposure cannot be certified absent.','Public validation cohort is development only.','Shared encyclopedia passages are retained; no source-disjoint generalization claim.','No truncation or token-admission filtering at selection; runner must report oversized branches without replacement.']}
    write('manifest.json',manifest);write('excluded.json',split['excluded']);return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'experiments/results/coordination_game_p1_prepare_v1'));a=p.parse_args();print(json.dumps(prepare(a.out),ensure_ascii=False,indent=2))
