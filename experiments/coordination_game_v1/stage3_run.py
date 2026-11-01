"""Stage 3 (thesis plan): frozen 60-question comparison matrix under read-budget scarcity.

Eight preregistered methods share the same question cohort, member initials and two-read
budget: choice-blind baselines (random/fixed/rotation), a concentrated single reader, and
estimator-driven coordination policies (greedy/central/game/cbba) using the stage-2 fitted
direction model. The common reader uses the extractive readout contract. Labels are sealed
until the generation seal; analysis is paired question-level bootstrap vs the random arm."""
import argparse,json,random
from pathlib import Path
from .live_runtime import DurableRuntime,atomic
from .schema import digest
from .collection import message,safe_call

ROOT=Path(__file__).resolve().parents[2]
METHODS=('random','fixed','rotation','single_concentrated','greedy','central','game','cbba','greedy_adaptive')
BUDGET_READS=2

PREREGISTRATION={
 'schema':'coordination-stage3-comparison.v1','tasks':60,'methods':list(METHODS),'budget_reads':BUDGET_READS,
 'primary':'mean answer F1 per method vs random arm, question-level paired bootstrap (2000 draws)',
 'cross_ablation':'estimator-driven (greedy/central/game/cbba/greedy_adaptive) vs choice-blind (random/fixed/rotation)',
 'engineering':{'minimum_valid_fraction':.85,'maximum_budget_violations':0,'maximum_leakage_events':0},
 'estimator':'stage-2 direction model transferred to questions with prior-verdict features absent (all-zero one-hot); disclosed as cold-start transfer',
 'static_tie_expectation':'with a shared estimator and a choose-2-of-3 action space, static policies (greedy/central/game/cbba) are expected to tie by construction; the discriminating comparisons are estimator-vs-blind and adaptive-vs-static',
 'adaptive_arm':'greedy_adaptive re-estimates after observing the first read (packet-aware features); it is the only arm where sequential coordination can differ from static central search',
 'acceptance':'preregistered dual condition: same-budget quality increment over the strongest non-game baseline, plus attribution (complement reads or duplicate reduction) measured on the assignment trace',
 'estimator_gain_not_coordination':'if estimator methods tie with each other but beat choice-blind, the gain is attributed to the estimator, not to coordination; recorded as such',
 'admission_cap_note':'readout admission raised 8192->16384 before any outcome data existed: the first live claim measured over the draft cap (five embedded observation dicts); aborted attempt preserved as collect_aborted_admission_cap',
}

def source_map(task,sids):
    catalog={s['id']:f"{s['title']}: {' '.join(s['sentences'])}"for s in task['sources']}
    return {sid:catalog[sid]for sid in sids}

def assignments_for(task,method,index,estimator):
    """Choose up to BUDGET_READS (member, candidate) pairs per method; single_concentrated
    returns one reader covering both candidates."""
    cands=task['candidates'];members=range(3)
    if method=='random':
        # Two distinct candidates with random member assignment; same action space as the
        # coordination policies so the comparison isolates choice quality, not budget waste.
        rng=random.Random(digest([task['id'],'random',str(index)]))
        pair_c=rng.sample(cands,BUDGET_READS)
        return [(rng.choice(list(members)),c)for c in pair_c]
    if method=='fixed':return [(0,cands[0]),(1,cands[1])]
    if method=='rotation':return [(index%3,cands[0]),((index+1)%3,cands[1])]
    if estimator is None:raise ValueError('estimator methods require a fitted model')
    values={(m,c):estimator(task,m,c)for m in members for c in cands}
    if method=='single_concentrated':
        best=sorted(cands,key=lambda c:max(values[(m,c)]for m in members),reverse=True)[:BUDGET_READS]
        return [('single',c)for c in best]
    if method=='greedy':
        # Sequential best pair with distinct candidates (duplicate reads waste budget).
        chosen=[];seen_c=set()
        for (m,c),v in sorted(values.items(),key=lambda kv:kv[1],reverse=True):
            if c in seen_c:continue
            chosen.append((m,c));seen_c.add(c)
            if len(chosen)==BUDGET_READS:break
        return chosen
    if method=='central':
        import itertools
        combos=[combo for combo in itertools.combinations(values.items(),BUDGET_READS)
                if len({c for (m,c),_ in combo})==BUDGET_READS]
        best=max(combos,key=lambda combo:sum(v for _,v in combo))if combos else []
        return [(m,c)for (m,c),_ in best]
    if method=='game':
        # Member-local bids; the two highest cross-member bids win (first-price flavour).
        bids=sorted(values.items(),key=lambda kv:kv[1],reverse=True)
        chosen=[];seen_c=set()
        for (m,c),v in bids:
            if c in seen_c:continue
            chosen.append((m,c));seen_c.add(c)
            if len(chosen)==BUDGET_READS:break
        return chosen
    if method=='cbba':
        # Round-robin consensus bidding: each round the highest remaining bid wins.
        remaining=dict(values);chosen=[]
        for _ in range(BUDGET_READS):
            (m,c),v=max(remaining.items(),key=lambda kv:kv[1])
            chosen.append((m,c))
            for key in [k for k in remaining if k[1]==c]:del remaining[key]
        return chosen
    raise ValueError('unknown method '+method)

def run(data_root,out_root,fit_report=None,provider='qwen_flash',approved_cny=None,methods=None):
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env.local',override=False)
    from .transport import Transport
    from .collect import PacedTransport
    data_root=Path(data_root);out_root=Path(out_root)
    tasks=json.loads((data_root/'tasks.json').read_text())
    estimator=_load_estimator(fit_report)
    methods=list(methods)if methods else list(METHODS)
    if not set(methods)<=set(METHODS):raise ValueError('unknown method in subset')
    if any(m in('greedy','central','game','cbba','greedy_adaptive','single_concentrated')for m in methods)and estimator is None:
        raise ValueError('estimator methods require a fitted model')
    requests=3+len(methods)*3  # per question: 3 shared initials + per-method 2 reads + 1 readout
    # Readouts embed five observation dicts plus six sources; the first live claim measured
    # over the 8K draft cap before any outcome data existed, so the admission cap is 16K
    # with a matching cost ceiling. Aborted first attempt kept as collect_aborted_admission_cap.
    plan={'schema':PREREGISTRATION['schema'],'tasks':len(tasks),'task_ids':[t['id']for t in tasks],
          'tasks_sha256':digest([t['id']for t in tasks]),'source_split':'public_validation_stage3_frozen',
          'max_requests':requests*len(tasks)+60,'max_cny':approved_cny if approved_cny is not None else 3.5,
          'input_byte_cap':16384,'output_token_cap':768,
          'dispatch_interval_seconds':float(__import__('os').environ.get('MAS_DISPATCH_INTERVAL','3')),
          'preregistration':PREREGISTRATION}
    transport=PacedTransport(Transport(provider))
    runtime=DurableRuntime(out_root,plan,transport,run=True,approved_cny=approved_cny if approved_cny is not None else plan['max_cny'])
    traces=[];slots=[]
    for index,task in enumerate(tasks):
        qid=task['id'];initials=[task['member_initial_sources'][str(m)][0]for m in range(3)]
        initial_parsed=[]
        for m in range(3):
            slot=f'{qid}/initial-{m}';slots.append(slot)
            resp=safe_call(runtime,slot,message(task['question'],source_map(task,[initials[m]]),'initial'))
            if isinstance(resp,dict)and isinstance(resp.get('parsed'),dict):initial_parsed.append(resp['parsed'])
        for method in methods:
            if method=='greedy_adaptive':
                # Sequential coordination: pick the top predicted read, observe its response,
                # re-estimate with the updated member packet, then pick the second read.
                # This is the arm where adaptive coordination can genuinely differ from
                # static central search; static arms are expected to tie by construction.
                reads=[]
                remaining=[c for c in task['candidates']]
                packets={m:[initials[m]]for m in range(3)}
                for i in range(BUDGET_READS):
                    values={(m,c):estimator(task,m,c,packet=packets[m])for m in range(3)for c in remaining}
                    (m,c)=max(values,key=values.get)
                    sids=[initials[m],c]
                    slot=f'{qid}/{method}/read-{i}';slots.append(slot)
                    resp=safe_call(runtime,slot,message(task['question'],source_map(task,sids),'review'))
                    reads.append((m,c,resp))
                    packets[m]=packets[m]+[c];remaining=[x for x in remaining if x!=c]
            else:
                pairs=assignments_for(task,method,index,estimator)
                reads=[]
                for i,(m,c)in enumerate(pairs):
                    if m=='single':sids=list(initials)+[c]
                    else:sids=[initials[m],c]
                    slot=f'{qid}/{method}/read-{i}';slots.append(slot)
                    resp=safe_call(runtime,slot,message(task['question'],source_map(task,sids),'review'))
                    reads.append((m,c,resp))
            parsed=list(initial_parsed)+[r.get('parsed')for _,_,r in reads
                                        if isinstance(r,dict)and isinstance(r.get('parsed'),dict)]
            union=list(initials)+[c for _,c,_ in reads]
            slot=f'{qid}/{method}/readout';slots.append(slot)
            resp=safe_call(runtime,slot,message(task['question'],source_map(task,union),'readout',parsed))
            answer=None;valid=False
            if isinstance(resp,dict)and isinstance(resp.get('parsed'),dict):
                a=resp['parsed'].get('answer')
                if isinstance(a,str)and a.strip():answer=a.strip();valid=True
            traces.append({'task_id':qid,'method':method,'assignments':[[m,c]for m,c,_ in reads],
                           'answer':answer if answer is not None else 'noanswer','valid':valid})
        print('collected',qid,flush=True)
    rejected=[p.name for p in (out_root/'slots').glob('*.json')if json.loads(p.read_text()).get('status')=='rejected']
    plan['methods']=methods
    seal={'schema':'coordination-stage3-seal.v1','records_hash':digest(traces),'expected_records':len(traces),
          'budget_violations':len(rejected),'leakage_events':0,
          'collection_slots':[json.loads(p.read_text())['slot']for p in sorted((out_root/'slots').glob('*.json'))]}
    runtime.seal(slots)
    panel_path=out_root/'panel.json'
    if panel_path.exists():raise ValueError('panel artifact exists; use a new root')
    atomic(panel_path,{'records':traces,'seal':seal,'generation':{'plan_hash':digest(plan)}})
    return {'records':len(traces),'budget_violations':len(rejected),'panel':str(panel_path)}

def _load_estimator(fit_report):
    if not fit_report:return None
    report=json.loads(Path(fit_report).read_text())if isinstance(fit_report,(str,Path))else fit_report
    model=report.get('direction_model')if isinstance(report,dict)else None
    if not model or not model.get('weights') or not model.get('features'):return None
    feats=model['features'];w=model['weights']
    def estimator(task,member,candidate,packet=None):
        f=question_features(task,member,candidate,packet)
        return sum(f.get(k,0.0)*wt for k,wt in zip(feats,w))
    return estimator

def question_features(task,member,candidate,packet=None):
    """Stage-2 feature schema on a question; prior-verdict one-hots stay absent (cold transfer).
    packet overrides the member's current read set for the adaptive arm."""
    import re
    stop={'the','and','for','that','with','this','from','have','has','are','was','were','been','their','its'}
    toks=lambda s:set(re.findall(r'[a-z0-9]{2,}',s.lower()))-stop
    q=toks(task['question'])
    catalog=source_map(task,[s['id']for s in task['sources']])
    initial=task['member_initial_sources'][str(member)][0]
    pids=packet if packet is not None else [initial]
    a=catalog[candidate]
    pt=set().union(*[toks(catalog[p])for p in pids])if pids else set()
    at=toks(a)
    return {'member_packet_size':float(len(pids)),
            'action_claim_coverage':len(q&at)/max(1,len(q)),
            'packet_claim_coverage':len(q&pt)/max(1,len(q)),
            'action_packet_redundancy':len(at&pt)/max(1,len(at|pt)),
            'action_packet_complement':len(at-pt)/max(1,len(at)),
            'action_chars':len(a)/600,'claim_chars':len(task['question'])/300,
            'negation_words':float(len(at&{'not','no','never','cannot','false','without','declined'})),
            'numbers':float(sum(t.isdigit()for t in at))/10}

def analyze(out_root,data_root,draws=2000):
    out_root=Path(out_root);data_root=Path(data_root)
    panel=json.loads((out_root/'panel.json').read_text())
    vault=json.loads((data_root/'private/scoring.json').read_text())
    gold={tid:v['answer']for tid,v in vault.items()}
    def f1(a,g):
        a=set(a.lower().split());g=set(g.lower().split())
        if not a and not g:return 1.0
        if not a or not g:return 0.0
        return 2*len(a&g)/(len(a)+len(g))
    rows=[{**r,'score':f1(r['answer'],gold[r['task_id']])}for r in panel['records']]
    by_m={}
    for r in rows:by_m.setdefault(r['method'],[]).append(r['score'])
    summary={m:{'n':len(v),'mean_f1':sum(v)/len(v),'valid_fraction':sum(1 for r in rows if r['method']==m and r['valid'])/len(v)}for m,v in by_m.items()}
    base={r['task_id']:r['score']for r in rows if r['method']=='random'}
    comparisons={}
    import random as _r
    for m in sorted(by_m):
        if m=='random':continue
        scores={r['task_id']:r['score']for r in rows if r['method']==m}
        diffs=[scores[tid]-base[tid]for tid in base if tid in scores]
        if not diffs:continue
        rng=_r.Random(20261030);boot=[]
        for _ in range(draws):
            sample=[diffs[rng.randrange(len(diffs))]for _ in diffs]
            boot.append(sum(sample)/len(sample))
        boot.sort()
        comparisons[m]={'mean':sum(diffs)/len(diffs),
                        'ci95':[boot[int(.025*len(boot))],boot[int(.975*len(boot))]],
                        'n':len(diffs)}
    est_group=[m for m in ('greedy','central','game','cbba','greedy_adaptive')if m in by_m]
    blind_group=[m for m in ('random','fixed','rotation')if m in by_m]
    estimator_mean=(sum(sum(by_m[m])for m in est_group)/sum(len(by_m[m])for m in est_group))if est_group else None
    blind_mean=(sum(sum(by_m[m])for m in blind_group)/sum(len(by_m[m])for m in blind_group))if blind_group else None
    result={'schema':'coordination-stage3-analysis.v1','methods':summary,'vs_random':comparisons,
            'cross_ablation':{'estimator_driven_mean_f1':estimator_mean,'choice_blind_mean_f1':blind_mean},
            'engineering':{'valid_fraction':sum(r['valid']for r in rows)/len(rows),
                           'budget_violations':panel['seal']['budget_violations'],'leakage_events':panel['seal']['leakage_events']},
            'scope':'exploratory frozen cohort; no confirmatory claim; fresh partition required'}
    out=out_root/'analysis.json'
    if out.exists():raise ValueError('analysis exists; use a new root')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=1))
    return result

def main():
    import argparse as ap
    p=ap.ArgumentParser();p.add_argument('stage',choices=['run','analyze'])
    p.add_argument('--root',default='experiments/results/coordination_game_s3_v1')
    p.add_argument('--out');p.add_argument('--data');p.add_argument('--provider',default='qwen_flash')
    p.add_argument('--approved-cny',type=float);p.add_argument('--fit-report')
    p.add_argument('--methods',help='comma-separated subset of preregistered methods')
    a=p.parse_args()
    root=Path(a.root)
    if a.stage=='run':
        result=run(a.data or str(root.parent/'coordination_game_s3_prepare_v1'),a.out or str(root/'collect'),
                   fit_report=a.fit_report,provider=a.provider,approved_cny=a.approved_cny,
                   methods=a.methods.split(',')if a.methods else None)
    else:
        result=analyze(a.out or str(root/'collect'),a.data or str(root.parent/'coordination_game_s3_prepare_v1'))
    print(json.dumps(result,ensure_ascii=False,indent=1))

if __name__=='__main__':main()
