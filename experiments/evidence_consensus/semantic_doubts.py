"""Claim-specific proposal construction; metadata heuristic, never truth certification."""
import re
from experiments.consensus_recommendation.common import fingerprint


def propose_doubts(task,state,reports,limit=8):
    if not 0<limit<=8:raise ValueError('limit must be 1..8')
    catalog=state['catalog'];claims={}
    for report in reports:
        for c in report.get('A',[]):
            if isinstance(c,dict)and c.get('id')and c.get('text'):claims[c['id']]=c
    def sources(text,preferred=()):
        words=set(re.findall(r'[a-z]{3,}',text.lower()));rows=[]
        for sid,s in catalog.items():
            # Deliberately exclude body text, even when available in memory.
            meta=' '.join(str(s.get(k,''))for k in ('title','kind','source_id','movie_id'))
            overlap=len(words&set(re.findall(r'[a-z]{3,}',meta.lower())))
            rows.append((sid,overlap+int(sid in preferred)))
        return [sid for sid,_ in sorted(rows,key=lambda x:(-x[1],x[0]))[:3]]
    proposals=[]
    def add(kind,targets,question,operation,scope,preferred=()):
        targets=sorted(set(targets))
        if not targets or not set(targets)<=set(claims):return
        row={'kind':kind,'target_claim_ids':targets,'question':question,'source_ids':sources(question,preferred),'operation':operation,'expected_decision_scope':scope,'semantic_truth_certified':False}
        row['doubt_id']='doubt:'+fingerprint([kind,targets,question,scope])[:20];proposals.append(row)
    for report in reports:
        for d in report.get('proposed_doubts',[]):
            if not isinstance(d,dict)or d.get('kind')not in ('premise','conflict','omission')or not isinstance(d.get('question'),str):continue
            add(d['kind'],d.get('target_claim_ids',[]),d['question'],d.get('operation','verify_premise'),d.get('expected_decision_scope',{'kind':task['kind']}),d.get('source_ids',[]))
    for cid,c in claims.items():
        edges=[e for r in reports for e in r.get('I',[])if e.get('claim_id')==cid]
        relations={e.get('relation')for e in edges}
        add('premise',[cid],f"Which supplied material supports or contradicts this exact premise: {c['text']}?",'verify_premise',{'kind':task['kind'],'claim_id':cid},[e.get('source_id')for e in edges])
        if {'support','refute'}<=relations:
            add('conflict',[cid],f"Resolve the conflicting support and refutation for: {c['text']}",'seek_counterexample',{'kind':task['kind'],'claim_id':cid})
    decisions=[r.get('decision')for r in reports if r.get('decision')is not None]
    if len({fingerprint(d)for d in decisions})>1 and claims:
        ids=list(claims)[:3]
        add('conflict',ids,'Which of these competing decisions follows from the supplied premises: '+repr(decisions)+'? Premises: '+'; '.join(claims[c]['text']for c in ids),'compare_evidence',{'kind':task['kind'],'decisions':decisions})
    if task['kind']=='rec' and claims:
        current=state['decision'][:10];supplied=[]
        for r in reports:
            for m in r.get('decision',[]):
                if m not in current and m not in supplied:supplied.append(m)
        native={str(c.get('movie_id',c.get('id')))if isinstance(c,dict)else str(c)for c in task['candidates']}
        for m in supplied[:3]:
            if m in native:add('omission',list(claims)[:3],f"Does evidence for omitted candidate {m} justify replacing any current Top10 item while preserving unrelated positions?",'evaluate_candidate',{'kind':'rec','candidate_id':m,'modifiable_positions':list(range(10))})
    unique={p['doubt_id']:p for p in proposals}
    # Round-robin categories so eight premise claims do not suppress conflicts.
    buckets={k:[p for p in unique.values()if p['kind']==k]for k in ('conflict','omission','premise')};out=[]
    while len(out)<limit and any(buckets.values()):
        for k in buckets:
            if buckets[k]and len(out)<limit:out.append(buckets[k].pop(0))
    return out
