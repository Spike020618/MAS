"""Explicit Hotpot public projection. Labels stay in the caller's private scoring vault."""
import re,unicodedata
from .schema import digest

def question_key(text):return re.sub(r'\W+','',unicodedata.normalize('NFKC',text).casefold())

def project(row):
    if not isinstance(row.get('question'),str)or not isinstance(row.get('context'),list):raise ValueError('Hotpot question/context required')
    sources=[]
    for i,(title,sentences)in enumerate(row['context']):
        if not isinstance(title,str)or not isinstance(sentences,list)or any(not isinstance(s,str)for s in sentences):raise ValueError('invalid context')
        sources.append({'id':f'doc:{i}','title':title,'sentences':list(sentences)})
    return {'id':str(row['_id']),'question':row['question'],'sources':sources,'question_key':question_key(row['question'])}

def partition(rows,exposed_ids=(),exposed_questions=()):
    exposed_ids=set(exposed_ids);exposed_questions=set(exposed_questions);seen_ids=set();seen_q=set();eligible=[];excluded=[]
    for row in rows:
        p=project(row);reason=None
        if p['id']in exposed_ids or p['question_key']in exposed_questions:reason='previously_exposed'
        elif p['id']in seen_ids or p['question_key']in seen_q:reason='duplicate'
        seen_ids.add(p['id']);seen_q.add(p['question_key'])
        if reason:excluded.append({'id':p['id'],'reason':reason})
        else:eligible.append(p)
    return {'eligible':sorted(eligible,key=lambda p:digest(p['id'])),'excluded':excluded,'near_duplicate_review_required':True}
