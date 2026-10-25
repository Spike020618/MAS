"""Label-free local document access with exact character provenance.

Lexical retrieval is not an entailment oracle. Verified quotation only means
the literal text occurs in the provided source window, never factual support.
"""
from collections import Counter
from math import log
import re


def terms(text):
    return re.findall(r'[a-z0-9]+',text.lower())


def chunks(article,size=220,stride=110):
    if type(size) is not int or type(stride) is not int or not 0<stride<=size:
        raise ValueError('invalid chunk parameters')
    words=list(re.finditer(r'\S+',article));result=[]
    for start in range(0,len(words),stride):
        end=min(start+size,len(words));a=words[start].start();b=words[end-1].end()
        result.append({'id':'p'+str(start),'start':a,'end':b,'text':article[a:b]})
        if end==len(words):break
    return result


def retrieve(article,query,limit=6):
    if type(limit) is not int or limit<1:raise ValueError('positive window limit required')
    blocks=chunks(article);query=Counter(terms(query));freq=Counter()
    for b in blocks:freq.update(set(terms(b['text'])))
    def score(b):
        counts=Counter(terms(b['text']))
        return sum(min(counts[t],3)*log(1+len(blocks)/(1+freq[t])) for t in query)
    ranked=sorted(blocks,key=lambda b:(-score(b),b['start']));selected=[]
    for b in ranked:
        if any(max(0,min(b['end'],x['end'])-max(b['start'],x['start']))/
               min(b['end']-b['start'],x['end']-x['start'])>.5 for x in selected):continue
        selected.append(b)
        if len(selected)==limit:break
    return selected


def citations_valid(citations,windows):
    byid={b['id']:b for b in windows};result=[]
    for citation in citations:
        valid=(isinstance(citation,dict) and isinstance(citation.get('quote'),str)
               and bool(citation['quote'].strip()) and isinstance(citation.get('chunk_id'),str)
               and citation.get('chunk_id') in byid
               and citation['quote'] in byid[citation['chunk_id']]['text'])
        result.append({'citation':citation,'literal_source_match':valid,
                       'entailment_verified':False})
    return result
