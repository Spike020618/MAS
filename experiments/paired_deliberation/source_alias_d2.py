"""Lossless local source-ID aliases; never searches history or changes quotations."""
import copy
import json

from experiments.consensus_recommendation.common import fingerprint


def canonicalize(record,request):
    if request['kind'] not in {'atomic_source_relations','utterance_witness_d2'}:return copy.deepcopy(record),[]
    payload=json.loads(request['messages'][1]['content']);sources=payload.get('sources',{})
    aliases={}
    for canonical,source in sources.items():
        for alias in (canonical,source.get('source_id')):
            if isinstance(alias,str):aliases.setdefault(alias,set()).add(canonical)
    result=copy.deepcopy(record);operations=[]
    field='relations' if request['kind']=='atomic_source_relations' else 'witnesses'
    rows=result['report'].get(field)
    if not isinstance(rows,list):return result,operations
    for row in rows:
        if not isinstance(row,dict):continue
        old=row.get('source_id')
        matches=aliases.get(old,set()) if isinstance(old,str) else set()
        if len(matches)==1:
            canonical=next(iter(matches))
            if canonical!=old:
                row['source_id']=canonical;operations.append({'old':old,'canonical':canonical,'operation':'unique alias within supplied current sources'})
    return result,operations


class SourceAliasClient:
    def __init__(self,client):self.client,self.source_aliases=client,[]
    def __getattr__(self,name):return getattr(self.client,name)
    @property
    def adaptations(self):return self.client.adaptations+self.source_aliases
    def complete(self,slot,request):
        before=self.client.complete(slot,request);after,ops=canonicalize(before,request)
        self.source_aliases.append({'slot':slot,'source_id_operations':ops,'before_report_hash':fingerprint(before['report']),
                                    'after_report_hash':fingerprint(after['report'])})
        return after
