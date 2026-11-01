from dataclasses import asdict
from .schema import digest

def snapshot(members,public_sources,relations):
    return {'members':[asdict(m)for m in members],'public_sources':sorted(public_sources),'relations':list(relations)}

def state_hash(members,public_sources,relations):return digest(snapshot(members,public_sources,relations))

def visible(member,public_sources,action,source_catalog):
    if action.member_id!=member.member_id:raise ValueError('action/member mismatch')
    ids=set(member.observed_sources)|set(public_sources)|set(action.source_ids)
    if not ids<=set(source_catalog):raise ValueError('unknown source')
    return {s:source_catalog[s]for s in sorted(ids)}
