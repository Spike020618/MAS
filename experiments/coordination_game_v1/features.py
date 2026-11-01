import re

def extract(member,action,catalog,relation):
    words=lambda s:set(re.findall(r'\w+',s.lower()))
    target=words(relation);texts=' '.join(catalog[s]for s in action.source_ids);read=set(member.observed_sources)
    return {'already_read_fraction':len(read&set(action.source_ids))/max(1,len(action.source_ids)),
            'local_source_count':len(read),'local_report_count':len(member.local_reports),
            'relation_overlap':len(target&words(texts))/max(1,len(target)),'source_count':len(action.source_ids)}
