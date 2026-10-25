"""Deterministic checks over observed source text, without benchmark labels.

These certify numerical consistency with a source, not source truth or recency.
"""
import re


def parse_ranked_kg_rows(excerpt):
    rows={}
    for line in excerpt.splitlines():
        match=re.search(r'(?:^|\|)\s*(\d+)\s+([A-Za-z][A-Za-z .\'-]*?)\s+(\d+(?:\.\d+)?)\s*kg\b',line)
        if not match:continue
        country=' '.join(match.group(2).split())
        value=float(match.group(3))
        if country in rows and rows[country]['value']!=value:
            raise ValueError('conflicting source rows for same entity')
        rows[country]={'value':value,'unit':'kg','quote':line,'rank':int(match.group(1))}
    return rows


def compare_entities(rows,reference,entities):
    required=[reference]+list(entities)
    missing=[entity for entity in required if entity not in rows]
    if missing:return {'status':'insufficient','missing':missing,'comparisons':[]}
    units={rows[entity]['unit'] for entity in required}
    if len(units)!=1:return {'status':'insufficient','reason':'unit mismatch','comparisons':[]}
    comparisons=[{'entity':entity,'value':rows[entity]['value'],
                  'reference':reference,'reference_value':rows[reference]['value'],
                  'greater':rows[entity]['value']>rows[reference]['value'],
                  'entity_quote':rows[entity]['quote'],'reference_quote':rows[reference]['quote']}
                 for entity in entities]
    return {'status':'checked_against_source','all_greater':all(x['greater'] for x in comparisons),
            'comparisons':comparisons,'scope':'source-relative numerical check; not independent factual truth'}


def check_tea_comparison(question,sources):
    """Narrow typed checker, selected from question semantics, never question ID.

    Covers a list of countries claimed to consume more tea per capita than the
    US. Refuses missing operands, ambiguous total/per-person measurement, dates,
    and conflicting rows. This is a task-specific checker, not general fact QA.
    """
    normalized=question.lower()
    if not ('tea' in normalized and 'more' in normalized and 'on average' in normalized
            and ('american' in normalized or 'united states' in normalized)):
        return {'status':'not_applicable'}
    if re.search(r'\b(?:19|20)\d{2}\b|\b(today|currently|latest)\b',normalized):
        return {'status':'insufficient','reason':'explicit time requirement not checked'}
    relevant=[s for s in sources if 'tea consumption per capita' in s.get('title','').lower()]
    if not relevant:return {'status':'insufficient','reason':'no matching measurement source'}
    try:
        rows=parse_ranked_kg_rows('\n'.join(s['excerpt'] for s in relevant))
    except ValueError as exc:return {'status':'insufficient','reason':str(exc)}
    normalized=re.sub(r'\buk\b','united kingdom',normalized)
    normalized=normalized.replace('americans','united states').replace('american','united states')
    entities=[name for name in rows if name!='United States' and
              re.search(r'\b'+re.escape(name.lower())+r'\b',normalized)]
    if not entities:return {'status':'insufficient','reason':'no comparison operands resolved'}
    # Ensure every explicit list item in the proposed answer is covered rather
    # than checking only the subset of countries returned by retrieval.
    proposed=re.search(r'proposed answer:\s*(.*?)\s+(?:consume|consumes)\b',question,re.I)
    if not proposed:return {'status':'insufficient','reason':'explicit country list not resolved'}
    names=[x.strip().removeprefix('the ').strip() for x in re.split(r',|\band\b',proposed.group(1),flags=re.I)
           if x.strip()]
    aliases={'uk':'United Kingdom','united kingdom':'United Kingdom'}
    lookup={name.lower():name for name in rows}
    resolved=[]
    for name in names:
        entity=aliases.get(name.lower(),lookup.get(name.lower()))
        if not entity or entity=='United States':
            return {'status':'insufficient','reason':'unresolved proposed comparison operand','operand':name}
        resolved.append(entity)
    result=compare_entities(rows,'United States',resolved)
    if result['status']=='checked_against_source':
        result.update(stance='yes' if result['all_greater'] else 'no',
                      source_urls=[s['url'] for s in relevant],
                      checker='tea_per_capita_country_comparison_v1',
                      temporal_limit='source-relative historical table; not a current-data certification')
    return result
