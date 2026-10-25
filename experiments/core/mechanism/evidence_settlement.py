"""Budget-capped settlement for traceable, authorized source contributions.

Semantic support is an external assessment, not inferred from literal quotation.
Sharing a source pool makes the payoff depend on other contributors' actions.
"""
import math
from collections import defaultdict


def settle_evidence(reports, budget):
    if not math.isfinite(budget) or budget < 0:
        raise ValueError('finite nonnegative budget required')
    agents=[r['agent_id'] for r in reports]
    if len(set(agents))!=len(agents): raise ValueError('duplicate agent')
    pools=defaultdict(set)
    costs={}
    for report in reports:
        agent=report['agent_id']
        cost=report['cost']
        if not isinstance(cost,(int,float)) or not math.isfinite(cost) or cost<0:
            raise ValueError('finite nonnegative cost required')
        costs[agent]=cost
        allowed=set(report['authorized_source_hashes'])
        for citation in report['citations']:
            source=citation['source_hash']
            if (source in allowed and citation.get('exact_quote_valid') is True
                    and citation.get('support_status')=='supported'):
                pools[source].add(agent)
    credits={a:0.0 for a in agents}
    for contributors in pools.values():
        share=budget/len(pools)/len(contributors)
        for agent in contributors: credits[agent]+=share
    return {'assessment_scope':'traceable contribution with externally assessed support, not truth guarantee',
            'budget':budget,'paid':sum(credits.values()),
            'source_pools':{s:sorted(v) for s,v in pools.items()},
            'agents':{a:{'credit':credits[a],'cost':costs[a],'utility':credits[a]-costs[a]} for a in agents}}
