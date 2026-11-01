"""E1 multi-source proposal heuristic. Not a trained information-value model."""
import re
from experiments.paired_deliberation.native_quality_review_game import QualityReviewGame


def proposals(task, state, reports, limit=8):
    words=set(re.findall(r'[a-z]{3,}',task['prompt'].lower()))
    explicit={str(s)for report in reports for s in report.get('proposed_source_ids',[])if isinstance(s,str)}
    rows=[]
    for sid,source in state['catalog'].items():
        if sid in state['visible_sources']:continue
        metadata=' '.join([source.get('title',''),source.get('kind',''),source.get('source_id',sid)]);overlap=len(words&set(re.findall(r'[a-z]{3,}',metadata.lower())))/max(1,len(words))
        kind=source.get('kind',source.get('source_kind','unknown'))
        score=.1+overlap+.25*(sid in explicit)+.1*(kind in ('knowledge_graph','historical_prediction_basis','external_evidence','model_forecast'))
        rows.append({'source_id':sid,'score':score,'kind':kind,'source_characters':len(source.get('text','')),'doubt':'Check whether this material supplies a relevant premise, counterexample, or omitted candidate.'})
    return sorted(rows,key=lambda r:(-r['score'],r['source_id']))[:limit]


def allocate(rows, method):
    if not rows:return {'actions':[],'scope':'no unseen permitted sources'}
    if method=='fixed':return {'actions':[(i,i)for i in range(min(3,len(rows)))],'scope':'deterministic source-order rotation'}
    gains=[[r['score']for r in rows]for i in range(3)]
    solved=QualityReviewGame(gains,[[.003+4*r['source_characters']*.15/1e6 for r in rows]for _ in range(3)],cost_weight=1.).solve()
    state=solved['central_actions']if method=='central'else solved['actions']
    return {'actions':[(i,q)for i,q in enumerate(state)if q is not None],'game':solved,
        'scope':'E1 uncalibrated metadata relevance/source coverage; homogeneous reviewers have identical public gains, no invented capability. Forecast review CNY .003+4*source_characters*.15/M, actual usage ledger separate'}
