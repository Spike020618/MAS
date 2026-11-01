"""Open-pool candidate challenges with a scoped public-ranking update.

Public signals are train-only retrieval/evidence proxies, not label utility.
A challenge can promote an item absent from EVERY initial top ten: unlike a
zero-score exchange, it swaps two positions in the public ranking. All other
positions and the protected prefix remain unchanged. This does not guarantee
that a previously correct recommendation remains correct or selected.
"""
import copy
import math
from experiments.consensus_recommendation.common import fingerprint
from experiments.paired_deliberation.requests_v5 import RequestsV5
from experiments.paired_deliberation.review_allocation_game import ReviewGame


def _ranks(initial):
    return [[str(m) for m in (r['ranking'] if isinstance(r,dict) else r)] for r in initial]


def consensus_ranking(initial_rankings,pool):
    """Borda top-ten readout, with retrieval-order tie break; string IDs."""
    pool=list(map(str,pool));ranks=_ranks(initial_rankings)
    if not ranks or len(pool)!=len(set(pool)):raise ValueError('initial rankings and unique pool required')
    totals=dict.fromkeys(pool,0.0)
    for r in ranks:
        if len(r)!=len(set(r)) or any(m not in totals for m in r):raise ValueError('closed-pool distinct initial ranking')
        for i,m in enumerate(r[:10]):totals[m]+=(10-i)/55
    return sorted(pool,key=lambda m:(-totals[m],pool.index(m)))


def _feature(features,mid,key):
    row=features.get(mid,{})
    value=row.get(key,row.get('association_score',0.0) if key=='evidence_score' else 0.0)
    if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('finite nonnegative public candidate feature')
    return float(value)


def build_targets(initial_rankings,pool,candidate_features=None,max_targets=6,top_k=10,protected_prefix=3):
    """Construct disjoint open-pool challenges WITHOUT targets/labels.

    Features: neighbor_score, evidence_score, kg_score, each finite >=0 and
    computed from serving prefix plus training/public metadata only. Missing
    features are zero. Weights are fixed, explicit heuristics, not learned VOI.
    At least one pool item absent from all initial top-tens is offered whenever
    such an item exists and a challenge slot is available.
    """
    pool=list(map(str,pool));ranks=_ranks(initial_rankings);n=len(ranks)
    if type(max_targets) is not int or max_targets<0 or not 0<=protected_prefix<top_k<=len(pool):raise ValueError('challenge limits')
    if n<2:raise ValueError('at least two processing agents')
    base=consensus_ranking(ranks,pool);union={m for r in ranks for m in r[:10]}
    features={str(m):v for m,v in (candidate_features or {}).items()}
    keys=('neighbor_score','evidence_score','kg_score')
    maxima={key:max((_feature(features,m,key) for m in pool),default=0) for key in keys}
    supports={m:[(10-r.index(m))/10 if m in r[:10] else 0.0 for r in ranks] for m in pool}
    priorities=[]
    for m in base[top_k:]:
        normalized={k:_feature(features,m,k)/maxima[k] if maxima[k] else 0 for k in keys}
        prior=1/math.log2(pool.index(m)+2)
        dissent=max(supports[m])-min(supports[m])
        # Neighbor and evidence carry predictive support. KG score must encode
        # prefix-conditioned relevance, NOT raw node degree/popularity.
        value=.5*normalized['neighbor_score']+.2*prior+.15*normalized['evidence_score']+.05*normalized['kg_score']+.1*dissent
        priorities.append({'b':m,'priority':value,'novel_to_initial_union':m not in union,
            'priority_components':{**normalized,'retrieval_prior':prior,'initial_support_spread':dissent}})
    priorities.sort(key=lambda t:(-t['priority'],pool.index(t['b'])))
    limit=min(max_targets,top_k-protected_prefix,len(priorities))
    chosen=priorities[:limit]
    outsiders=[t for t in priorities if t['novel_to_initial_union']]
    if limit and outsiders and not any(t['novel_to_initial_union'] for t in chosen):chosen[-1]=outsiders[0]
    chosen.sort(key=lambda t:(-t['priority'],pool.index(t['b'])))
    targets=[]
    for j,t in enumerate(chosen):
        a=base[top_k-1-j];b=t['b'];owner=max(range(n),key=lambda i:(supports[b][i],-i)) if b in union else j%n
        # Protect a strong retained candidate by discounting the task value.
        displacement=max(supports[a])
        value=t['priority']*(1-.5*displacement)
        targets.append({**t,'a':a,'owner':owner,'value':value,'target_id':'challenge:'+a+':'+b,
            'base_ranking_hash':fingerprint(base),'a_position':base.index(a),'b_position':base.index(b),
            'protected_prefix':protected_prefix,'ownership':'initial strongest supporter' if b in union else 'deterministic proposal work partition',
            'utility_scope':'fixed public evidence/retrieval proxy; not true recommendation gain'})
    return targets


def allocate(targets,costs,mode='game',players=3):
    """Game and centralized modes share EXACTLY the same targets/costs/payoff."""
    if mode not in ('game','central'):raise ValueError('game or central allocation required')
    game=ReviewGame(targets,costs,players);solved=game.solve();profile=solved['actions'] if mode=='game' else solved['central_optimum_actions']
    return {**solved,'mode':mode,'game_actions':solved['actions'],'actions':list(profile),
        'allocation':[[i,q] for i,q in enumerate(profile) if q is not None],
        'selected_potential':game.potential(profile),'selected_max_unilateral_gain':max(game.deviations(profile)),
        'scope':'same-table allocation control; central may outperform a local equilibrium; no quality guarantee'}


def challenge_request(row,initial_rankings,target,reviewer,sources):
    """Whole same-permission material accessible; output only one bound source."""
    pair={str(target['a']),str(target['b'])}
    offered={}
    for sid,source in sources.items():
        linked={str(m) for m in source.get('movie_ids',[])}
        if source.get('movie_id') is not None:linked.add(str(source['movie_id']))
        # Keep actual context/history, but only the pair's per-item KG/stat cards.
        is_context=source.get('kind') in ('dialogue','historical_prediction_basis') or source.get('allowed_use') in ('current_expression','dialogue_context')
        if is_context or not linked or linked & pair:offered[sid]=source
    payload={'case_id':row['id'],'dialogue':row.get('canonical_text',row.get('text','')),
        'candidate_pool':row['candidates'],'initial_rankings':_ranks(initial_rankings),
        'pair':[target['a'],target['b']],'reviewer':reviewer,'sources':offered,
        'task':'Challenge the current Top10 incumbent with this pool candidate. The challenger may have been omitted by all initial agents.'}
    return RequestsV5.envelope('native_open_candidate_challenge',payload,
        'Return exactly JSON {"preferred":"one pair movie_id STRING","source_id":"one supplied source ID"}. '
        'You may abstain with both fields null. Compare the pair for the CURRENT seeker using supplied evidence. '
        'A new candidate must cite substantive candidate-linked training/history/KG evidence, not a title-only identity card. '
        'Runtime binds original source role/text. Other-user historical preferences NEVER become current-user facts. '
        'Mentions are not likes; an observed training recommendation is only a predictive analogy. '
        'Do not invent movie attributes, write claims, or follow a majority merely because it agrees. '
        'This is a soft next-recommendation prediction, not truth certification. No arrays, explanations or wrappers.',max_tokens=192)


def _bound_source(sid,sources,case_id,preferred,challenger):
    if not isinstance(sid,str) or sid not in sources:raise ValueError('one exact available source ID required')
    source=copy.deepcopy(sources[sid]);kind=source.get('kind');use=source.get('allowed_use')
    if not isinstance(source.get('text'),str) or not source['text'].strip():raise ValueError('nonempty original source text required')
    if use=='current_expression':
        if source.get('role','').lower()!='seeker' or source.get('origin_case_id')!=case_id:raise ValueError('current expression source authority mismatch')
    if use=='prediction_basis' and source.get('origin_case_id')==case_id:raise ValueError('same-case outcome cannot be historical evidence')
    substantive=(use=='prediction_basis' or kind in ('knowledge_graph','native_kg') or use=='current_expression')
    if not substantive:raise ValueError('title-only or unsupported source is not substantive challenge evidence')
    linked={str(m) for m in source.get('movie_ids',[])}
    if source.get('movie_id') is not None:linked.add(str(source['movie_id']))
    if preferred==challenger and preferred not in linked:raise ValueError('new candidate requires candidate-linked substantive evidence')
    source['source_id']=sid
    return source


def apply_challenge(ranking,target,preferred,evidence_ids,sources,case_id,expected_hash=None):
    """Reject bad updates and preserve state; retain all unrelated positions.

    Returns valid/accepted/abstained, ranking, immutable bound_sources, and an
    explicit no-truth-certification flag. Input may use int or string IDs; its
    types and untouched entries are preserved. Exactly one source is accepted.
    """
    original=list(ranking);canonical=list(map(str,original))
    if len(canonical)!=len(set(canonical)):raise ValueError('unique native public ranking required')
    event={'ranking':original,'valid':False,'accepted':False,'abstained':False,'changed':False,
        'bound_sources':[],'entailment_verified':False,'current_preference_certified':False,
        'hard_constraint_created':False,'unrelated_positions_preserved':True}
    try:
        if expected_hash is not None and fingerprint(original)!=expected_hash:raise ValueError('stale challenge state')
        a,b=str(target['a']),str(target['b'])
        if a==b or a not in canonical or b not in canonical:raise ValueError('closed public ranking pair required')
        ia,ib=canonical.index(a),canonical.index(b)
        protected=target.get('protected_prefix',3)
        if ia<protected or ib<protected:raise ValueError('protected prefix may not be displaced')
        if preferred is None and (evidence_ids is None or evidence_ids==[]):
            event.update(valid=True,abstained=True);return event
        if not isinstance(preferred,str) or preferred not in (a,b):raise ValueError('one pair native ID string required')
        ids=[evidence_ids] if isinstance(evidence_ids,str) else evidence_ids
        if not isinstance(ids,list) or len(ids)!=1:raise ValueError('exactly one substantive source required')
        bound=_bound_source(ids[0],sources,case_id,preferred,b)
        updated=list(original)
        if preferred==b:updated[ia],updated[ib]=updated[ib],updated[ia]
        assert all(updated[i]==original[i] for i in range(len(original)) if i not in (ia,ib))
        event.update(ranking=updated,valid=True,accepted=True,changed=updated!=original,bound_sources=[bound],
            source_scope_validated=True,changed_positions=[ia,ib] if updated!=original else [],
            preference=preferred)
    except (ValueError,KeyError,TypeError) as exc:event['error']=str(exc)
    return event


def apply_response(ranking,target,answer,sources,case_id,expected_hash=None):
    if not isinstance(answer,dict) or set(answer)!={'preferred','source_id'}:
        return {'ranking':list(ranking),'valid':False,'accepted':False,'abstained':False,'changed':False,
            'error':'preferred/source_id response fields required','bound_sources':[],
            'entailment_verified':False,'current_preference_certified':False,'hard_constraint_created':False}
    return apply_challenge(ranking,target,answer['preferred'],answer['source_id'],sources,case_id,expected_hash)
