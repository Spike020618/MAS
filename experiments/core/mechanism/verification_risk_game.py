"""Finite zero-sum VERIFICATION-RISK game, not agreement optimization.

Calibration uses independently resolved claim labels. Inference samples a
frozen verifier protocol without labels. The attacker chooses a predeclared
message protocol, not the truth label or the verifier's private random draw.
Exact Fraction LP vertex enumeration is intended for small finite games only.
It solves the observed calibration matrix, not the unknown population game.
"""
from fractions import Fraction
from itertools import combinations
from math import comb, isfinite


TRUTH_KINDS={'official_answer','executable_check','independent_adjudication'}


def rational(value):
    if isinstance(value,bool):raise ValueError('boolean is not a payoff')
    if isinstance(value,Fraction):return value
    if isinstance(value,int):return Fraction(value)
    if isinstance(value,str):return Fraction(value)
    # Deliberately disallow floating-point "exact certificates".
    raise ValueError('exact rational payoff required: int, fraction or rational string')


def calibration_matrix(rows,verifiers,attacks):
    """Question-macro balanced classification error; labels ONLY offline.

    Every (question, claim) occurs under every verifier x attack. Message
    protocols must preserve claim identity and truth. Invalid verdicts count
    as error, never as a convenient rejection. Each question needs both truth
    classes, avoiding the always-reject and varying-option-count shortcuts.
    """
    verifiers,attacks=tuple(verifiers),tuple(attacks)
    if not rows or not verifiers or not attacks or len(set(verifiers))!=len(verifiers) or len(set(attacks))!=len(attacks):
        raise ValueError('nonempty rows and unique policy names required')
    index={};truth={};questions={}
    required={'question_id','claim_id','verifier','attack','verdict','truth','truth_kind','truth_ref'}
    for row in rows:
        if set(row)!=required:raise ValueError('closed calibration schema required')
        if (row['verifier'] not in verifiers or row['attack'] not in attacks or
            row['verdict'] not in ('accept','reject','invalid') or type(row['truth']) is not int or row['truth'] not in (0,1) or
            row['truth_kind'] not in TRUTH_KINDS or not isinstance(row['truth_ref'],str) or not row['truth_ref']):
            raise ValueError('invalid resolved verification trial')
        q,c=row['question_id'],row['claim_id']
        if any(not isinstance(x,str) or not x for x in (q,c)):raise ValueError('question/claim IDs required')
        key=(q,c,row['verifier'],row['attack'])
        if key in index:raise ValueError('duplicate trial')
        index[key]=row
        fact=(row['truth'],row['truth_kind'],row['truth_ref'])
        if (q,c) in truth and truth[q,c]!=fact:raise ValueError('claim truth/provenance changed across protocols')
        truth[q,c]=fact;questions.setdefault(q,set()).add(c)
    for q,claims in questions.items():
        if {truth[q,c][0] for c in claims}!={0,1}:raise ValueError('both true and false claims required per question')
        for c in claims:
            for v in verifiers:
                for a in attacks:
                    if (q,c,v,a) not in index:raise ValueError('incomplete paired protocol table')
    losses=[]
    for v in verifiers:
        line=[]
        for a in attacks:
            per_question=[]
            for q,claims in questions.items():
                class_error=[]
                for label in (0,1):
                    selected=[index[q,c,v,a] for c in claims if truth[q,c][0]==label]
                    errors=sum(r['verdict']=='invalid' or (r['verdict']=='accept')!=bool(label) for r in selected)
                    class_error.append(Fraction(errors,len(selected)))
                per_question.append(sum(class_error)/2)
            line.append(sum(per_question)/len(per_question))
        losses.append(tuple(line))
    return {'verifiers':verifiers,'attacks':attacks,'losses':tuple(losses),
            'questions':len(questions),'claims':len(truth),'trials':len(rows),
            'scope':'empirical question-macro balanced candidate-verification error; not rationale validity or native QA accuracy'}


def _linear_solve(matrix,rhs):
    n=len(rhs);a=[list(row)+[rhs[i]] for i,row in enumerate(matrix)]
    for col in range(n):
        pivot=next((i for i in range(col,n) if a[i][col]),None)
        if pivot is None:return None
        a[col],a[pivot]=a[pivot],a[col]
        p=a[col][col];a[col]=[v/p for v in a[col]]
        for i in range(n):
            if i!=col:
                multiple=a[i][col]
                a[i]=[x-multiple*y for x,y in zip(a[i],a[col])]
    return tuple(row[-1] for row in a)


def _vertex_optimize(inequalities,nprob,minimize,max_vertices):
    if comb(len(inequalities),nprob)>max_vertices:raise ValueError('finite enumeration budget exceeded')
    zero=Fraction(0);one=Fraction(1);best=None
    total=0
    for active in combinations(inequalities,nprob):
        matrix=[(one,)*nprob+(zero,)]+list(active)
        solution=_linear_solve(matrix,[one]+[zero]*nprob)
        total+=1
        if solution is None or any(sum(a*b for a,b in zip(row,solution))>0 for row in inequalities):continue
        score=solution[-1] if minimize else -solution[-1]
        candidate=(score,solution)
        if best is None or candidate<best:best=candidate
    if best is None:raise ArithmeticError('no feasible vertex')
    return best[1],total


def solve_risk_game(losses,max_vertices=50000):
    matrix=tuple(tuple(rational(x) for x in row) for row in losses)
    if not matrix or not matrix[0] or any(len(row)!=len(matrix[0]) for row in matrix):
        raise ValueError('nonempty rectangular risk matrix required')
    if any(x<0 or x>1 for row in matrix for x in row):raise ValueError('verification losses must lie in [0,1]')
    m,n=len(matrix),len(matrix[0]);z=Fraction(0);one=Fraction(1)
    # Verifier minimizes t, subject to pi^T L[:,j] <= t.
    primal=[tuple(matrix[i][j] for i in range(m))+(-one,) for j in range(n)]
    primal += [tuple(-one if i==k else z for i in range(m))+(z,) for k in range(m)]
    # Adversary maximizes s, subject to L[i,:] q >= s.
    dual=[tuple(-matrix[i][j] for j in range(n))+(one,) for i in range(m)]
    dual += [tuple(-one if j==k else z for j in range(n))+(z,) for k in range(n)]
    p,np=_vertex_optimize(primal,m,True,max_vertices)
    q,nq=_vertex_optimize(dual,n,False,max_vertices)
    pi,attack=p[:-1],q[:-1]
    upper=max(sum(pi[i]*matrix[i][j] for i in range(m)) for j in range(n))
    lower=min(sum(matrix[i][j]*attack[j] for j in range(n)) for i in range(m))
    if upper!=lower:raise ArithmeticError('exact primal/dual game gap did not close')
    return {'verifier_policy':pi,'attacker_policy':attack,'value':upper,'duality_gap':upper-lower,
            'verifier_best_response_gap':upper-lower,'attacker_best_response_gap':upper-lower,
            'best_pure_worst_loss':min(max(row) for row in matrix),
            'uniform_worst_loss':max(sum(matrix[i][j] for i in range(m))/m for j in range(n)),
            'vertices_examined':np+nq,
            'scope':'exact zero-sum equilibrium of supplied empirical finite matrix only; not population truth/effectiveness guarantee'}


def select_verifier(names,policy,uniform_draw):
    """Deployment: sample ONE verifier, not weighted voting. No labels accepted."""
    names,policy=tuple(names),tuple(rational(p) for p in policy)
    if not names or len(names)!=len(policy) or len(set(names))!=len(names) or any(p<0 for p in policy) or sum(policy)!=1:
        raise ValueError('invalid frozen policy')
    if type(uniform_draw) not in (int,float) or not isfinite(uniform_draw) or not 0<=uniform_draw<1:
        raise ValueError('independent draw in [0,1) required')
    draw=Fraction(str(uniform_draw));cumulative=Fraction(0)
    for name,p in zip(names,policy):
        cumulative+=p
        if draw<cumulative:return name
    raise ArithmeticError('unreachable sampling boundary')


def portable(value):
    if isinstance(value,Fraction):return str(value)
    if isinstance(value,dict):return {k:portable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [portable(v) for v in value]
    return value
