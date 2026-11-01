"""D2 purpose-separated calibration; deterministic labels stored outside inference."""
import copy

from experiments.consensus_recommendation.common import ROOT,fingerprint
from experiments.paired_deliberation.evidence_d2 import EvidenceRegistry
from experiments.paired_deliberation.training_evidence_view import COLUMNS


def context(texts,tag,genre='Documentary'):
    e={'case_id':'d2_'+fingerprint(tag)[:16],
       'observations':[{'source_id':'message:'+str(i),'role':role,'text':text} for i,(role,text) in enumerate(texts)],
       'previous_mentions_excluded_not_asserted_seen':[],
       'candidates_in_shared_retrieval_order':[{'source_id':'movie:'+str(i),'movie_id':str(i),'title':'Calibration '+str(i),
           'year':2000,'genres':[genre if i%2 else 'Comedy']} for i in range(1,7)]}
    view={'kind':'historical_training_prediction_evidence','case_id':e['case_id'],'current_case_labels_available':False,
          'candidate_columns':list(COLUMNS),'candidate_rows':[[str(i)]+[0]*(len(COLUMNS)-1) for i in range(1,7)],
          'training_source_hash':fingerprint('synthetic training reference control'),
          'historical_examples':[{'training_case_id':'historical:control','training_prefix':'seeker: A previous request.',
                                  'historical_targets_in_pool':[]}]}
    return e,view


def make_claim(reg,kind,text,obj='Documentary',polarity=True,predicate='wants',hard=False):
    predictive=kind in {'hypothesis','recommendation_prediction'}
    ref='historical:control' if predictive else 'movie:1' if kind=='catalog_fact' else reg.evidence['observations'][0]['source_id']
    return {'claim_id':'c1','text':text,'claim_kind':kind,
        'evidence_refs':[{'source_id':reg.resolve(ref),'use':'prediction_basis' if predictive else 'supports'}],
        'candidate_ids':['1'],'depends_on':[],
        'scope':{'scope':'utterance' if kind=='utterance' else 'preference_state','subject':'seeker','predicate':predicate,
                 'object':obj,'polarity':polarity,'time':'current'} if kind in {'utterance','current_preference'} else None,
        'constraint_mode':'hard' if hard else 'soft','stance':'support'}


def build():
    inputs=[];labels=[]
    def add(e,view,claim,purpose,verdict,group='validation',critical=False,hard=False,rejected=False):
        body={'evidence':e,'training_view':view,'claim':claim}
        cid='case_'+fingerprint(body)[:16]
        if cid in {r['id'] for r in inputs}:raise ValueError('duplicate calibration input')
        inputs.append({'id':cid,**copy.deepcopy(body)})
        labels.append({'id':cid,'purpose':purpose,'expected_verdict':verdict,'expected_scope':copy.deepcopy(claim['scope']),
                       'group':group,'critical_conflict':critical,'expected_hard':hard,'expected_rejected':rejected})
    objects=('Documentary','Animation','Fantasy','Crime')
    for obj in objects:
        other='Comedy'
        for kind in ('utterance','current_preference'):
            for polarity in (True,False):
                matching='I '+('want ' if polarity else 'do not want ')+obj+' tonight.'
                opposite='I '+('do not want ' if polarity else 'want ')+obj+' tonight.'
                for mode in ('matching','opposite','conflict','irrelevant'):
                    if mode=='matching':texts=[('seeker',matching+' I do not want '+other+' tonight.')]
                    elif mode=='opposite':texts=[('seeker',opposite+' I want '+other+' tonight.')]
                    elif mode=='conflict':texts=[('seeker',matching+' '+opposite+' Both statements refer to the same moment; neither overrides the other.')]
                    else:texts=[('recommender',matching),('seeker','I want '+other+' tonight.')]
                    e,view=context(texts,[obj,kind,polarity,mode],obj);reg=EvidenceRegistry(e,view)
                    text=('The seeker said they '+('want ' if polarity else 'do not want ') if kind=='utterance' else
                          'The seeker '+('wants ' if polarity else 'does not want '))+obj+' tonight.'
                    c=make_claim(reg,kind,text,obj,polarity)
                    if kind=='utterance':verdict='supported' if mode in {'matching','conflict'} else 'insufficient'
                    else:verdict={'matching':'supported','opposite':'refuted','conflict':'insufficient','irrelevant':'insufficient'}[mode]
                    add(e,view,c,kind,verdict,critical=kind=='current_preference' and mode=='conflict')
        for correct in (True,False):
            e,view=context([('seeker','Please suggest a movie.')],[obj,'catalog',correct],obj);reg=EvidenceRegistry(e,view)
            c=make_claim(reg,'catalog_fact','Movie 1 has catalog year %d.'%(2000 if correct else 1999))
            add(e,view,c,'catalog_fact','supported' if correct else 'refuted')
        for kind in ('hypothesis','recommendation_prediction'):
            e,view=context([('seeker','Please suggest a movie.')],[obj,kind],obj);reg=EvidenceRegistry(e,view)
            c=make_claim(reg,kind,'Movie 1 may suit a similar historical request.')
            if kind=='recommendation_prediction':
                sid=next(s for s,row in reg.snapshot().items() if row['kind']=='train_stat')
                c['evidence_refs']=[{'source_id':sid,'use':'prediction_basis'}]
            add(e,view,c,'prediction_basis','not_evaluated')
        for attack in ('fact_promotion','hard_prediction'):
            e,view=context([('seeker','Please suggest a movie.')],[obj,attack],obj);reg=EvidenceRegistry(e,view)
            if attack=='fact_promotion':
                c=make_claim(reg,'current_preference','The seeker wants '+obj+' tonight.',obj)
                c['evidence_refs']=[{'source_id':reg.resolve('historical:control'),'use':'supports'}]
            else:c=make_claim(reg,'hypothesis','Historical data implies a mandatory exclusion.');c['constraint_mode']='hard'
            add(e,view,c,'authority','rejected',rejected=True)
    for mode in ('explicit','soft','opposition','other_speaker'):
        obj='Documentary';command='Do not recommend '+obj+'.'
        texts=([('seeker',command)] if mode=='explicit' else [('seeker','I do not want '+obj+' tonight.')] if mode=='soft' else
               [('seeker',command),('seeker','I want '+obj+' tonight. Both simultaneous statements have equal standing.')] if mode=='opposition' else
               [('recommender',command),('seeker','I have no specified preference.')])
        e,view=context(texts,['hard_guard',mode],obj);reg=EvidenceRegistry(e,view)
        c=make_claim(reg,'current_preference','The seeker does not want '+obj+' tonight.',obj,False,hard=True)
        add(e,view,c,'hard_guard','supported' if mode in {'explicit','soft'} else 'insufficient',hard=mode=='explicit')
    # Known D1 development cases: new purpose contract, NEVER rescoring old release.
    import json
    prior=ROOT/'experiments/results/paired_deliberation_commit_v4_joint_validation_r1'
    old_inputs={r['id']:r for r in json.loads((prior/'inputs.json').read_text())}
    old_labels={r['id']:r for r in json.loads((prior/'labels.json').read_text())}
    selected=['scope_2eef0ed08926','atomic_e1ad09e3a992828a']
    selected += sorted(r['id'] for r in old_labels.values() if r['group']=='new' and r['expected_atom']['scope']=='preference_state' and r['verdict']=='insufficient')[:2]
    for cid in selected:
        old=old_inputs[cid];label=old_labels[cid];e=copy.deepcopy(old['evidence']);reg=EvidenceRegistry(e)
        legacy=old['reports']['current_request']['claims'][0];scope=label['expected_atom']
        kind='utterance' if scope['scope']=='utterance' else 'current_preference'
        c=reg.migrate_legacy(legacy,explicit_kind=kind,scope=scope)
        add(e,None,c,kind,label['verdict'],group='regression')
    if len(inputs)!=96:raise ValueError('expected 96 frozen purpose cases')
    return sorted(inputs,key=lambda r:r['id']),sorted(labels,key=lambda r:r['id'])
