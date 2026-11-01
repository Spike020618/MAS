"""Member-conditioned response channels and decision-focused value of information."""
import json,math
from pathlib import Path
import numpy as np
import joblib
from experiments import member_evidence_data as d
from experiments.calibrated_action_value import OutcomeModel,LABELS,tokens
from experiments.consensus_mechanism import core as c
from experiments.consensus_mechanism.data import read
from experiments.consensus_recommendation.common import create,fingerprint,sha

ANSWERS=('supported','refuted','not_enough_info')

def initial_features(task,member):
    card=task['source_catalog'][task['metadata']['seed_ids'][member]]
    q=tokens(task['metadata']['given_claim']);words=tokens(card['text'])
    return {'coverage':len(q&words)/max(1,len(q)),'chars':len(card['text'])/3000.,
            'retrieval_score':card['retrieval_score'],'negations':len(words&{'not','never','no','false'})/5}

def world_features(features,world):return {**features,'hypothesized_world:'+world:1.}

def distribution(model,features,world):
    p=model.probs([world_features(features,world)])[0]
    row=np.array([p.get(i,0.)for i in range(3)],dtype=float)
    # Fixed smoothing avoids false certainty from a missing response class.
    return .95*row/max(row.sum(),1e-12)+.05/3

def train(root):
    tasks=[t for t in read(root/'tasks.json')if t['metadata']['acquisition_partition']!='test']
    cases=[read(root/'train/cases'/f'case_{i}.json')for i in range(len(tasks))]
    if read(root/'train/generation_complete.json')!={'task_ids':[t['id']for t in tasks],'hashes':[fingerprint(x)for x in cases]}:raise ValueError('training seal mismatch')
    sequence=None
    if (root/'sequence/generation_complete.json').exists():
        sequence=[read(root/'sequence/cases'/f'case_{i}.json')for i in range(len(tasks))]
        if read(root/'sequence/generation_complete.json')!={'task_ids':[t['id']for t in tasks],'hashes':[fingerprint(x)for x in sequence]}:raise ValueError('conditional seal mismatch')
    labels=read(root/'train.labels.private.json');fit=[];cal=[];initial=[];initial_cal=[];observations=[]
    for case_index,(t,case)in enumerate(zip(tasks,cases)):
        truth=labels[t['id']]['answer'];partition=t['metadata']['acquisition_partition']
        for member,(report,event)in enumerate(zip(case['initial']['reports'],case['initial']['events'])):
            if event['valid']and report['decision']in ANSWERS:
                row={'features':initial_features(t,member),'response':ANSWERS.index(report['decision']),'world':truth,'task_id':t['id']}
                (initial if partition=='fit'else initial_cal).append(row)
        for arm in case['arms']+([sequence[case_index]['arm']]if sequence else []):
            row={'features':arm['features'],'valid':arm['event']['valid'],'task_id':t['id'],'world':truth,
                 'member':arm['reviewer'],'branch':arm['branch']}
            raw=arm['event'].get('raw_report')or{}
            row['response']=ANSWERS.index(raw['decision'])if row['valid']and raw.get('decision')in ANSWERS else None
            (fit if partition=='fit'else cal).append(row);observations.append(row)
    good=[r for r in fit if r['response']is not None]
    if len(good)<12 or len(initial)<8:raise ValueError('insufficient valid training channels')
    im=OutcomeModel().fit([world_features(r['features'],r['world'])for r in initial],[r['response']for r in initial])
    rm=OutcomeModel().fit([world_features(r['features'],r['world'])for r in good],[r['response']for r in good])
    vm=OutcomeModel().fit([r['features']for r in fit],[int(r['valid'])for r in fit])
    temperatures={}
    for name,model,rows in [('initial',im,initial_cal),('review',rm,[r for r in cal if r['response']is not None])]:
        scores={}
        for temp in (1.,2.,4.):
            model.temperature=temp
            losses=[-math.log(distribution(model,r['features'],r['world'])[r['response']])for r in rows]
            scores[str(temp)]=float(np.mean(losses))if losses else 0.
        model.temperature=float(min(scores,key=scores.get));temperatures[name]={'nll':scores,'selected':model.temperature}
    fit_truth=[labels[t['id']]['answer']for t in tasks if t['metadata']['acquisition_partition']=='fit']
    prior=np.array([fit_truth.count(a)+1 for a in ANSWERS],dtype=float);prior/=prior.sum()
    model={'initial':im,'review':rm,'validity':vm,'prior':prior,'training_claim_ids':[t['id']for t in tasks],
      'scope':'conditional response models; hypothesized truth enumerated at inference, never actual test truth',
      'cost_weight':.1,'temperatures':temperatures}
    model['posterior_temperature']=1.
    calibration_tasks=[(t,x)for t,x in zip(tasks,cases)if t['metadata']['acquisition_partition']=='calibration']
    posterior_losses={}
    for temperature in (1.,2.,4.):
        losses=[]
        for t,x in calibration_tasks:
            p=posterior(model,t,x['initial']);p=np.maximum(p,1e-12)**(1/temperature);p/=p.sum()
            losses.append(-math.log(p[ANSWERS.index(labels[t['id']]['answer'])]))
        posterior_losses[str(temperature)]=float(np.mean(losses))if losses else 0.
    model['posterior_temperature']=float(min(posterior_losses,key=posterior_losses.get))
    dest=root/'model/model.joblib';dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise ValueError('model already frozen')
    joblib.dump(model,dest);create(root/'model/observations.json',observations)
    create(root/'model/summary.json',{'fit_review_valid':len(good),'fit_invalid_excluded':len(fit)-len(good),
      'fit_initial_valid':len(initial),'fit_review_response_counts':{a:sum(r['response']==i for r in good)for i,a in enumerate(ANSWERS)},
      'fit_truth_counts':{a:fit_truth.count(a)for a in ANSWERS},'temperatures':temperatures,
      'posterior_calibration_nll':posterior_losses,'posterior_temperature':model['posterior_temperature'],
      'member_training_counts':{str(i):sum(r['member']==i for r in good)for i in range(4)},'test_labels_read':False})
    return model

def posterior(model,task,initial):
    p=model['prior'].copy()
    for i,(report,event)in enumerate(zip(initial['reports'],initial['events'])):
        if not event['valid']or report['decision']not in ANSWERS:continue
        z=ANSWERS.index(report['decision']);f=initial_features(task,i)
        p*=np.array([distribution(model['initial'],f,y)[z]for y in ANSWERS]);p/=max(p.sum(),1e-12)
    p=np.maximum(p,1e-12)**(1/model.get('posterior_temperature',1.));return p/p.sum()

def response_channel(model,task,state,reports,member,ids):
    f=d.features(task,state,reports,member,ids)
    valid=model['validity'].probs([f])[0].get(1,0.)
    q=np.stack([distribution(model['review'],f,y)for y in ANSWERS])
    # Failure is a separate observable and conveys no truth evidence.
    return np.concatenate([valid*q,np.full((3,1),1-valid)],axis=1)

def update(p,channel,event):
    if not event['valid']:return p.copy()
    answer=event['raw_report']['decision'];z=ANSWERS.index(answer)
    out=p*channel[:,z]
    return out/max(out.sum(),1e-12)
