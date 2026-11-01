"""One-parameter signed-risk calibration, fitted to expected utility on calibration claims."""
import numpy as np
class SignedRiskOutcome:
    def __init__(self,base,bias):self.base=base;self.bias=float(bias)
    def probs(self,X,prior=False):
        rows=self.base.probs(X,prior=prior);out=[]
        for row in rows:
            weighted={k:float(row.get(k,0.))*np.exp(self.bias*k)for k in (-1,0,1)}
            total=sum(weighted.values())
            out.append({k:float(value/total)for k,value in weighted.items()})
        return out

def calibrate(bundle,observations):
    from scipy.optimize import minimize_scalar
    rows=[r for r in observations if r['partition']=='calibration'and r['valid']]
    if not rows:raise ValueError('calibration claims required')
    X=[r['features']for r in rows];y=np.array([r['reward']for r in rows]);counts={}
    for r in rows:counts[r['task_id']]=counts.get(r['task_id'],0)+1
    weights=np.array([1/counts[r['task_id']]for r in rows]);weights/=weights.sum()
    valid=np.array([p.get(1,0.)for p in bundle['validity'].probs(X)])
    def loss(b):
        model=SignedRiskOutcome(bundle['outcome'],b);p=model.probs(X)
        gains=valid*np.array([r.get(1,0.)-r.get(-1,0.)for r in p])
        return float(np.sum(weights*(gains-y)**2)+.05*b*b)
    fit=minimize_scalar(loss,bounds=(-2.,2.),method='bounded')
    bias=float(fit.x)
    return {**bundle,'outcome':SignedRiskOutcome(bundle['outcome'],bias)}, {
        'signed_log_odds_bias':bias,'loss_before':loss(0.),'loss_after':loss(bias),
        'calibration_claims':sorted(counts),'regularization':.05,'bounds':[-2,2],
        'objective':'claim-balanced MSE of expected net decision gain plus L2; no test labels',
        'scope':'one-parameter calibration; base outcome model unchanged'}
