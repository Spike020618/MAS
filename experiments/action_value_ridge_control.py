"""Same-data/same-feature Ridge control; isolates architecture from domain adaptation."""
import numpy as np
class RidgeOutcome:
    def __init__(self,pipeline,bias):self.pipeline=pipeline;self.bias=bias
    def probs(self,X,prior=False):
        gains=np.clip(self.pipeline.predict(X)+self.bias,-1,1)
        return [{-1:max(-float(g),0.),0:1-abs(float(g)),1:max(float(g),0.)}for g in gains]

def fit_control(observations):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import Ridge
    fit=[r for r in observations if r['partition']=='fit'and r['valid']]
    cal=[r for r in observations if r['partition']=='calibration'and r['valid']]
    model=make_pipeline(DictVectorizer(),Ridge(alpha=10.,solver='lsqr')).fit([r['features']for r in fit],[r['reward']for r in fit])
    bias=float(np.mean([r['reward']for r in cal])-np.mean(model.predict([r['features']for r in cal])))if cal else 0.
    return RidgeOutcome(model,bias)
