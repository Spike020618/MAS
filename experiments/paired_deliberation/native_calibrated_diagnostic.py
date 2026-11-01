"""Held-conversation audit of calibrated count ranker and proposal envelopes.

No public validation/test or LM API. Proposal oracle and blind-swap outcomes are
numerical diagnostics, never claims about actual review accuracy or game gains.
"""
import copy,json,math
from collections import Counter,defaultdict
from pathlib import Path
import joblib
import numpy as np
from experiments.consensus_recommendation.common import ROOT,create,sha,fingerprint
from experiments.paired_deliberation.native_calibrated_challenge import RESULT,DATA,load_fold,split_indices,conv,candidate_features,calibrated_targets,allocate_calibrated
from experiments.paired_deliberation.native_candidate_challenge import consensus_ranking,build_targets,allocate
from experiments.paired_deliberation.expected_review_value import expected_review_value

OUT=ROOT/'experiments/results/native_calibrated_diagnostic_v1'


def micro_sums(ranking,targets):
    positions={m:i for i,m in enumerate(ranking[:10])};hits=ndcg=0
    for m in targets:
        if m in positions:hits+=1;ndcg+=1/math.log2(positions[m]+2)
    return hits,ndcg,len(targets)


def paired_cluster_ci(records,left,right,replicates=10000):
    grouped=defaultdict(lambda:np.zeros(5))
    for row in records:
        group=grouped[conv(row['id'])];a=row[left];b=row[right]
        group+=np.array([a[0],b[0],a[1],b[1],a[2]])
    values=np.array(list(grouped.values()));point=values.sum(axis=0);rng=np.random.default_rng(271028);draws=[]
    for start in range(0,replicates,500):
        idx=rng.integers(0,len(values),size=(min(500,replicates-start),len(values)));totals=values[idx].sum(axis=1)
        draws.append(np.column_stack(((totals[:,0]-totals[:,1])/totals[:,4],(totals[:,2]-totals[:,3])/totals[:,4])))
    draw=np.concatenate(draws);ci=np.quantile(draw,[.025,.975],axis=0)
    return {'left':left,'right':right,'conversations':len(values),'item_targets':int(point[4]),'left_hits@10':int(point[0]),'right_hits@10':int(point[1]),
        'recall@10_delta':float((point[0]-point[1])/point[4]),'recall@10_cluster95':ci[:,0].tolist(),
        'ndcg@10_delta':float((point[2]-point[3])/point[4]),'ndcg@10_cluster95':ci[:,1].tolist(),
        'replicates':replicates,'seed':271028,'scope':'calibrator diagnostic, not global pristine confirmation; no multiplicity adjustment'}


def net_voi_targets(ranking,counts,members,initial,accuracy,max_targets=6):
    if accuracy<=.5:return []
    rank=list(map(str,ranking));options=[]
    for a in rank[3:10]:
        for b in rank[10:]:
            ds=np.array([m[b]-m[a] for m in members]);mu=counts[b]-counts[a];sd=float(ds.std())
            assumption=expected_review_value(mu,sd,accuracy,baseline='best_mean')
            if assumption['net_value']<=1e-12:continue
            options.append({'a':a,'b':b,'value':assumption['net_value'],'priority':assumption['net_value'],
                'owner':max(range(3),key=lambda i:(10-initial[i].index(a) if a in initial[i][:10] else 0,-i)),
                'ideal_swap_gain':mu,'gain_count':counts[b],'harm_count':counts[a],'assumption':assumption,'protected_prefix':3})
    options.sort(key=lambda t:(-t['value'],t['harm_count'],rank.index(t['b'])));out=[];used=set()
    for target in options:
        if target['a'] in used or target['b'] in used:continue
        out.append(target);used.update((target['a'],target['b']))
        if len(out)>=max_targets:break
    return out


def proposal_record(baseline,targets,truth,cost=.0015):
    before=sum(truth[m] for m in baseline[:10]);out={}
    for mode in ('central','game'):
        decision=allocate_calibrated(targets,[[cost]*len(targets) for _ in range(3)],mode,expected_hits_per_cny=1.)
        chosen=[targets[q] for i,q in decision['allocation']];blind_gain=sum(truth[int(t['b'])]-truth[int(t['a'])] for t in chosen)
        insertion=sum(truth[int(t['b'])] for t in chosen);removal=sum(truth[int(t['a'])] for t in chosen)
        oracle=before+sum(max(0,truth[int(t['b'])]-truth[int(t['a'])]) for t in chosen)
        out[mode]={'selected':len(chosen),'blind_swap_hits':before+blind_gain,'blind_swap_gain_count':insertion,'blind_swap_harm_count':removal,
            'selected_action_oracle_hits':oracle,'selected_true_positive_gain_queries':sum(truth[int(t['b'])]>truth[int(t['a'])] for t in chosen),
            'selected_true_harm_queries':sum(truth[int(t['b'])]<truth[int(t['a'])] for t in chosen),'selected_zero_queries':sum(truth[int(t['b'])]==truth[int(t['a'])] for t in chosen),
            'predicted_ideal_swap_gain':sum(t.get('ideal_swap_gain',0) for t in chosen),'potential':decision['selected_potential'],
            'proxy_unilateral_gain':decision['selected_max_unilateral_gain'],'game_central_proxy_gap':decision['central_potential_gap']}
    return {'baseline_hits':before,'proposed':len(targets),'proposed_b_target_occurrences':sum(truth[int(t['b'])] for t in targets),
        'proposed_positive_gain_queries':sum(truth[int(t['b'])]>truth[int(t['a'])] for t in targets),'modes':out}


def run(root=OUT,calibrator=RESULT,data=DATA):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    ids,items,rows,labels,arrays=load_fold(data);splits=split_indices(ids);diag=splits['diagnostic'];item_lookup={int(m):j for j,m in enumerate(items)}
    p=np.load(Path(calibrator)/'hold20.expected_count.npy',mmap_mode='r');members=np.load(Path(calibrator)/'hold20.member_expected_count.npy',mmap_mode='r')
    candidates=np.load(Path(calibrator)/'candidate_ids.npy',mmap_mode='r');offsets=np.load(Path(calibrator)/'candidate_offsets.npy',mmap_mode='r')
    mass_model=joblib.load(Path(calibrator)/'macro_mass_baseline.joblib');case_records=[];proposal_cases=[];curve_rows=[]
    for ordinal,q in enumerate(diag):
        q=int(q);sl=slice(offsets[q],offsets[q+1]);mids=np.asarray(candidates[sl]);indices=np.array([item_lookup[int(m)] for m in mids]);scores=np.asarray(p[sl]);member=np.asarray(members[:,sl])
        sr=np.asarray(arrays['sparse_ranks'][q,indices],dtype=float);kr=np.asarray(arrays['kbrd_ranks'][q,indices],dtype=float)
        x=candidate_features(rows[q],items,indices,**{k:np.asarray(v[q]) for k,v in arrays.items()})
        mass=mass_model['estimator'].predict_proba(mass_model['scaler'].transform(x))[:,1]
        rankings={'ranker':mids[np.argsort(-scores,kind='stable')].tolist(),'sparse':mids[np.argsort(sr,kind='stable')].tolist(),
            'kbrd':mids[np.argsort(kr,kind='stable')].tolist(),'rrf60':mids.tolist(),'macro_mass':mids[np.argsort(-mass,kind='stable')].tolist()}
        targets=labels[q]['targets'];truth=Counter(targets)
        case_records.append({'id':ids[q],**{name:micro_sums(rank,targets) for name,rank in rankings.items()}})
        selected=set(rankings['ranker'][:10])
        curve_rows.extend((float(scores[j]),truth[int(m)],1/len(mids),int(m) in selected) for j,m in enumerate(mids))
        initial=[list(map(str,rankings[k][:10])) for k in ('sparse','kbrd','rrf60')];pool=list(map(str,mids));borda=list(map(int,consensus_ranking(initial,pool)))
        counts=dict(zip(pool,map(float,scores)));member_maps=[dict(zip(pool,map(float,member[i]))) for i in range(3)]
        # The fixed proxy uses only fold80-produced scores, never full-train on-train evidence.
        nz=np.asarray(arrays['kbrd_logits'][q,indices],dtype=float);nz=(nz-nz.min())/max(float(nz.max()-nz.min()),1e-9)
        ns=np.asarray(arrays['sparse_scores'][q,indices],dtype=float)
        features={m:{'neighbor_score':max(0.,float(ns[i])),'evidence_score':float(.5/(60+sr[i])+.5/(60+kr[i])),'kg_score':float(nz[i])} for i,m in enumerate(pool)}
        sets={'borda_fixed_proxy':(borda,build_targets(initial,pool,features,max_targets=6)),
            'borda_calibrated_swap':(borda,calibrated_targets(borda,counts,member_maps,initial)),
            'ranker_calibrated_uncertainty':(rankings['ranker'],calibrated_targets(rankings['ranker'],counts,member_maps,initial))}
        for accuracy in (1.,.75,.5):sets['ranker_netVOI_q'+str(accuracy)]=(rankings['ranker'],net_voi_targets(rankings['ranker'],counts,member_maps,initial,accuracy))
        proposal_cases.append({'id':ids[q],'item_targets':len(targets),'variants':{name:proposal_record(base,ts,truth) for name,(base,ts) in sets.items()}})
        if ordinal%100==0:print('audit',ordinal,'/',len(diag),flush=True)
    ci={name:paired_cluster_ci(case_records,'ranker',name) for name in ('sparse','kbrd','rrf60','macro_mass')}
    curve=[];edges=(0,.001,.0025,.005,.01,.025,.05,.1,.2,.5,1,2,float('inf'))
    for which in ('all_candidates','ranker_top10'):
        records=[r for r in curve_rows if which=='all_candidates' or r[3]]
        a=np.array([r[:3] for r in records]);pval,y,w=a.T
        for low,high in zip(edges[:-1],edges[1:]):
            mask=(pval>=low)&(pval<high)
            if mask.any():curve.append({'scope':which,'low':low,'high':high if math.isfinite(high) else 'infinity','candidate_rows':int(mask.sum()),
               'predicted_count':float(np.average(pval[mask],weights=w[mask])),'actual_count':float(np.average(y[mask],weights=w[mask]))})
    totals={}
    for variant in proposal_cases[0]['variants']:
        rowsv=[r['variants'][variant] for r in proposal_cases];entry={'baseline_hits':sum(r['baseline_hits'] for r in rowsv),'proposed':sum(r['proposed'] for r in rowsv),
            'proposed_b_target_occurrences':sum(r['proposed_b_target_occurrences'] for r in rowsv),'proposed_positive_gain_queries':sum(r['proposed_positive_gain_queries'] for r in rowsv),'modes':{}}
        for mode in ('central','game'):
            keys=('selected','blind_swap_hits','blind_swap_gain_count','blind_swap_harm_count','selected_action_oracle_hits','selected_true_positive_gain_queries','selected_true_harm_queries','selected_zero_queries','predicted_ideal_swap_gain')
            entry['modes'][mode]={k:sum(r['modes'][mode][k] for r in rowsv) for k in keys}
            entry['modes'][mode]['max_game_central_proxy_gap']=max(r['modes'][mode]['game_central_proxy_gap'] for r in rowsv)
        totals[variant]=entry
    summary={'scope':'held-conversation original-train diagnostic; no public valid/test, no API; predictor/proposal gains are not real review effects',
        'diagnostic_contexts':len(diag),'diagnostic_conversations':len({conv(ids[i]) for i in diag}),'item_targets':sum(len(labels[i]['targets']) for i in diag),
        'cluster_comparisons':ci,'proposal_variants':totals,'source_sha256':sha(Path(__file__)),
        'calibrator_sha256':sha(Path(calibrator)/'calibrated_model.joblib'),'nominal_review_cost_cny':.0015,'expected_hits_per_cny':1.,
        'q_scope':'q=1 is an oracle acquisition envelope, q=.75/.5 are assumed sensitivity only; no review reliability was measured',
        'sigma_scope':'three fitted-model stability standard deviation, not established Bayesian uncertainty',
        'blind_swap_scope':'deterministic numerical swap simulation; all candidate/count predictions frozen before reading diagnostic outcomes'}
    create(root/'summary.json',summary);create(root/'paired_cases.json',case_records);create(root/'proposal_cases.json',proposal_cases);create(root/'calibration_curve.json',curve)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),constrained_layout=True)
    colors={'all_candidates':'#2864a0','ranker_top10':'#d06a26'}
    bound=max(max(p['predicted_count'],p['actual_count']) for p in curve)*1.08
    for ax,limit,title in zip(axes,(bound,.05),('Held-conversation count calibration','Decision-frontier zoom')):
        ax.plot([0,limit],[0,limit],color='#777777',linestyle='--',linewidth=1,label='ideal')
        for scope in colors:
            points=[p for p in curve if p['scope']==scope and p['predicted_count']<=limit and p['actual_count']<=limit]
            ax.plot([p['predicted_count'] for p in points],[p['actual_count'] for p in points],marker='o',markersize=4,linewidth=1.2,color=colors[scope],label=scope.replace('_',' '))
        ax.set(xlim=(0,limit),ylim=(0,limit),xlabel='Predicted target occurrence count',ylabel='Observed target occurrence count',title=title);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Count calibration: 1,157 prefixes / 439 original conversations',fontsize=11)
    fig.savefig(root/'calibration_curve.png',dpi=180);fig.savefig(root/'calibration_curve.svg');plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True);return summary

if __name__=='__main__':run()
