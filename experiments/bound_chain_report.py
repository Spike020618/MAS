"""Join sealed action opportunities with pre-generation allocations and real effects."""
import json
from pathlib import Path
from experiments.consensus_recommendation.common import ROOT, create, fingerprint
from experiments.consensus_mechanism import core as c
from experiments.consensus_mechanism.data import read

def sealed(root):
    rows=read(root/'manifest.private.json')
    cases=[read(root/'cases'/f'case_{i}.json')for i in range(len(rows))]
    expected={'manifest_hash':fingerprint(rows),'hashes':[fingerprint(x)for x in cases]}
    if read(root/'generation_complete.json')!=expected:raise ValueError('seal mismatch: '+str(root))
    return rows,cases

def report():
    probe=ROOT/'experiments/results/bound_opportunity_v3'
    bridge=ROOT/'experiments/results/bound_game_bridge_v3'
    rows,opportunities=sealed(probe);other,executions=sealed(bridge)
    if fingerprint(rows)!=fingerprint(other):raise ValueError('different bound states/tasks')
    forecasts=read(bridge/'selection.before_generation.json');details=[]
    for row,panel,case in zip(rows,opportunities,executions):
        arms={k:panel['arms'][k+':linked']for k in ['empty','A','B','AB']}
        valid=all(x['event']['valid']for x in arms.values())
        # Only after BOTH seals: private label access here never controls selection.
        quality={k:c.quality(x['decision'],row['evaluation'])for k,x in arms.items()}
        gains={k:quality[k]-quality['empty']for k in ['A','B','AB']}if valid else None
        game_start=forecasts[row['task']['id']]['game']
        entry={'task_id':row['task']['id'],'stratum':row['stratum'],'all_panel_branches_valid':valid,
               'observed_linked_evidence_gains':gains,
               'predicted_game_initial_sources':[a['source_ids']for a in game_start[0]],
               'game_initial_potential':game_start[1]['potential'],
               'game_actual_actions':[e['action']['source_ids']for e in case['methods']['game']['events']],
               'method_quality':{m:c.quality(x['decision'],row['evaluation'])for m,x in case['methods'].items()}}
        entry['observed_signal_but_game_initially_stops']=bool(valid and max(gains.values())>0 and not game_start[0])
        before=c.quality(row['state']['decision'],row['evaluation'])
        entry['executed_alternative_improves_but_game_stops']=bool(not game_start[0] and any(
            entry['method_quality'][m]>before for m in ('fixed','random')))
        ordinary={k:panel['arms'][k+':ordinary']for k in ['empty','A','B','AB']}
        if all(v['event']['valid']for v in ordinary.values()):
            oq={k:c.quality(v['decision'],row['evaluation'])for k,v in ordinary.items()}
            entry['ordinary_evidence_gains']={k:oq[k]-oq['empty']for k in ['A','B','AB']}
        else:entry['ordinary_evidence_gains']=None
        details.append(entry)
    result={'details':details,'counts':{'tasks':len(rows),
             'valid_panel_tasks':sum(x['all_panel_branches_valid']for x in details),
             'observed_signal_but_game_initially_stops':sum(x['observed_signal_but_game_initially_stops']for x in details),
             'executed_alternative_improves_but_game_stops':sum(x['executed_alternative_improves_but_game_stops']for x in details)},
             'scope':'Cross-run controlled diagnostic; member/context and sampling differences remain. Not an unbiased causal comparison on natural tasks.'}
    out=ROOT/'experiments/results/bound_chain_report_v2';create(out/'summary.json',result)
    lines=['# 固定机会—动作—决策—博弈选择：实测对照','',
           '两批均核对封存后生成；构造域诊断，不替代自然推荐主表。','',
           '| 类型 | 有效四分支 | 材料增量 A/B/AB | 博弈初次选材 | 博弈最终质量 |','|---|---|---|---|---|']
    for x in details:
        lines.append('| '+ ' | '.join([x['stratum'],str(x['all_panel_branches_valid']),str(x['observed_linked_evidence_gains']),str(x['predicted_game_initial_sources']),str(x['method_quality']['game'])])+' |')
    lines.extend(['','有效四分支显示材料增量而博弈初次停止的任务数：'+str(result['counts']['observed_signal_but_game_initially_stops']),
                  '同一核查执行器下，固定/随机实际纠错而博弈初次停止的任务数：'+str(result['counts']['executed_alternative_improves_but_game_stops']),
                  '', '若此数大于0，优先检验旧估计器跨域校准与零收益训练分布；不能从构造任务直接推出自然任务也有同样漏检。'])
    out.mkdir(parents=True,exist_ok=True);(out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':report()
