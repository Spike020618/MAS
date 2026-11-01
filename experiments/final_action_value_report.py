"""Independent sealed report, including full-information single and independent controls."""
import collections,json
from pathlib import Path
from experiments.consensus_mechanism import core as c
from experiments.consensus_mechanism.data import read
from experiments.consensus_recommendation.common import ROOT,create,fingerprint
from experiments.calibrated_value_report import paired
from experiments.paired_deliberation.native_optimization_runtime import audit

def verify(root,tasks):
    cases=[read(root/'cases'/f'case_{i}.json')for i in range(len(tasks))]
    if read(root/'generation_complete.json')!={'task_ids':[t['id']for t in tasks],'hashes':[fingerprint(x)for x in cases]}:raise ValueError('seal mismatch: '+str(root))
    return cases

def report(root):
    tasks=[t for t in read(root/'tasks.json')if t['metadata']['acquisition_partition']=='test']
    source=ROOT/'experiments/data/public/climate_fever/climate-fever.jsonl'
    from experiments.consensus_recommendation.common import sha
    if sha(source)!=read(root/'candidate_pool/data_manifest.json')['source_sha256']:raise ValueError('native label source drift')
    originals={str(r['claim_id']):r for line in source.read_text().splitlines()if line.strip()for r in [json.loads(line)]}
    private=read(root/'test.labels.private.json')
    for t in tasks:
        cid=t['metadata']['native_id'].split('/')[-1]
        if private[t['id']]['answer']!={'SUPPORTS':'supported','REFUTES':'refuted','NOT_ENOUGH_INFO':'not_enough_info'}[originals[cid]['claim_label']]:raise ValueError('native label mismatch')
    main=verify(root/'evaluation_v2',tasks)
    unions=verify(root/'state_union',tasks);ridge_unions=verify(root/'ridge_state_union',tasks)
    panels=verify(root/'heldout_panel',tasks);labels=read(root/'test.labels.private.json')
    decisions={m:[x['methods'][m]['decision']for x in main]for m in main[0]['methods']}
    events={m:[e for x in main for e in x['methods'][m]['events']]for m in main[0]['methods']}
    for prefix,cases in [('union',unions),('ridge_union',ridge_unions)]:
        for m in ('game','greedy','central'):
            decisions[prefix+'_'+m]=[x['methods'][m]['decision']for x in cases]
            events[prefix+'_'+m]=[e for x in cases for e in x['methods'][m]['events']]
    decisions['single_full']=[x['branches']['AB']['decision']for x in panels]
    events['single_full']=[x['branches']['AB']['event']for x in panels]
    decisions['independent']=[];events['independent']=[]
    for t,case in zip(tasks,panels):
        arms=[case['branches'][m]for m in ('A','B')]
        valid=[a['decision']for a in arms if a['event']['valid']]
        decisions['independent'].append(c.aggregate(t,valid)if valid else t['initial'])
        events['independent'].extend(a['event']for a in arms)
    before=[c.quality(t['initial'],labels[t['id']])for t in tasks]
    quality={m:[c.quality(d,labels[t['id']])for t,d in zip(tasks,ds)]for m,ds in decisions.items()}
    metrics={m:{'correct':int(sum(q)),'initial_correct':int(sum(before)),
        'improved':sum(y>x for x,y in zip(before,q)),'harmed':sum(y<x for x,y in zip(before,q)),
        'calls':len(events[m]),'invalid':sum(not e['valid']for e in events[m])}for m,q in quality.items()}
    comparisons={m:paired(quality['union_game'],q)for m,q in quality.items()if m!='union_game'}
    costs={};roots=['calibrated_action_value_v1/train','calibrated_action_value_v2/train',
        'calibrated_action_value_v2/evaluation','calibrated_action_value_v2/same_data_ridge',
        'calibrated_action_value_v2/heldout_panel','calibrated_action_value_v3/evaluation',
        'calibrated_action_value_v3/evaluation_v2','calibrated_action_value_v3/state_union',
        'calibrated_action_value_v3/ridge_state_union','calibrated_action_value_v3/heldout_panel']
    for path in roots:costs[path]=audit(ROOT/'experiments/results'/path,'qwen_flash')
    result={'tasks':len(tasks),'metrics':metrics,'paired_union_game':comparisons,
      'costs':costs,'configured_known_cny':sum(x['configured_known_cny']for x in costs.values()),
      'scope':'Natural human-labelled claim evidence with controlled provisional decisions. Shared reviewer model; two evidence packets fit within the source budget. Component diagnostic, not proof of unique multi-agent game gain.'}
    dest=root/'final_report';create(dest/'summary.json',result)
    display=['old_game','base_game','calibrated_game','ridge_game','union_game','union_greedy','union_central',
             'ridge_union_game','prior_game','fixed','random','single_full','independent','reflection']
    names={'old_game':'历史贡献模型＋博弈','base_game':'新概率估计（未校正方向）＋博弈',
      'calibrated_game':'收益方向校准＋博弈','ridge_game':'同训练数据Ridge＋博弈',
      'union_game':'条件贡献＋博弈','union_greedy':'条件贡献＋贪心','union_central':'条件贡献＋中央',
      'ridge_union_game':'Ridge条件贡献＋博弈','prior_game':'总体均值＋博弈','fixed':'固定顺序',
      'random':'随机选择','single_full':'单Agent一次读完整资料','independent':'两次独立核查后汇总','reflection':'只反思，不读资料'}
    lines=['# 贡献值修订：最终实测验收','',result['scope'],'',
      '| 方法 | 正确/24 | 纠正 | 损害 | 核查调用 | 无效 |','|---|---:|---:|---:|---:|---:|']
    for m in display:
        r=metrics[m];lines.append('| '+' | '.join(map(str,[names[m],str(r['correct'])+'/24',r['improved'],r['harmed'],r['calls'],r['invalid']]))+' |')
    lines.extend(['','## 结果归因','',
      '有效性与纠正/损害分开建模，非法响应不再充当零收益训练；收益方向在原四条校准主张上拟合，新24条不用于调参。',
      '条件贡献定义为F(已读材料并上新动作)-F(已读材料)，保留固定初始参照，避免第二步忘记第一步。',
      '原evaluation存在方法别名导致的逻辑槽位碰撞，完整保留；evaluation_v2按方法命名空间回放，复用成功物理回执，不重发unknown、不改模型/请求/数据。',
      '单Agent完整资料和独立汇总来自同状态empty/A/B/AB真实响应。它们是在选择器冻结并封存后补充的标准诊断对照，不能反馈给本批训练。',
      '两个资料包均可在预算内一次读完，因此本批不满足严格的材料选择稀缺条件。不能把相对零行动模型的提升解释为博弈独有收益；多疑点竞争、异质核查者和推荐迁移仍须验证。',
      '', '## 配对区间（条件贡献＋博弈为参照）',''])
    for m in ('old_game','base_game','ridge_game','fixed','single_full','independent'):
        s=comparisons[m];lines.append('- '+names[m]+'：差值'+str(s['difference'])+'，新增/损失'+str(s['wins'])+'/'+str(s['losses'])+'，95%区间'+str(s['bootstrap_95']))
    lines.extend(['','## 推荐训练集的完整机会审计','',
      '独立核对原训练封存：60条推荐，38条初始零命中、5条部分命中、17条全部命中。20条存在候选池内Top10外目标（其中19条初始零命中、1条部分命中）。不能用截取的6条候选清单代表完整机会总数；机械可换入不等于材料充分或语义有依据。',
      '', '## 费用','', '本轮配置已知费用合计约'+str(result['configured_known_cny'])+'元；unknown完成另有计费不确定性，未补发。重复回放不重复计入新增费用。'])
    (dest/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'metrics':metrics,'configured_known_cny':result['configured_known_cny']},ensure_ascii=False,indent=2))

if __name__=='__main__':report(ROOT/'experiments/results/calibrated_action_value_v3')
