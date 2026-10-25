"""Offline report and integrity checks; no additional model calls."""
import hashlib
import json
import re
import subprocess
import sys

from experiments.mechanism.probe_probability_routing import OUT,analyze
from experiments.mechanism import probe_resource_allocation as p
from experiments.mechanism.run_upgrade_pilot import freeze
from experiments.mechanism.complete_legacy_gain_audit import frozen_hashes


def fmt(v):return str(round(v[0],4)) if v[0]==v[1] else f'{v[0]:.4g}—{v[1]:.4g}'


def render(result):
    selections=json.loads((OUT/'selections.json').read_text())
    labels={'all_low':'全弱模型','all_high':'直接全强模型','disagreement':'分歧门控',
            'probability':'概率均值不确定性','individual_uncertainty':'个体不确定性均值',
            'fixed':'固定哈希配额','length':'请求长度排序','random_expected':'同配额随机期望'}
    quota=result['quota_questions'];counts=result['correct_count_bounds']
    lines=['# 概率识别升级价值：24题诊断结果','',
        '## 固定问题与执行','',
        '检验执行前的自报概率能否比结论分歧更好地挑选实际升级纠正，而不是只预测初始错误。使用原24道开卷开发题，不按结果换题；没有奖励、拟合或领域规则。不是独立验证，也不是博弈机制验收。',
        '弱/强后端在相同角色、同提示、温度0.3和600输出上限下，同一次调用输出原生答案和概率。先完成72次弱请求并封存全部分配，再采集72次强对照；整批结束并审计后才读标签。概率是模型自述，不是token likelihood或已校准后验。',
        f"本轮{result['new_attempts']}次尝试，{result['successful_responses']}份响应，{result['unknown']}次未知传输；成功响应Token共{result['collection_total_known_tokens']}。",'',
        '## 等升级次数比较','',
        f'分歧门控选中{quota}/24题，故各路由策略固定升级{quota}题、每题3个角色，共{3*quota}次强调用；都另付72次初始弱请求。直接全强不支付初始弱请求。','',
        '| 方法 | 正确题数/24 | 已知Token |','|---|---:|---:|']
    for name in ['all_low','all_high','disagreement','probability','individual_uncertainty','fixed','length','random_expected']:
        lines.append(f"| {labels[name]} | {fmt(counts[name])} | {result['known_token_costs'][name]:.1f} |")
    if quota in [0,24]:lines+=['','配额处于0或24的边界，没有实质路由分配辨识空间，不能宣称策略优势。']
    lines+=['','若存在区间，表示缺失响应的可能答案界而非置信区间。随机是所有相同题数分配的期望，不是取最佳随机种子。各方法复用同一份潜在结果池，不是分别重新在线执行。','',
        '## 真正纠正与伤害','',
        '| 分配策略 | 选中题目的纠正数 | 选中题目的伤害数 |','|---|---:|---:|']
    for name,pair in result['selected_corrections_harms'].items():
        lines.append(f"| {labels[name]} | {fmt(pair['corrected_bounds'])} | {fmt(pair['harm_bounds'])} |")
    primary=counts['probability'];baseline=counts['disagreement']
    if primary[0]>baseline[1]:decision='主策略在本批保存结果中数值超过分歧门控，保留为探索信号，不能宣布概率路由已独立有效。'
    elif primary[1]<baseline[0]:decision='主策略在本批保存结果中低于分歧门控，没有支持它替代已有简单策略的证据。'
    else:decision='主策略没有超过分歧门控的明确正确数增量，不把概率字段或更详细输出本身当作分配贡献。'
    lines+=['',decision]
    gate=set(selections['policies']['disagreement']);prob=set(selections['policies']['probability'])
    lines.append(f'概率与分歧策略共同选中{len(gate&prob)}题，各自独有{len(prob-gate)}和{len(gate-prob)}题。总正确数相同不自动意味着选题相同。')
    diag=result['gain_identification']
    lines+=['','## 概率质量与价值识别','',
        '每题三份分布均合法且升级结果明确的题数：'+str(diag['eligible_questions'])+'；其中低错高对的题数：'+str(diag['gain_positive_questions'])+'。',
        '预测升级纠正事件的描述性AUC：概率均值不确定性='+str(diag['ensemble_uncertainty_gain_auc'])+'；个体不确定性='+str(diag['individual_uncertainty_gain_auc'])+'。None表示无正例或负例，不能计算。AUC不直接证明校准、低预算优势或激励相容。','',
        '| 后端 | 合法概率槽位/72 | Brier均值 | 同槽位均匀预测Brier |','|---|---:|---:|---:|']
    for backend,values in result['brier_descriptive'].items():
        lines.append(f"| {backend} | {result['valid_probability_slots'][backend]} | {values['mean']} | {values['uniform_mean_same_slots']} |")
    invalid=sum(not r['features']['probabilities_valid'] for r in result['rows'])
    lines+=['',f'{invalid}/24题至少一份概率无效。非法分布不归一化、不补写、不补问，回退优先级必须与真正概率信号分开。Brier按槽位汇总只是描述，不能将72角色当作72独立题。',
        '本轮答案接口不再要求AEIC论据，与旧AEIC批次不同；不能直接把跨批次正确数差归因于模型或协议。','',
        '## 研究结论边界','',
        '同升级次数不等于同总费用。所有路由都支付初始弱调用，概率没有额外模型选择器，但分布输出本身仍消耗Token。未知请求费用不记零；Token不能替代跨后端货币成本。',
        '本轮只检查升级价值识别的必要条件，不证明任何支付能诱导诚实报告。尤其不能使用“升级前后评分差”直接宣布激励相容，因为故意压低初始预测可能获得虚假改善奖励。',
        '若概率不优于简单策略，停止本轮扩展，不调阈值或奖励追效果；若有正向数值差，仍需固定规则后来源题干隔离验证，不自动进入300题。论文、专利、系统、PPT未修改。','',
        '实际返回模型：'+json.dumps(result['actual_models'],ensure_ascii=False)+'。',
        '原始记录与封存选择：`experiments/results/resource_allocation/probability_routing_v1/`。']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    result=analyze()
    modules=['test_probability_routing','test_dispute_review','test_aeic_context_probe','test_low_cost_router',
             'test_resource_allocation_probe','test_upgrade_pilot','test_upgrade_v3']
    tests=subprocess.run([sys.executable,'-m','unittest',*['experiments.tests.'+m for m in modules]],cwd=p.ROOT,text=True,capture_output=True)
    count=re.search(r'Ran (\d+) tests',tests.stderr)
    if tests.returncode or not count:raise ValueError('tests failed: '+tests.stdout+tests.stderr)
    diff=subprocess.run(['git','diff','--check'],cwd=p.ROOT,text=True,capture_output=True)
    if diff.returncode:raise ValueError('diff check failed: '+diff.stdout+diff.stderr)
    path=p.ROOT/'docs/research/概率识别升级价值_结果_2026-10-22.md';content=render(result)
    if path.exists():
        if path.read_text()!=content:raise ValueError('existing report mismatch')
    else:
        with path.open('x',encoding='utf-8') as stream:stream.write(content)
    files=[*OUT.glob('*.json'),*OUT.glob('*.jsonl'),path,
        p.ROOT/'experiments/mechanism/probe_probability_routing.py',
        p.ROOT/'experiments/mechanism/report_probability_routing.py',
        p.ROOT/'experiments/tests/test_probability_routing.py',
        p.ROOT/'docs/research/概率识别升级价值_方案_2026-10-22.md']
    freeze(OUT/'acceptance.json',{'status':'OPENED_DATA_VALUE_SIGNAL_PROBE_VERIFIED_NOT_GAME_ACCEPTANCE',
        'tests_passed':int(count[1]),'old_frozen_files_verified':len(frozen_hashes()),
        'git_diff_check_passed':True,'new_attempts':result['new_attempts'],
        'successful_responses':result['successful_responses'],'unknown':result['unknown'],
        'api_calls_during_verification':0,'snapshot_sha256':{
            str(f.relative_to(p.ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.name!='acceptance.json'}})
    print(json.dumps({'report':str(path),'tests':int(count[1]),'new_attempts':result['new_attempts']}))
