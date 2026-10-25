"""Offline source-isolation backstop, results report, and acceptance snapshot."""
import hashlib
import json
import re
import subprocess
import sys

from experiments.mechanism import probe_resource_allocation as p
from experiments.mechanism.validate_routing_complementarity import OUT,DATA,analyze
from experiments.mechanism.prepare_routing_holdout import AUDIT,PINS
from experiments.mechanism.prepare_upgrade_pilot import canonical,strings
from experiments.mechanism.run_upgrade_pilot import freeze
from experiments.mechanism.complete_legacy_gain_audit import frozen_hashes


def verify_isolation():
    prov=json.loads((DATA/'provenance.json').read_text())
    coverage=json.loads((AUDIT/'backstop_coverage.json').read_text())
    if p.digest(coverage)!=prov['backstop_coverage_hash']:raise ValueError('coverage snapshot changed')
    if hashlib.sha256((AUDIT/'history_snapshot.json').read_bytes()).hexdigest()!=prov['history_snapshot_sha256']:
        raise ValueError('history snapshot changed')
    groups=prov['source_groups'];ids={g['id'].split(':',1)[1] for g in groups}
    stems={g['id']:canonical(g['base_question']) for g in groups}
    if len(set(stems.values()))!=24:raise ValueError('source stem duplication')
    for file in coverage:
        path=p.ROOT/file['path'];raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=file['sha256']:raise ValueError('history file drift '+file['path'])
        data=[json.loads(s) for s in raw.decode().splitlines() if s.strip()] if path.suffix=='.jsonl' else json.loads(raw)
        leaves=list(strings(data))
        if ids&set(leaves):raise ValueError('source ID exposure '+file['path'])
        content='\n'.join(canonical(v) for v in leaves)
        if any(stem in content for stem in stems.values()):raise ValueError('source stem exposure '+file['path'])
    for source in prov['sources']:
        path=__import__('pathlib').Path(source['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=PINS[source['domain']]:raise ValueError('original source changed')
    return {'history_files_indexed':prov['history_files'],'unique_files_backstop_verified':len(coverage),
        'source_stems':24,'source_stem_or_id_matches':0,
        'limit':'repository snapshot exact normalized stem/ID only, not paraphrase or model pretraining novelty'}


def fmt(value):return str(round(value[0],4)) if value[0]==value[1] else f'{value[0]:.4g}—{value[1]:.4g}'


def render(result,isolation):
    names={'all_low':'全弱','all_high':'直接全强','hybrid_union':'组合G∪P',
        'disagreement_priority':'分歧优先补足M','probability_top_m':'概率Top-M',
        'fixed':'固定哈希M','length':'请求长度M','random_expected':'随机M期望',
        'native_gate':'原分歧G，k题','native_probability':'原概率P，k题'}
    m=result['matched_quota'];k=result['native_quota']
    lines=['# 分歧与概率互补：未使用题组验证结果','',
        '## 前置锁定与来源隔离','',
        f"新24题、三域各8题；历史索引{isolation['history_files_indexed']}份文件，复核{isolation['unique_files_backstop_verified']}份唯一文件，选中24题来源题干/ID匹配0。只保证该仓库快照的规范化题干与ID排除，不保证语义改写或模型预训练未见。来源文件SHA固定，选题不使用标签或历史难度。", 
        '概率接口和十进制字符串兼容在采集前锁定，不补写、不归一化、不按新结果改规则。先72次弱调用封存选题，后72次强对照，完整采集并审计后才读取私有标签。',
        f"本轮{result['attempts']}次新尝试、{result['responses']}份响应、{result['unknown']}次未知传输；成功响应Token共{result['collection_total_tokens']}。不是支付机制、激励相容或均衡验收。",'',
        '## 相同升级预算的主比较','',
        f'G包含{k}题，P为概率最高的同数量{k}题；组合H为两者并集，共M={m}题。主比较各升级{m}题、{3*m}次强模型调用，均另付72次初始弱请求。', '',
        '| 方法 | 正确数/24 | 已知Token |','|---|---:|---:|']
    for name in ['all_low','all_high','hybrid_union','disagreement_priority','probability_top_m','fixed','length','random_expected']:
        lines.append(f"| {names[name]} | {fmt(result['counts'][name])} | {result['known_token_costs'][name]:.1f} |")
    if m in [0,24]:lines.append('\nM处于0或24边界，本轮没有实质路由分配辨识空间，不重新换题寻找差异。')
    lines+=['','未知响应若产生区间，它是可能答案界而非置信区间。所有方法复用同一份潜在结果，不冒称分别在线重跑；随机项是全部等配额分配的期望。','',
        '## 两项预设主对照','']
    for name,pair in result['primary_contrasts'].items():
        if 'corrections' in pair:
            lines.append(f"相对{names[name]}：组合纠正{pair['corrections']}、伤害{pair['harms']}、净差{pair['net']:+d}；Exact McNemar p={pair['exact_mcnemar_p']:.6g}，两主比较Holm p={pair.get('holm_p')}。")
        else:lines.append('相对'+names[name]+'：有未知结果，不能给唯一精确配对结论。')
    comparisons=result['primary_contrasts']
    if all(v.get('net',0)>0 for v in comparisons.values()):
        decision='组合在两项等预算主比较上均有正向数值差，可保留为小验证信号；是否统计支持须同时看校正p，不能称普适最优或博弈贡献。'
    else:
        decision='组合没有同时超过两个等预算单一信号参照。按预设，本轮不支持因“互补”而继续增加支付层或宣布新分配优势，不在这些题上改权重或阈值。'
    lines+=['',decision,'',
        '24题精度有限，没有差异不证明等效；数值优势也不代表已完成广泛确认。来源题干隔离使这批数据不同于前一天开卷探索，但不增加样本量或消除单次抽样局限。','',
        '## 原生预算与分域描述','',
        f"原G在{k}题升级时正确{fmt(result['counts']['native_gate'])}/24，原P同{k}题升级时正确{fmt(result['counts']['native_probability'])}/24。它们预算小于或等于组合；不能将组合相对这些数值的提升直接称效率增益。",'',
        '| 域 | 全弱 | 全强 | 组合 | 等预算分歧 | 等预算概率 |','|---|---:|---:|---:|---:|---:|']
    for domain in ['strategyqa','mmlu','truthfulqa']:
        subset=[r for r in result['rows'] if r['id'].startswith(domain+':')]
        values={name:[sum(r['bounds'][name][j] for r in subset) for j in [0,1]] for name in ['all_low','all_high','hybrid_union','disagreement_priority','probability_top_m']}
        lines.append('| '+domain+' | '+' | '.join(fmt(v)+'/8' for v in values.values())+' |')
    lines+=['','分域仅描述，不据此为不同任务单独调规则。','',
        '## 信号、成本与后续边界','',
        '合法概率槽位：'+json.dumps(result['valid_probability_slots'],ensure_ascii=False)+'；三份概率均合法且结果明确的题数'+str(result['valid_probability_questions'])+'。',
        f"完整概率题中升级纠正正例{result['gain_positive_valid_questions']}个，描述性AUC={result['gain_auc']}；不能当作校准或支付可靠性证明。", 
        '直接全强只有72次强调用，不支付初始弱调用。路由选择器没有额外模型调用，但初始概率输出和无效回退都要付费。相同强调用数不等于相同总Token或货币；未知服务端用量不记零。',
        '实际返回模型：'+json.dumps(result['actual_models'],ensure_ascii=False)+'。',
        '按前置方案不追加支付、不自动扩到300题、不根据本批结果换题或修改组合。前一天的探索结果保留，不能与本批数据拼接后冒称预注册大样本。论文、专利、系统与PPT未修改。','',
        '结果与原始日志：`experiments/results/resource_allocation/routing_holdout_validation_v1/`；来源隔离数据：`routing_holdout_data_v1/`。']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    result=analyze();isolation=verify_isolation()
    modules=['test_routing_holdout','test_probability_compatibility','test_probability_routing','test_dispute_review',
        'test_aeic_context_probe','test_low_cost_router','test_resource_allocation_probe','test_upgrade_pilot','test_upgrade_v3']
    tests=subprocess.run([sys.executable,'-m','unittest',*['experiments.tests.'+m for m in modules]],cwd=p.ROOT,text=True,capture_output=True)
    count=re.search(r'Ran (\d+) tests',tests.stderr)
    if tests.returncode or not count:raise ValueError('tests failed: '+tests.stdout+tests.stderr)
    diff=subprocess.run(['git','diff','--check'],cwd=p.ROOT,text=True,capture_output=True)
    if diff.returncode:raise ValueError('diff check failed: '+diff.stdout+diff.stderr)
    path=p.ROOT/'docs/research/分歧概率互补_未使用题组验证_结果_2026-10-23.md';content=render(result,isolation)
    if path.exists():
        if path.read_text()!=content:raise ValueError('existing report mismatch')
    else:
        with path.open('x',encoding='utf-8') as stream:stream.write(content)
    files=[*OUT.glob('*.json'),*OUT.glob('*.jsonl'),path,DATA/'provenance.json',
        p.ROOT/'experiments/mechanism/validate_routing_complementarity.py',
        p.ROOT/'experiments/mechanism/prepare_routing_holdout.py',
        p.ROOT/'experiments/mechanism/report_routing_holdout.py',
        p.ROOT/'experiments/tests/test_routing_holdout.py',
        p.ROOT/'docs/research/分歧概率互补_未使用题组验证_方案_2026-10-23.md']
    freeze(OUT/'acceptance.json',{'status':'SOURCE_SCREENED_SMALL_VALIDATION_VERIFIED_NOT_GAME_ACCEPTANCE',
        'tests_passed':int(count[1]),'old_frozen_files_verified':len(frozen_hashes()),
        'git_diff_check_passed':True,'source_isolation':isolation,'attempts':result['attempts'],
        'responses':result['responses'],'unknown':result['unknown'],'api_calls_during_verification':0,
        'snapshot_sha256':{str(f.relative_to(p.ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in files if f.name!='acceptance.json'}})
    print(json.dumps({'report':str(path),'tests':int(count[1]),'source_isolation':isolation}))
