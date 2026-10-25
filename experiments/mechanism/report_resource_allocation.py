"""Offline diagnostic report; requires a complete, committed collection."""
import json
from math import comb
import subprocess
import sys

from experiments.mechanism.probe_resource_allocation import OUT, ROOT, analyze
from experiments.mechanism.run_upgrade_pilot import read_rows, freeze
from experiments.mechanism.complete_legacy_gain_audit import frozen_hashes


def paired(rows, left, right):
    gains=sum(r[left]==0 and r[right]==1 for r in rows)
    harms=sum(r[left]==1 and r[right]==0 for r in rows)
    n=gains+harms
    p=min(1.,2*sum(comb(n,k) for k in range(min(gains,harms)+1))/2**n) if n else 1.
    return gains,harms,p


def render(result):
    rows=result['rows'];n=result['n'];counts=result['correct_counts']
    gain,harm,p=paired(rows,'all_low','all_high')
    c=result['cost']
    lines=['# 执行模型与资源分配：24题诊断结果','',
           '## 研究问题','',
           '固定报告指令与输出上限，仅改变执行模型，检查信息增量是否存在；再比较旧合同与简单分配。此处不验证新均衡或AEIC字段不可替代性。','',
           '## 采集与比较','',
           f"已开卷开发题{n}道、每题3角色；成功响应{result['calls_collected']}份。报告先选择后采集，完整采集后离线评分。所有方法复用同一份潜在报告池，并非分别在线重跑。",'',
           '| 方法 | 正确题数/24 | 比较含义 |','|---|---:|---|']
    names=[('all_low','全GLM','低资源参照'),('all_high','全DeepSeek','模型执行变化'),
           ('contract','旧合同选择','固定旧信用与成本，两行动子问题'),
           ('fixed_quota','固定配额','相同DeepSeek报告数，无选择调用'),
           ('same_k_random_expected','逐题同k随机期望','保持每题升级数量'),
           ('global_quota_random_expected','全局等配额随机期望','保持总升级数量'),
           ('realized_oracle','标签oracle上限','不可部署，仅检查保存报告池')]
    for key,name,meaning in names:lines.append(f'| {name} | {counts[key]:.3f} | {meaning} |')
    lines+=['','## 信息增量','',
            f'全低→全高：纠正{gain}题、伤害{harm}题，净变化{gain-harm:+d}题；配对Exact McNemar双侧p={p:.6g}。小样本探索性单项比较，不作确认性显著性结论。',
            f"报告槽位低→高：{result['report_slot_low_to_high']['wrong_correct']}次纠正、{result['report_slot_low_to_high']['correct_wrong']}次伤害。槽位相关，不把72份报告当72道独立题。",'',
            '| 域 | 全低 | 全高 | 合同 |','|---|---:|---:|---:|']
    for domain in ['strategyqa','mmlu','truthfulqa']:
        subset=[r for r in rows if r['id'].startswith(domain+':')]
        values=[sum(r[k] for r in subset) for k in ['all_low','all_high','contract']]
        lines.append(f'| {domain} | {values[0]}/8 | {values[1]}/8 | {values[2]}/8 |')
    lines+=['','## 资源与开销','',
            f"合同使用{result['high_report_quota']}/72份DeepSeek报告。合同共{c['contract_total_calls']}次调用、{c['contract_total_tokens']} Token，其中选择器{c['selector_calls']}次、{c['selector_tokens']} Token。", 
            f"固定配额72次调用、{c['fixed_report_tokens']} Token；等配额随机72次调用、期望{c['random_report_tokens_expected']:.1f} Token。全低{c['all_low_report_tokens']} Token，全高{c['all_high_report_tokens']} Token，均72次报告调用。",
            f"本轮实际采集共{c['collection_total_tokens']} Token；逐请求串行墙钟耗时之和{c['collection_serial_wall_seconds']:.1f}秒。它不是所有部署方法的端到端延迟，也不含旧实验或v1 HTTP400的时间。",'',
            '等升级次数不等于等总预算。缺少可靠计价时不将Token相同等同于费用相同。','',
            '## 可保留与不能声称','']
    if gain>harm:
        lines+=['本次保存报告池支持“执行模型变化可以带来正向信息增量”的局部观察；不能外推为所有模型/领域的稳定收益。']
    else:
        lines+=['本次全高没有优于全低；不能据此声称这条执行桥产生正向增量，也不能证明所有模型升级均无用。']
    if counts['contract']>max(counts['fixed_quota'],counts['global_quota_random_expected']):
        lines+=['合同在本次回放数值上超过两项等配额参照，但额外选择成本、小样本和已开卷边界尚未解决；不是分配机制优势的独立验证。']
    else:
        lines+=['本次合同未同时超过固定配额和全局随机期望；尚无证据支持旧合同具有额外的分配优势。']
    lines+=['旧合同升级成本1高于最高信用0.4，模型选择升级不能证明效用最优；不得附会为理性均衡。AEIC在本轮是固定报告接口，未与普通报告作隔离对照，故不归因于四字段。',
            '不开启300题，不改奖励追结果；后续若提出新分配规则，须独立定义可观测输入、支付与成本，并先与无选择开销的简单策略比较。','',
            '实际返回模型：'+json.dumps(result['actual_returned_models'],ensure_ascii=False)+'。',
            f"字段异常记录{len(result['schema_issues'])}份；合法C与字段完整性分开，不以字段缺失替代答案错误。",'',
            '原始协议、请求、响应、时间、Token及逐题结果见`experiments/results/resource_allocation/probe_v2/`。v1另有1次HTTP400，没有成功答案；完整保留，不计为成功生成。']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    result=analyze()
    tests=subprocess.run([sys.executable,'-m','unittest',
        'experiments.tests.test_resource_allocation_probe',
        'experiments.tests.test_upgrade_pilot','experiments.tests.test_upgrade_v3'],
        cwd=ROOT,capture_output=True,text=True)
    print(tests.stdout+tests.stderr)
    if tests.returncode:raise RuntimeError('verification tests failed')
    diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    if diff.returncode:raise RuntimeError('diff check failed: '+diff.stdout+diff.stderr)
    old_frozen=frozen_hashes()
    attempts=read_rows(OUT/'attempts.jsonl')
    previous=read_rows(OUT.parent/'probe_v1/attempts.jsonl')
    freeze(OUT/'acceptance.json',{
        'status':'COLLECTION_AND_REPLAY_VERIFIED_NOT_EFFICACY_ACCEPTANCE',
        'v2_attempts':len(attempts),'v1_failed_attempts':len(previous),
        'new_attempts_total':len(attempts)+len(previous),
        'successful_v2_responses':result['calls_collected'],
        'original_frozen_files_verified':len(old_frozen),
        'related_tests_passed':True,'git_diff_check_passed':True,
        'limitations':['reused development questions','one report draw per path',
            'fixed old dominated upgrade payoff','potential report replay not independent online runs']})
    path=ROOT/'docs/research/资源分配小验证_结果_2026-10-22.md'
    # Artifact generation, not edits to research sources.
    with path.open('x',encoding='utf-8') as stream:stream.write(render(result))
    print(path)
