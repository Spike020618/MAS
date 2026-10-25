"""Offline reporting and engineering verification for the locked context probe."""
import hashlib
import json
import re
import subprocess
import sys

from experiments.mechanism.probe_aeic_context import OUT,ARMS,analyze
from experiments.mechanism import probe_resource_allocation as p
from experiments.mechanism.run_upgrade_pilot import freeze
from experiments.mechanism.complete_legacy_gain_audit import frozen_hashes


def interval(v):return str(v[0]) if v[0]==v[1] else f'{v[0]}—{v[1]}'


def render(result):
    names={'independent':'独立重新求解','conclusions':'只看各方C','aeic':'看实际AEIC'}
    lines=['# AEIC争议审阅三臂验证：结果','',
        '## 这轮检验的范围','',
        '检验相同升级执行资源下，实际A/E/I上下文是否比只看结论C更有用。12道既有门控题全部纳入，7道有效分歧为主分析，5道含无效/弃答/缺失单列。开卷组件探索，不是独立确认，不涉及博弈均衡。',
        '三臂均重新执行，模型、角色、温度、输出上限和系统提示相同，仅同伴上下文不同；逐题逐角色交错顺序。报告独立生成，末轮严格多数。没有自适应重采样或格式补问。','',
        f"共{result['attempts']}次新尝试，{result['responses']}份响应，{result['unknown_requests']}次传输结果未知。整批结束后离线读标签；未知响应按所有可能原生选项与弃答给界。",'',
        '| 输入条件 | 7道有效分歧 | 5道含异常 | 全部12题 |',
        '|---|---:|---:|---:|']
    for arm in ARMS:
        counts=[interval(result['groups'][g]['correct_bounds'][arm]) for g in ['valid_disagreement','invalid_or_abstain','all']]
        lines.append(f'| {names[arm]} | {counts[0]}/7 | {counts[1]}/5 | {counts[2]}/12 |')
    lines+=['','范围表示缺失响应的不确定界，不是置信区间。合法C不因A/E/I缺失而判错。','',
        '## 主比较：AEIC相对C-only','']
    for group,title in [('valid_disagreement','主分析7题'),('all','全部12题'),('invalid_or_abstain','含异常5题')]:
        pair=result['groups'][group]['paired']['conclusions->aeic']
        if 'corrections' in pair:
            lines.append(f"{title}：纠正{pair['corrections']}题、伤害{pair['harms']}题；Exact McNemar双侧探索性p={pair['exact_mcnemar_p_exploratory']:.6g}。")
        else:lines.append(title+'：存在未知结果，不能给出唯一的纠正/伤害计数或精确p。')
    primary=result['groups']['valid_disagreement']['correct_bounds']
    if primary['aeic'][0]>primary['conclusions'][1]:
        decision='主分析观察到AEIC上下文相对C-only的正向数值差。仅作小样本信号，不能宣布独立验证成功，也不能证明四字段分别必要。'
    elif primary['aeic'][1]<primary['conclusions'][0]:
        decision='主分析AEIC低于C-only，本轮不支持增加实际论据上下文；不能将更多字段或更多Token包装为准确率优势。'
    else:
        decision='主分析没有观察到AEIC相对C-only的明确正向差，本轮不足以支持A/E/I的额外准确率贡献。不显著也不是等效性证明。'
    lines+=['',decision,'',
        '独立求解与C-only的比较用来区分更强执行路径和意见上下文；即使AEIC优于独立求解，若不优于C-only，也不能将收益归因于A/E/I。','',
        '## 调用与Token','',
        '| 条件 | 成功调用 | 输入Token | 输出Token | 总Token | 串行请求耗时之和/秒 |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        u=result['usage'][arm]
        lines.append(f"| {names[arm]} | {u['calls']} | {u['prompt_tokens']} | {u['completion_tokens']} | {u['total_tokens']} | {u['serial_wall_seconds']:.2f} |")
    lines+=['',f"本轮新增成功响应Token合计{sum(u['total_tokens'] for u in result['usage'].values())}；未知请求用量不记零。初始弱报告是三臂共享的既有输入，仍属于部署成本，未重复计入这次新API采集量。", 
        '输出上限相同不代表总预算相同；完整上下文的额外输入全部计入。请求耗时之和不是并行端到端延迟，不能据此宣称某部署方案更快。','',
        '## 输入质量与输出质量','',
        '初始36个报告槽位中实际可用字段数量：'+json.dumps(result['initial_report_coverage'],ensure_ascii=False)+'。缺失字段显式标记；E为模型自述，未经外部事实核验。']
    for arm in ARMS:
        u=result['usage'][arm]
        lines.append(f"{names[arm]}：合法C {u['usable_C']}/{u['calls']}，完整AEIC {u['complete_AEIC']}/{u['calls']}；实际返回模型{', '.join(u['actual_models'])}。")
    lines+=['','## 后续口径','',
        '这轮只检验当前已有报告的上下文价值。初始A/E/I不完整，因此负结果不能外推为所有完整、高质量AEIC均无效；但不能因此无条件追加字段、支付或实验。正结果也不等于机制理性或均衡成立。',
        '不按结果改题、改奖励或选择更有利的主比较；不自动启动300题。先保留执行模型变化与分歧门控的独立证据，只有上下文增量值得继续时，才锁定来源隔离验证。论文、专利、系统与PPT未修改。','',
        '结果入口：`experiments/results/resource_allocation/aeic_context_v1/result.json`；原始协议、输入和调用日志同目录。']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    result=analyze()
    tests=subprocess.run([sys.executable,'-m','unittest','experiments.tests.test_aeic_context_probe',
        'experiments.tests.test_low_cost_router','experiments.tests.test_resource_allocation_probe',
        'experiments.tests.test_upgrade_pilot','experiments.tests.test_upgrade_v3'],cwd=p.ROOT,text=True,capture_output=True)
    count=re.search(r'Ran (\d+) tests',tests.stderr)
    if tests.returncode or not count:raise ValueError('tests failed: '+tests.stdout+tests.stderr)
    diff=subprocess.run(['git','diff','--check'],cwd=p.ROOT,text=True,capture_output=True)
    if diff.returncode:raise ValueError('diff check failed: '+diff.stdout+diff.stderr)
    path=p.ROOT/'docs/research/AEIC争议审阅三臂结果_2026-10-22.md'
    content=render(result)
    if path.exists():
        if path.read_text()!=content:raise ValueError('existing report mismatch')
    else:
        with path.open('x',encoding='utf-8') as stream:stream.write(content)
    files=[*OUT.glob('*.json'),*OUT.glob('*.jsonl'),path,
        p.ROOT/'experiments/mechanism/probe_aeic_context.py',
        p.ROOT/'experiments/mechanism/report_aeic_context.py',
        p.ROOT/'experiments/tests/test_aeic_context_probe.py',
        p.ROOT/'docs/research/AEIC争议审阅三臂验证_2026-10-22.md']
    freeze(OUT/'acceptance.json',{'status':'OPENED_DATA_CONTEXT_DIAGNOSTIC_VERIFIED_NOT_INDEPENDENT_CONFIRMATION',
        'tests_passed':int(count[1]),'git_diff_check_passed':True,'old_frozen_files_verified':len(frozen_hashes()),
        'new_attempts':result['attempts'],'successful_responses':result['responses'],
        'unknown_requests':result['unknown_requests'],'api_calls_during_verification':0,
        'snapshot_sha256':{str(f.relative_to(p.ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in files if f.name!='acceptance.json'}})
    print(json.dumps({'report':str(path),'tests':int(count[1]),'new_attempts':result['attempts']}))
