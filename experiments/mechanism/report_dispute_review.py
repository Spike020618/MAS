"""Reproducible offline report for the sealed-answer review experiment."""
import hashlib
import json
import re
import subprocess
import sys

from experiments.mechanism.probe_dispute_review import OUT,ARMS,analyze
from experiments.mechanism import probe_resource_allocation as p
from experiments.mechanism.run_upgrade_pilot import freeze
from experiments.mechanism.complete_legacy_gain_audit import frozen_hashes


def fmt(value):return str(value[0]) if value[0]==value[1] else f'{value[0]}—{value[1]}'


def render(result):
    names={'stop':'封存独立判断后停止','ordinary':'普通复查','targeted':'针对性争议审阅'}
    lines=['# 独立判断后争议核查：小验证结果','',
        '## 对照与范围','',
        '同12道既有开卷题，使用上一轮独立臂的36份真实报告作为共享初始状态；停止臂不新增调用，两复查臂分别在相同初始状态上执行。固定主分析7道有效分歧题，次分析5道含异常题。不是独立测试、事实核验效果或博弈验收。',
        '两复查臂同模型、角色、系统指令、温度、600输出上限和审阅结构；针对性臂仅多看相反C报告里的固定优先级原文卡片。输入长度差异完全计入。整批完成后才评分，无格式补问或自动重试。','',
        f"本轮{result['new_attempts']}次新请求，{result['responses']}份响应，{result['unknown']}次未知传输。初始36次调用是此前真实发生的共享成本，未重复计入新采集量，也不是部署免费成本。",'',
        '| 条件 | 有效分歧7题 | 含异常5题 | 全部12题 |','|---|---:|---:|---:|']
    for arm in ['stop',*ARMS]:
        counts=[fmt(result['groups'][g]['correct_bounds'][arm]) for g in ['valid_disagreement','invalid_or_abstain','all']]
        lines.append(f'| {names[arm]} | {counts[0]}/7 | {counts[1]}/5 | {counts[2]}/12 |')
    lines+=['','正确数范围若存在，来自未知响应可能答案，不是置信区间。合法C不因缺少或无效审阅字段而自动判错。','',
        '## 纠正与伤害','']
    for group,title in [('valid_disagreement','主分析7题'),('all','全部12题')]:
        lines.append(title+'：')
        for pair,values in result['groups'][group]['paired'].items():
            lines.append(f"- {pair}："+(f"纠正{values['corrections']}、伤害{values['harms']}。" if 'corrections' in values else '存在未知结果，不能给出唯一计数。'))
    primary=result['groups']['valid_disagreement']['correct_bounds']
    if primary['targeted'][0]>primary['ordinary'][1]:
        conclusion='主分析针对性臂在本次抽样中数值优于普通复查，保留为小样本候选信号；不能宣布独立贡献已验证。'
    elif primary['targeted'][1]<primary['ordinary'][0]:
        conclusion='主分析针对性臂低于普通复查，本轮不支持增加争议卡。不能用结构更复杂或审阅记录更多替代准确率证据。'
    else:
        conclusion='主分析未观察到针对性臂超出普通复查的明确正确数增量，不将共同的复查收益归因于争议卡或AEIC。'
    lines+=['',conclusion,'',
        '逐题纠正/伤害必须与汇总一起读：总正确数相同并不证明逐题等价，不显著也不能证明等效。单次抽样、小样本和已开卷条件下不作确认性统计推断。','',
        '## 真实开销与审阅记录','',
        '| 条件 | 成功调用 | 输入Token | 输出Token | 总Token | 声明式审阅记录合规 | C改变 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        u=result['usage'][arm]
        lines.append(f"| {names[arm]} | {u['calls']} | {u['prompt_tokens']} | {u['completion_tokens']} | {u['total_tokens']} | {u['valid_audit']}/{u['calls']} | {u['changed_C']} |")
    prior=json.loads((p.ROOT/'experiments/results/resource_allocation/aeic_context_v1/result.json').read_text())
    initial_tokens=prior['usage']['independent']['total_tokens']
    lines+=['',f'计入共同初始执行成本：停止臂36次调用、{initial_tokens} Token；'+
        '；'.join(f"{names[a]}72次调用、{initial_tokens+result['usage'][a]['total_tokens']} Token" for a in ARMS)+'。这是相应保存执行路径的Token记账，不是货币计费或独立部署端到端耗时。']
    lines+=['',f"本轮新增成功响应Token合计{sum(u['total_tokens'] for u in result['usage'].values())}。所有卡片共{result['card_count']}条；{result['empty_card_seats']}/36个角色槽位没有可提取的反方原文，不补造卡片。", 
        '审阅合规只检查声明结构、来源引用和未外部核验的标记，不证明检查真正有效、推理真实或声明的变更依据正确。C改变按角色计数，不能当作独立题目数。停止臂输出来自真实封存初始报告，不能当作零成本执行基线。']
    for arm in ARMS:
        u=result['usage'][arm]
        lines.append(f"{names[arm]}：返回模型{', '.join(u['models'])}；成功请求串行耗时之和{u['wall_seconds']:.2f}秒；改变C但未声明变更依据{u['changed_without_declared_basis']}次。请求时间之和不是并行端到端耗时。")
    lines+=['','## 研究口径','',
        '该实现改变的是先后顺序与信息组织，不是证明了AEIC四字段必要。卡片只是相反结论报告中的原文，并非已经定位到有效事实争议；输入来源仍是未经核验的模型输出。',
        '若针对性审阅没有增量，按预设停止本轮扩展，不根据个别题目重排选句优先级、不改奖励追效果。若出现正向数值差，也只能锁定候选后进行来源题干隔离验证；不能拼接旧450题效果与其他游戏定理。',
        '原样AEIC共享的上一轮负结果保留，不能因当前协议名称变化而覆盖。论文、专利、系统与PPT未修改。','',
        '原始请求、响应、初始状态及卡片：`experiments/results/resource_allocation/dispute_review_v1/`。']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    result=analyze()
    modules=['test_dispute_review','test_aeic_context_probe','test_low_cost_router',
             'test_resource_allocation_probe','test_upgrade_pilot','test_upgrade_v3']
    tests=subprocess.run([sys.executable,'-m','unittest',*['experiments.tests.'+m for m in modules]],
        cwd=p.ROOT,text=True,capture_output=True)
    count=re.search(r'Ran (\d+) tests',tests.stderr)
    if tests.returncode or not count:raise ValueError('tests failed: '+tests.stdout+tests.stderr)
    diff=subprocess.run(['git','diff','--check'],cwd=p.ROOT,text=True,capture_output=True)
    if diff.returncode:raise ValueError('diff check failed: '+diff.stdout+diff.stderr)
    path=p.ROOT/'docs/research/独立判断后争议核查_结果_2026-10-22.md';content=render(result)
    if path.exists():
        if path.read_text()!=content:raise ValueError('existing result mismatch')
    else:
        with path.open('x',encoding='utf-8') as stream:stream.write(content)
    files=[*OUT.glob('*.json'),*OUT.glob('*.jsonl'),path,
        p.ROOT/'experiments/mechanism/probe_dispute_review.py',
        p.ROOT/'experiments/mechanism/report_dispute_review.py',
        p.ROOT/'experiments/tests/test_dispute_review.py',
        p.ROOT/'docs/research/独立判断后争议核查_方案_2026-10-22.md']
    freeze(OUT/'acceptance.json',{'status':'OPENED_DATA_REVIEW_DIAGNOSTIC_VERIFIED_NOT_RESEARCH_ACCEPTANCE',
        'tests_passed':int(count[1]),'old_frozen_files_verified':len(frozen_hashes()),
        'git_diff_check_passed':True,'new_attempts':result['new_attempts'],
        'successful_responses':result['responses'],'unknown':result['unknown'],
        'api_calls_during_verification':0,
        'snapshot_sha256':{str(f.relative_to(p.ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in files if f.name!='acceptance.json'}})
    print(json.dumps({'report':str(path),'tests':int(count[1]),'new_attempts':result['new_attempts']}))
