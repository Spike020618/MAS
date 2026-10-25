"""Report completed paired pilot without network access or modifying old evidence."""
from collections import Counter
import json
from math import comb
from experiments.mechanism.run_premise_review import OUT, p
from experiments.mechanism.run_upgrade_pilot import freeze, read_rows


def main():
    result = json.loads((OUT/'result.json').read_text())
    rows = json.loads((OUT/'scoring_records.json').read_text())
    initial = json.loads((OUT/'sealed_initial.json').read_text())
    reviews = json.loads((OUT/'sealed_reviews.json').read_text())
    cases = {r['id']: r for r in json.loads((OUT/'public_cases.json').read_text())}
    domains = {}
    paired = Counter()
    changed = []
    for r in rows:
        d = r['id'].split(':')[0]
        cell = domains.setdefault(d, Counter())
        cell['n'] += 1
        cell['initial_correct'] += r['initial_answer'] == r['gold']
        answers = {}
        for a in ('ordinary', 'premise'):
            obs = reviews[r['id']][a]
            answers[a] = obs['answer'] if obs['valid'] else 'abstain'
            cell[a+'_correct'] += answers[a] == r['gold']
            cell[a+'_invalid'] += not obs['valid']
        oc, pc = answers['ordinary'] == r['gold'], answers['premise'] == r['gold']
        paired['corrections'] += pc and not oc
        paired['harms'] += oc and not pc
        if answers['ordinary'] != answers['premise'] or any(v != r['initial_answer'] for v in answers.values()):
            changed.append({'id': r['id'], 'question': cases[r['id']]['question'], 'gold': r['gold'],
                            'initial_answer': r['initial_answer'], 'initial_reports': initial[r['id']],
                            'reviews': reviews[r['id']]})
    b, c = paired['corrections'], paired['harms']
    n = b+c
    descriptive_p = min(1., 2*sum(comb(n, k) for k in range(min(b,c)+1))/2**n) if n else 1.
    diagnostics = {'domains': {d: dict(v) for d,v in domains.items()}, 'paired': dict(paired),
                   'exact_mcnemar_p_descriptive_only': descriptive_p,
                   'changed_cases': changed,
                   'all_attempts_with_http_status': dict(Counter(str(r['response']['http_status']) for r in read_rows(OUT/'responses.jsonl'))),
                   'initial_invalid': sum(not v['valid'] for rs in initial.values() for v in rs),
                   'limitation': 'small exploratory screen; no causal process adjudication or equilibrium claim'}
    freeze(OUT/'diagnostics.json', diagnostics)
    lines = ['# 关键前提核查：24题小批量结果', '',
             '本次不强制AEIC四字段。三个独立回答封存后，普通审阅与前提核查看到同一组输入；只改变审阅指令。', '',
             '## 实际结果', '', '| 来源 | 题数 | 初始多数 | 普通审阅 | 前提核查 |', '|---|---:|---:|---:|---:|']
    for d, v in domains.items():
        lines.append(f"| {d} | {v['n']} | {v['initial_correct']} | {v['ordinary_correct']} | {v['premise_correct']} |")
    lines += ['', f'前提核查相对普通审阅：{b}次纠正、{c}次退化。Exact McNemar描述性p={descriptive_p:.6g}；小样本不显著不能说明等效。', '',
              '## 成本与执行', '', '```json', json.dumps(result, ensure_ascii=False, indent=2), '```', '',
              'Token计入共同初始生成；serial_path_wall_seconds是逐请求耗时累加，不是并行部署用户延迟。HTTP耗时总和含已有响应的重试；未知传输结果会停机，不能按零成本处理。路径成本若有HTTP重试需另加重试账单，不用最终响应的usage冒充全量账单。', '',
              '## 结论边界', '']
    if b <= c:
        lines += ['本轮未观察到前提核查优于普通充分审阅的净纠正。不扩大到正式大样本，不追加支付包装，不按本批题目调提示词。']
    else:
        lines += ['本轮出现净纠正信号，但尚不足以证明稳定优势；须检查退化、实际成本及独立复验，不能直接进入支付或均衡结论。']
    lines += ['', '这次测试未证成新的去中心化共识机制、真实外部证据核验或博弈必要性。即使较初始多数改善，也需先与普通审阅比较，不能归因于前提协议独有价值。', '',
              '具体变化题面的三份初始报告与两份审阅原文均保存在diagnostics.json；这些是事后诊断，不作新样本。模型别名不是不可变版本，实际返回模型见result.json。未修改旧450题主实验及论文、专利、系统、PPT。']
    path = p.ROOT/'docs/research/关键前提核查_小批量结果_2026-10-24.md'
    content = '\n'.join(lines)+'\n'
    if path.exists() and path.read_text() != content:
        raise ValueError('existing report mismatch')
    if not path.exists():
        path.write_text(content)
    print(json.dumps({k:v for k,v in diagnostics.items() if k!='changed_cases'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
