"""Render an evidence-linked P2 report after collection and comparison."""
import argparse
import json
from pathlib import Path

from experiments.mechanism.calibrate_upgrade_v2 import audit_stage
from experiments.mechanism.run_upgrade_pilot import read_rows


def write_report(root, result_path, model_path, output):
    result = json.loads(result_path.read_text())
    model = json.loads(model_path.read_text())
    review = json.loads((root / 'p2_robustness_review_v2.json').read_text())
    diagnostics = json.loads((root / 'p2_development_diagnostics_v2.json').read_text())
    stages = {stage: root / ('p2_' + stage + '_v2') for stage in ('calibration', 'development')}
    audits = {stage: audit_stage(path) for stage, path in stages.items()}
    phase_costs = {}
    for stage, path in stages.items():
        rows = read_rows(path / 'responses.jsonl')
        phase_costs[stage] = {}
        for phase in ('initial', 'upgrade', 'diagnostic'):
            subset = [r for r in rows if r['key'].split('/')[1] == phase]
            phase_costs[stage][phase] = {'calls': len(subset),
                'tokens': sum(json.loads(r['response']['body'])['usage']['total_tokens'] for r in subset)}
    lines = ['# 升级博弈P2：独立校准与简单策略比较', '',
             '## 验收结论', '',
             '完整三域P2：' + ('是' if result['full_P2_completed'] else '否') + '。',
             '开发资格门：' + result['effect_gate'] + '。',
             '**研究有效性核查：未建立语义增量；300题正式测试不启动。**', '',
             '数值门通过的原JSON不改写；事后核查发现，两个纠正答案在初始C中已经存在，严格字段校验将其弃答后再恢复，不能当作新信息收益。该事后诊断不是新的确认性效果检验，但足以阻止把数值门误当核心贡献成立。', '',
             '开发阶段用于选参，以下不是最终独立测试结论。合同内均衡、真实价值误差和任务表现分别判断；真实价值均衡尚未建立。', '',
             '## 估计与隔离', '',
             f"校准{result['calibration_n']}题，开发{result['development_n']}题。校准状态组样本量：" +
             '；'.join(k + '=' + str(v['n']) for k, v in model['states'].items()) + '。',
             f"开发8组合的状态代理MSE={result['state_proxy_mse']:.6f}；仅校准全局均值MSE={result['pooled_proxy_mse']:.6f}。",
             f"开发集中动作影响正确性的题数：{result['questions_with_action_dependent_correctness']}。",
             '状态仅为初始C的相等关系和弃答位置，不用领域或题面。A/E/I未被宣称具有独立预测贡献。', '',
             '## 简单策略', '', '| 策略 | 正确数 | 升级数 | 部署调用 | Token |', '|---|---:|---:|---:|---:|']
    names = ['all_K', 'all_U', 'calibration_fixed', 'calibration_rotation_k']
    for name in names:
        row = result['baselines'][name]
        lines.append(f"| {name} | {row['correct']}/{row['n']} | {row['upgrades']} | {row['deployed_calls']} | {row['tokens']} |")
    row = review['initial_C_only_baseline']
    lines.append(f"| 初始C多数（事后完整性核查） | {row['correct']}/{row['n']} | 0 | {row['deployed_calls']} | {row['tokens']} |")
    lines += ['', '固定组合与轮询k在校准阶段决定，不在开发逐题用答案选择。完整固定/轮询前沿保留在JSON中。', '',
              '## 全部费用候选', '', '| λ | 正确数 | 升级数 | 同k随机期望正确数 | 策略分布 | 开发资格门 |',
              '|---|---:|---:|---:|---|---|']
    for c in result['candidates']:
        m = c['mechanism']
        lines.append(f"| {c['cost_lambda']} | {m['correct']}/{m['n']} | {m['upgrades']} | {c['matched_k_random_expected_correct']:.2f} | {c['action_counts']} | {'通过' if c['development_signal_gate'] else '未过'} |")
    lines += ['', 'λ=0.01为实际在线开发执行；λ=0.02等其他参数为固定初始状态潜在报告回放，不伪称已分别在线执行。', '',
              '中央最优只使用同一代理表，不查询本题答案。局部均衡不保证全局最优。费用是虚拟预约单位，实际Token另计；同k随机仅是条件资源诊断。', '',
              '## 核心收益核查', '',
              'λ=0.02回放策略：49/60，24次升级，78,921 Token。初始C多数：49/60，0次升级，57,573 Token，逐题正确/错误结果完全相同。两题纠正均可由初始C恢复，无须任何新增调用。',
              '同k随机期望正确数也是49/60。简单非一致态升级两人的轮询策略：49/60、24次升级、78,893 Token。拟合模型在全部64种三人原始结论模式上的升级数量均与这一简单门控相同；这不证明未来总体上完全等价，但没有展示选择谁升级的额外价值。',
              '53道全部6份报告合法的开发题：各个费用候选均为0纠正、0退化。该分组是事后潜在输出条件诊断，不当作独立因果估计。',
              f"增量预测MSE：状态代理{diagnostics['state_increment_prediction_mse']:.6f}，全局均值{diagnostics['pooled_increment_prediction_mse']:.6f}。绝对得分MSE较低不等于已经产生有用的新信息。", '',
              '## 配对变化与区间', '']
    for c in result['candidates']:
        d = c['paired_diagnostics']['all_K']
        lo, hi = d['bootstrap_95_interval']
        lines.append(f"- λ={c['cost_lambda']}：相对全K纠正{d['corrected']}、退化{d['harmed']}；准确率差{100*d['accuracy_delta']:.2f}pp，问题级分域bootstrap区间[{100*lo:.2f}, {100*hi:.2f}]pp。")
    lines += ['', '区间为开发探索分析，不能因选参后好看而称确认性显著。若输出没有变化，不把增加Token或更多AEIC文字当作增量价值。', '',
              '## 全部采集费用', '', '| 阶段/路径 | 调用 | Token |', '|---|---:|---:|']
    for stage, phases in phase_costs.items():
        for phase, values in phases.items():
            lines.append(f"| {stage}/{phase} | {values['calls']} | {values['tokens']} |")
    lines += ['', '未选升级的诊断调用计入总采集费用，不混入部署成本。重试和工程试运行也不计入以上P2主采集量。', '',
              '## 文件与边界', '',
              '- 结果：`' + str(result_path) + '`。', '- 冻结代理：`' + str(model_path) + '`。',
              '- 原始响应、请求、用量、时间及顺序：各collection目录的responses/attempts/timeline日志。',
              '- 规则：`docs/research/升级博弈_P2预注册_2026-10-22.md`。',
              '- 单Agent强基线尚未采集，不声称已经比较。',
              '- 工程v2将无效报告显式记为失败弃答，原文保留；v1导入调用仅计费一次，不是新增重复实验。',
              '- 失败槽位数：' + str({s: a.get('invalid_report_slots', 0) for s, a in audits.items()}) + '。',
              '- 原450题属于旧协议，不并入当前结果。', '',
              '## 本轮决定', '',
              'P2采集、校准及比较完成；合同代理的局部均衡可计算可检查，但核心语义增量未成立。不启动300题，不继续调支付逼出升级。首要方法问题是字段校验不应丢弃已有合法结论，以及多数阈值下单边更新会停在一致态低努力均衡；这不是某个知识领域的特例。',
              '若再改变真实升级路径、收益目标或协调更新，须作为新结构版本另行验证；不得把本轮数值门通过追认为其有效，更不得恢复已证明无额外信息的表面收益。', '']
    if output.exists():
        if output.read_text() != '\n'.join(lines):
            raise ValueError('refuse to overwrite a changed report')
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('\n'.join(lines))
    return {'path': str(output), 'audits': audits, 'formal_300_started': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('experiments/results/upgrade_game'))
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_report(args.root, args.result, args.model, args.output)))
