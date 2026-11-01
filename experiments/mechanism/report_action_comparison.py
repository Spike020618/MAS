"""Render scored measurements; never fabricate causal adjudication."""
from experiments.mechanism import run_action_comparison as r


def main():
    result = r.read(r.OUT / 'analysis.json')
    names = {'direct': '初始回答', 'retrieve': '追加取证', 'check': '推导检查',
             'review': '普通审阅', 'resample': '重新求解'}
    lines = ['# 三行动配对小实验：实测结果', '',
        '24道新题，开发/确认各12；相同固定DeepSeek后端，初始及最终求解均为thinking配置。不是已训练路由器或博弈机制。', '',
        '重要限制：事后发现C的11份none附引用报告、V的10份额外自评字段报告被丢弃；提示与校验约定存在不一致。以下是v1按既定规则实际运行的结果，不是修复后方法的效果，也不是推导检查普遍无效的干净证据。全部4个EM变化案例均属表达变化；详见《案例审查与接口修复.md》。', '',
        f"整批实际{result['physical_calls']}次请求、{result['physical_tokens']:,} Token；按高峰未命中输入与输出费率计的保守费用代理为{result['usd_proxy']:.6f}美元，不是实际账单。", '',
        '路径消耗包含共同初答与该路径必要行动、最终回答，完全相同请求复用后按实际源请求去重。共享初答不可把各路径消耗相加当物理总消耗；等调用不等于等Token。', '',
        '只有确认组C对V、C对S两项是主检验；开发组和全体结果是描述分析，不从三个划分中择优当作主结果。', '']
    for split, title in (('development', '开发12题'), ('confirmation', '确认12题'), ('all', '全部24题')):
        cell = result['cells'][split]
        lines += ['## ' + title, '', '|路径|EM正确|F1|纠正/误改|无效最终答案|Token|调用|HTTP累加秒|', '|---|---:|---:|---:|---:|---:|---:|---:|']
        for arm, values in cell['arms'].items():
            lines.append(f"|{names[arm]}|{values['correct']:.0f}/{values['n']}|{values['mean_f1']:.4f}|{len(values['corrections'])}/{len(values['harms'])}|{values['invalid']}|{values['tokens']:,}|{values['calls']}|{values['http_seconds']:.2f}|")
        lines += ['', '主比较（问题级配对）：']
        for pair in cell['primary_pairs']:
            lines.append(f"- 推导检查对{names[pair['control']]}：检查独赢{pair['check_wins']}题，对照独赢{pair['control_wins']}题；Exact McNemar p={pair['p']:.6g}，Holm p={pair['holm_p']:.6g}。")
        lines += ['', '相对初答、且另一行动未纠正的独有纠正：',
            '- 取证：' + (', '.join(cell['exclusive_corrections']['retrieve']) or '无'),
            '- 检查：' + (', '.join(cell['exclusive_corrections']['check']) or '无'), '']
    lines += ['## 预定筛查', '',
        '数值门槛：' + ('通过，但还须事实案例核查，不能自动扩样或训练。' if result['numeric_screen_pass'] else '未通过，不进入路由器或博弈训练，不在本批上修改提示后重新验收。'), '',
        '## 变化案例索引', '',
        '以下只按官方EM列出，不能自动称为事实纠正；原始解释、引句和资料必须回查。', '']
    for arm in ('retrieve', 'check', 'review', 'resample'):
        cell = result['cells']['all']['arms'][arm]
        for label in ('corrections', 'harms'):
            for cid in cell[label]:
                before, after = result['per_case'][cid]['direct'], result['per_case'][cid][arm]
                lines.append(f"- {names[arm]} {label} `{cid}`：{before['answer']} → {after['answer']}。")
    lines += ['', '原始材料：`experiments/results/action_comparison_v1/`。执行时间是HTTP与检索分项观测，不把断点运行时间冒充新鲜在线端到端延迟。', '',
        '本报告由已封存的analysis.json生成；因果解释和语义案例审查另记，不回写标签或修改已冻结请求。']
    target = r.ROOT / 'docs/research/action_targets_v1/实测结果.md'
    target.write_text('\n'.join(lines) + '\n')
    print(target)


if __name__ == '__main__':
    main()
