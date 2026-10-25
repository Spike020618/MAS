"""Generate a readable dossier from saved outputs; no API/model dependencies."""
import json
from pathlib import Path
from experiments.mechanism.audit_interface_verdicts import ROOT, OUT

NOTES = {
    ('StrategyQA', '143'): '目录标签下的真实判断变化。普通结构强调浅水、扰动等不利条件；AEIC强调低盐、低氧等保存条件。并无外部来源核验，不能把生成的E字段视为真实新增证据；木材蛀虫与钢制船体的关联也未得到验证。',
    ('StrategyQA', '214'): '判断口径变化：普通结构按直接生态位竞争作答，AEIC扩大到间接植物资源竞争。这不是已确认的知识补全，需冻结直接/间接关系判定标准后再验证。',
    ('StrategyQA', '233'): '由能力对比推测嫉妒，转为要求角色动机依据。AEIC的角色性格描述同样未核验；自由文本本来也答对，不能认定四字段独有作用。',
    ('TruthfulQA', '260'): '解析伪纠正。普通结构answer以Yes开头，后接“No, vampires are not real”；旧解析取最后一个No，误判。普通结构与AEIC的明确结论相同。',
    ('TruthfulQA', '29'): '解析伪纠正。普通结构answer以Yes开头，解释包含“no one language”；旧解析把解释中的no当结论。两种结构均接受给定答案。',
    ('TruthfulQA', '33'): '真实退化。普通结构拒绝无地域限定的笼统判断，AEIC自行加入美国语境并据此接受。此处仅描述输出差异，不核定法律事实。风险是假设字段引入题面未给的条件。',
    ('TruthfulQA', '85'): '真实退化。普通结构接受填补“Elon what”的姓名余部；AEIC要求包含Elon的完整姓名，改变答案完整性标准。自由文本也发生同类错误。',
}


def main():
    a = json.loads((OUT / 'audit_v2.json').read_text())
    lines = ['# 报告接口差异：七例原始输出核查', '',
             '本文件逐字保留原始回答。标签仅用于离线评分；案例解释是事后诊断，不是独立事实核验或因果识别。', '']
    for case in a['original_generic_AEIC_discordances']:
        d, qid, seed = case['key']
        lines += [f'## {d} / {qid}', '', case['question'], '',
                  f"缓存标签：{case['gold']}；seed记录：{seed}（不表示API随机种子已设置）。", '', NOTES[(d, qid)], '']
        for p, r in case['records'].items():
            lines += [f'### {p}', '',
                      f"旧解析：{r['observation']['stance']}；明确结论：{r['audit']['stance']}；输入/输出Token：{r['token_input']}/{r['token_output']}。", '',
                      '```text', r['raw_response'], '```', '']
    target = ROOT / 'docs/research/报告接口七例逐条核查_2026-10-24.md'
    payload = '\n'.join(lines)
    if target.exists() and target.read_text() != payload:
        raise ValueError('refusing incompatible dossier overwrite')
    if not target.exists():
        target.write_text(payload)


if __name__ == '__main__':
    main()
