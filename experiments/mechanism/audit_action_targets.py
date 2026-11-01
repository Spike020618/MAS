"""Offline, descriptive review of frozen trajectories; never a runtime classifier.

The manually reviewed categories below are hypotheses, not ground-truth causes.
Gold answers/supporting facts enter only this offline module, never action inputs.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'experiments/results/sequential_allocation_v1'
DEST = ROOT / 'docs/research/action_targets_v1'

# Stable IDs, not dataset-specific rules in an agent prompt.
REVIEW = {
    '5a7d6a2d5542991319bc93c6': ('expression', 'expression', '资料及回答均明确包含Southern Isles；额外Norse名称导致严格EM不匹配。'),
    '5ae69e6d5542996d980e7c62': ('missing_evidence', 'missing_evidence', '原生资料未提供Flaw及2004第二张专辑关系；两次查询持续锁定Nerves Junior，未补齐关系。'),
    '5abc19705542993a06baf86e': ('missing_evidence', None, '非thinking初始弃答；第二次取证加入Carice van Houten国籍资料后答Dutch，但gold支持人物为Halina Reijn，不能称目标人物链条已完整核验。'),
    '5a7261635542997f8278398a': ('missing_evidence', 'expression', '原生资料只给出生地Utena；新增Utena资料补上区域。thinking初始northeastern与gold同区域，但严格EM不匹配。'),
    '5abbd4035542993f40c73bcb': ('missing_evidence', None, '原生歌曲资料给出作者，未给入选名人堂的关系；查询停留在歌曲/纪录片，未找到Kevin Richardson入选事实。'),
    '5a7bd25a554299294a54ab13': ('missing_evidence', None, '原生资料缺两人寿命；S1只补Sherman，S2补Kaplan。thinking初始答案虽正确，却声称无来源支持的1947–2022寿命；EM正确不等于推导可信。'),
    '5ae3d67f5542992f92d8238f': ('missing_evidence', 'uncertain', '资料只确认Dooley身份；未给前任教学院。模型提及Tennessee和Louisiana Tech，题干used to并未唯一限定，保留歧义，不硬判为推导失败。'),
    '5a8e942b5542995085b3745c': ('missing_evidence', 'missing_evidence', '模型根据EDM相似性猜Weekend Festival；缺少gold所需Tropical house到Tomorrowland关系，不能以一般类型相似代替证据。'),
    '5ab1e3bb554299340b52541a': ('missing_evidence', 'missing_evidence', '当前资料能支持Political hip hop，但未给更上位的Underground hip hop关系；不能直接把漏掉第二跳叫作已有证据推导失败。'),
    '5ab9be4f554299753720f843': ('uncertain', 'uncertain', '两段均称mountain，但题干land elevation可被理解为海拔；回答海拔而gold要地貌类型，是任务解释候选，不是确定的逻辑错误。'),
    '5ac4920d5542996feb3fe8d3': ('missing_evidence', 'expression', 'S0缺Bob Gurr资料；S1已明确提供Autopia与生日，非thinking仍称未提Autopia，是新增资料后的可定位阅读矛盾。thinking生日正确，仅少born。'),
    '5a7905f555429974737f7947': ('missing_evidence', None, 'S0缺出生地，非thinking猜Casablanca；thinking给对second largest却错称出生于Fez。S1补Rabat，S2仍缺Rabat排名；按历史gold评分，不用当前城市排名改标签。'),
    '5a7cd0fd55429909bec76823': ('missing_evidence', None, '非thinking缺Switchfoot存续资料而弃答；新增Switchfoot资料后回答。'),
    '5a83aaeb5542996488c2e483': ('uncertain', 'uncertain', '原生Graduation确有Roc-A-Fella与Dwele，Power又连接gold专辑；题干约束可能不唯一，不能把选择Graduation简单称推导错误。'),
    '5a7c9d2e55429935c91b5261': (None, 'expression', 'Wendigo与Wendigo (Windigo)为同一legend的括号别名差异。'),
    '5ae56a3f5542992663a4f1f4': ('missing_evidence', None, 'S0未给该保护区或动物，S1新增保护区→Bonobo→pygmy chimpanzee的完整可追溯链。'),
    '5ae0b82855429906c02daaff': ('missing_evidence', 'expression', '非thinking未得片名；thinking原答已含Hjernevask，后续仅去括号。新增资料未给该节目名，不算事实纠正。'),
    '5a89706055429951533612f9': ('expression', 'expression', 'St. Louis County与Saint Louis County为缩写差异；保留官方EM，不计成事实错误。'),
    '5abb291355429939ce03dd9e': ('uncertain', 'uncertain', 'Moor Park资料解释Manor of More的旧园址，未明确等同The More；实体/历史命名关系有疑问，保留未知。'),
    '5ac2381355429951e9e684a9': ('missing_evidence', None, '非thinking初始缺Colorado球队所属联盟，新增球队相关资料后命中；thinking后续Pac-12省略Conference导致EM下降，不是明确事实伤害。'),
    '5abd0d635542992ac4f381b8': ('missing_evidence', 'missing_evidence', '原生资料缺Adele和销量纪录；查询被Zac Brown Band锚定。thinking的70 days无来源支持，S1/S2又触发输出截断，分开记录。'),
    '5ae1dac95542997283cd229b': ('expression', 'expression', '原答已含SEAL Team Six，多出的DEVGRU括号别名造成EM不匹配。'),
}
LABELS = {'none': 'EM命中（非推导正确证明）', 'expression': '表达差异',
          'missing_evidence': '关键关系缺失', 'uncertain': '无法确定/题意歧义'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def support_presence(source, documents):
    """Exact annotated sentences are structural diagnostics, not entailment checks."""
    context = dict(zip(source['context']['title'], source['context']['sentences']))
    facts = []
    for title, index in zip(source['supporting_facts']['title'], source['supporting_facts']['sent_id']):
        sentences = context.get(title, [])
        text = sentences[index] if 0 <= index < len(sentences) else None
        hits = [d['docid'] for d in documents if text and text in d['text']]
        facts.append({'title': title, 'sent_id': index, 'text': text, 'matching_document_ids': hits})
    return facts


def review_category(cid, mode, score):
    if score['em']:
        return 'none'
    if cid not in REVIEW:
        raise ValueError('Unreviewed S0 mismatch: ' + cid)
    category = REVIEW[cid][0 if mode == 'disabled' else 1]
    if category is None:
        raise ValueError('Review disagrees with S0 EM: ' + cid)
    return category


def build(cases, outputs, labels, splits, original, scorer):
    result = []
    for case in cases:
        cid = case['id']
        stages = outputs[cid]
        row = {'id': cid, 'question': case['question'], 'gold_offline_only': labels[cid],
               'former_split_now_retired': splits[cid], 'modes': {},
               'review_note': REVIEW.get(cid, (None, None, '本次未出现S0严格EM不匹配；不据此认可解释或引文。'))[2],
               'source_trace': str((SOURCE / cid / 'sealed_outputs.json').relative_to(ROOT))}
        for mode in ('disabled', 'enabled'):
            items = []
            for state in stages:
                parsed = state['answers'][mode]['parsed']
                items.append({'stage': state['stage'], 'answer': parsed['answer'],
                              'valid': parsed['valid'], 'score': scorer(parsed['answer'], labels[cid]),
                              'reasoning': parsed.get('reasoning', ''), 'citations': parsed.get('docids', []),
                              'source_key': state['answers'][mode]['source_key'],
                              'annotated_support_exact_presence': support_presence(original[cid], state['documents'])})
            row['modes'][mode] = {'s0_review': review_category(cid, mode, items[0]['score']), 'stages': items}
        result.append(row)
    return result


def render(rows, summary):
    lines = ['# 已有40题轨迹：行动目标核查', '',
        '这是已查看标签后的离线、单一分析者归因；不是独立人工标注、因果证明或在线错误检测器。原40题全部转为开发审计材料，不再作为新方法的独立确认集。', '',
        '## S0严格EM不匹配的归因', '',
        '|配置|关键关系缺失|表达差异|无法确定/歧义|合计|', '|---|---:|---:|---:|---:|']
    for mode, values in summary.items():
        lines.append(f"|{mode}|{values['missing_evidence']}|{values['expression']}|{values['uncertain']}|{sum(v for k,v in values.items() if k != 'none')}|")
    lines += ['', '表达差异不改变官方EM，也不被重新计成主实验正确数；关键关系缺失是可见资料诊断，不证明检索一定纠正；支持句精确缺失也不证明其他文档无法表达相同事实。', '',
        '## 每题轨迹', '', '|ID|问题|gold（离线）|非thinking S0→S1→S2 / S0归因|thinking S0→S1→S2 / S0归因|核查说明|', '|---|---|---|---|---|---|']
    def clean(text):
        return str(text).replace('|', '\\|').replace('\n', ' ')
    for row in rows:
        cells = [row['id'], row['question'], row['gold_offline_only']]
        for mode in ('disabled', 'enabled'):
            m = row['modes'][mode]
            cells.append(' → '.join(s['answer'] for s in m['stages']) + ' / ' + LABELS[m['s0_review']])
        cells.append(row['review_note'])
        lines.append('|' + '|'.join(clean(c) for c in cells) + '|')
    lines += ['', '## 需要保留的反例', '',
        '- `5ac4920d5542996feb3fe8d3`：S1 Bob Gurr原文明确包含Autopia与生日，非thinking却说没有Autopia。是值得开发推导检查的阅读矛盾，但尚未执行检查，更不是独立验证成功。S0缺证据，不能把S1问题倒算为S0推导错误。',
        '- `5ab9be4f554299753720f843`：mountain/海拔的题意分歧；只作为解释歧义，不拿来证明逻辑检查能纠错。',
        '- `5a83aaeb5542996488c2e483`：Graduation有资料支持，gold是另一专辑；不把单一gold当成所有其他候选错误的证明。',
        '- `5a7bd25a554299294a54ab13`、`5a7905f555429974737f7947`：初始thinking可以EM正确同时给出错误或无来源支持的中间事实。自述解释不是内部推理记录，也不是事实核验器。',
        '- `5abd0d635542992ac4f381b8`：缺证据与后续两次输出截断同时存在，不能归成单一错误原因。', '',
        '## 机器材料与限制', '',
        '`轨迹归因.json`保存80条配置轨迹、240个阶段答案/解释、引用ID及官方支持句的逐句原文匹配情况。支持标注仅供离线核查，不进入查询或求解输入。人工判断写在脚本REVIEW中，属于可修改的开发假设；不对其计算标注一致率。', '',
        '本次没有训练路由器，没有新的API调用，没有改写原实验评分、原始轨迹或旧结论。']
    return '\n'.join(lines) + '\n'


def observations(outputs):
    cid = '5ac4920d5542996feb3fe8d3'
    state = outputs[cid][1]
    doc = next(d for d in state['documents'] if d['title'] == 'Bob Gurr')
    passage = 'including Autopia, Haunted Mansion'
    assert passage in doc['text']
    explanation = state['answers']['disabled']['parsed']['reasoning']
    assert 'does not mention Autopia' in explanation
    invalid = [{'id': key, 'mode': mode, 'stage': s['stage'],
                'source_key': s['answers'][mode]['source_key']}
               for key, states in outputs.items() for s in states for mode in ('disabled', 'enabled')
               if not s['answers'][mode]['parsed']['valid']]
    return {'reading_mismatch_development_candidate': {
        'id': cid, 'stage': 1, 'mode': 'disabled', 'document_id': doc['docid'],
        'literal_passage': passage, 'full_source_text': doc['text'],
        'observed_explanation': explanation, 'check_action_executed': False,
        'not_an_s0_reasoning_error': True},
        'invalid_stage_outputs': invalid,
        'interpretation': 'Execution flags and posthoc hypotheses; not independent causal labels.'}


def main():
    from experiments.mechanism import run_sequential_allocation_screen as r
    r.verify()
    paths = sorted(p for p in SOURCE.rglob('*') if p.is_file())
    before = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    cases, outputs, labels, splits = [read(SOURCE / name) for name in
        ('public_cases.json', 'sealed_outputs.json', 'private_labels.json', 'splits.json')]
    original = {c['id']: c for c in r.p.data.parquet_module().read_table(r.p.data.RAW / 'dev_fullwiki.parquet').to_pylist() if c['id'] in outputs}
    rows = build(cases, outputs, labels, splits, original, r.p.answer_score)
    summary = {m: dict(Counter(row['modes'][m]['s0_review'] for row in rows)) for m in r.MODES}
    assert len(rows) == 40 and sum(len(row['modes']) for row in rows) == 80
    assert before == {str(p.relative_to(ROOT)): sha(p) for p in paths}
    DEST.mkdir(parents=True, exist_ok=True)
    # Generated analysis output, not an experiment or source rewrite.
    (DEST / '轨迹归因.json').write_text(json.dumps({'status': 'offline_development_audit',
        'new_api_calls': 0, 'n_questions': 40, 'n_mode_trajectories': 80, 's0_summary': summary,
        'source_sha256': before, 'observations': observations(outputs),
        'original_dataset_sha256': sha(r.p.data.RAW / 'dev_fullwiki.parquet'),
        'rows': rows}, ensure_ascii=False, indent=2) + '\n')
    (DEST / '轨迹归因.md').write_text(render(rows, summary))
    print(json.dumps({'n': len(rows), 'summary': summary, 'new_api_calls': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
