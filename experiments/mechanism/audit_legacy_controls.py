"""Audit what historical controls actually changed; no redefinition of outcomes."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/results/audit/legacy_gain_attribution_2026-10-22'


def summarize(rows):
    turns = [t for r in rows for t in r['rounds']]
    routes = Counter()
    mask = (rows[0]['condition'].split('__signals@')[-1] if '__signals@' in rows[0]['condition'] else None)
    for t in turns:
        for call in t['action_trace']:
            if call['stage'] != 'choose' and not call['stage'].startswith('format_repair'):
                routes[(t['chosen_action'], call.get('backend'), call.get('model'))] += 1
    return {'records': len(rows), 'source_ids': len({(r['dataset'], str(r['question_id'])) for r in rows}),
        'correct': sum(r['final_decision'] == r['ground_truth'] for r in rows),
        'actions': dict(Counter(t['chosen_action'] for t in turns)),
        'execution_calls_by_action_backend_model': [{'action': a, 'backend': b, 'model': m, 'calls': n}
                                                  for (a,b,m), n in sorted(routes.items(), key=lambda x: str(x[0]))],
        'complete_AEIC_reports': sum(all(t['report'].get(k) for k in ['A','E','I','C']) for t in turns),
        'report_key_sets': sorted({','.join(sorted(t['report'])) for t in turns}),
        'unrewarded_field_present_counts': ({k:sum(bool(t['report'].get(k)) for t in turns)
                                            for k in 'AEIC' if k not in mask} if mask else {}),
        'agent_rounds': len(turns),
        'tokens': sum(t['token_input']+t['token_output'] for t in turns),
        'calls': sum(len(t['action_trace']) for t in turns)}


def main():
    source_files, suites = {}, {}
    frozen = ROOT / 'experiments/results/mechanism/frozen'
    for folder in sorted(frozen.iterdir()):
        if not folder.is_dir() or folder.name.endswith('.recomputed'):
            continue
        path = folder / 'question_level.jsonl'
        if not path.exists():
            continue
        source_files[str(path.relative_to(ROOT))] = sha256(path.read_bytes()).hexdigest()
        groups = defaultdict(list)
        for line in path.read_text().splitlines():
            r = json.loads(line)
            # Seed kept distinct: not independent new questions.
            groups[(r['condition'], r['seed'])].append(r)
        suites[folder.name] = [{'condition': c, 'seed': seed, **summarize(rs)} for (c,seed), rs in sorted(groups.items())]
    # Earlier homogeneous protocol is historical development, not a randomized
    # change of backend within formal_v10. No pooling / causal contrast.
    pilot_path = ROOT / 'experiments/results/mechanism/pilot/pilot_v6_twostage_dev10.jsonl'
    groups = defaultdict(dict)
    failed = []
    for line in pilot_path.read_text().splitlines():
        r = json.loads(line)
        key = (r['dataset'], str(r['question_id']))
        if r.get('completed') and not r.get('error'):
            if key in groups[r['condition']]:
                raise ValueError('duplicate valid pilot key')
            groups[r['condition']][key] = r
        else:
            failed.append({'condition': r['condition'], 'dataset': r['dataset'], 'id': r['question_id']})
    common = set.intersection(*(set(v) for v in groups.values()))
    pilot = {c: summarize([rs[k] for k in sorted(common)]) for c,rs in groups.items()}
    source_files[str(pilot_path.relative_to(ROOT))] = sha256(pilot_path.read_bytes()).hexdigest()
    glm_aeic = [call for group in suites['heldout_glm_v11_dev']
                for call in group['execution_calls_by_action_backend_model'] if call['action']=='aeic_effort']
    signal_schemas = [group['report_key_sets'] for group in suites['signal_ablation_v11_dev']]
    main_matrix = json.loads((frozen/'formal_v10/matrix.json').read_text())
    result = {'no_api_calls': True, 'frozen_suites': suites,
        'homogeneous_v6_complete_case_development_only': {'conditions': pilot, 'n_paired': len(common),
            'failed_records': failed, 'scope': 'different historical sample/protocol, not a backend-only ablation'},
        'findings_to_verify_against_source': {
            'heldout_glm_AEIC_execution_observed_only_glm': bool(glm_aeic) and all(c['backend']=='zhipu_flash' for c in glm_aeic),
            'signal_conditions_all_keep_AEIC_report_key_set': all(x==['A,C,E,I'] for x in signal_schemas),
            'mainline_configured_independent': main_matrix['defaults']['behavior']=='independent'},
        'source_hashes': source_files,
        'limits': ['raw historical request bodies absent: current source plus metadata do not establish byte-identical past prompts',
                   'no suite is pooled with formal_v10 as additional independent subjects',
                   'homogeneous v6 transport traces omit executor identity; consult protocol, not assumed provenance']}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'controls.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'out': str(OUT/'controls.json'), 'v6_paired': len(common),
        'v6_correct': {c:v['correct'] for c,v in pilot.items()},
        'suites': {s:len(g) for s,g in suites.items()}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
