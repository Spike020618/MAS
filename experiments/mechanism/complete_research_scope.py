"""Current-mainline research closeout audit, no APIs or historical rewrites.

Inventory is research-only. Hashing historical files does not validate their
scientific claims. Sequence bootstrap is a diagnostic, not a cure for feedback.
"""
from collections import Counter, defaultdict
import hashlib
import json
from math import comb
from pathlib import Path
import random
import subprocess
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'experiments/results'


def fingerprint(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def classification(relative):
    parts = relative.parts
    if parts[0] in {'movie', 'movie_acceptance', 'system_traces', 'human_review', 'audit'}:
        return None
    if parts[0] == 'mechanism':
        if 'synthesis' in parts: return 'derived_analysis_not_independent_evidence'
        if any(p.endswith('.recomputed') for p in parts): return 'recomputed_view_not_new_samples'
        if 'frozen' in parts: return 'current_mainline_frozen_subject_to_audit'
        if 'interface' in parts or 'decomposition' in parts: return 'current_explanatory_controls'
        if 'stackelberg' in parts: return 'paused_alternative_mechanisms_not_mainline'
        return 'current_lineage_or_development_not_pooled'
    if parts[0] == 'review': return 'historical_review_diagnosis_not_current_QA_method'
    if parts[0] in {'cache', 'signals'}: return 'historical_QA_sources_and_topology_not_new_method'
    if parts[0] == 'configs': return 'historical_optimization_routing_requires_split_audit'
    if parts[0] == 'archive': return 'prototype_archive_not_primary_evidence'
    if parts[0] == 'online_stackelberg': return 'historical_online_regulation_not_equilibrium_proof'
    if parts[0] == 'memory_diverse': return 'historical_memory_diversity_not_current_method'
    return 'historical_summary_or_exploration_requires_lineage'


def inventory():
    files = []
    for path in sorted(RESULTS.rglob('*')):
        if path.suffix not in {'.json', '.jsonl'} or not path.is_file(): continue
        relative = path.relative_to(RESULTS)
        group = classification(relative)
        if group is None: continue
        files.append({'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
                      'sha256': fingerprint(path), 'disposition': group})
    sizes = Counter(f['sha256'] for f in files)
    return {'scope': 'all research JSON/JSONL result paths; excludes film/system/human review/audit; not all binary snapshots',
        'files': files, 'n_files': len(files),
        'dispositions': dict(Counter(f['disposition'] for f in files)),
        'unique_byte_contents': len(sizes),
        'duplicate_byte_copies': sum(n-1 for n in sizes.values()),
        'warning': 'files and copies are not independent experiments or unique questions'}


def git_history():
    result = subprocess.run(['git', 'log', '--since=2026-06-01', '--date=short',
        '--format=%H\t%ad\t%s'], cwd=ROOT, text=True, capture_output=True, check=True)
    commits = []
    for line in result.stdout.splitlines():
        sha, date, subject = line.split('\t', 2)
        paths = subprocess.run(['git', 'show', '--format=', '--name-only', sha], cwd=ROOT,
            text=True, capture_output=True, check=True).stdout.splitlines()
        research = [p for p in paths if p.startswith(('core/', 'experiments/', 'docs/'))
                    and not p.startswith(('docs/patent/', 'docs/thesis/', 'docs/presentations/'))]
        commits.append({'commit': sha, 'date': date, 'subject': subject, 'research_paths': research})
    return {'scope': 'available git history, commit date is not proof of experiment execution time',
            'commits': commits}


def history_check(rows, ordered_ids):
    by_id = {str(r['question_id']): r for r in rows}
    if len(by_id) != len(rows): raise ValueError('duplicate sequence question')
    if set(by_id) != set(ordered_ids): raise ValueError('ordered question set differs')
    errors = Counter(); previous = {}
    for index, qid in enumerate(ordered_ids):
        row = by_id[qid]
        for rid in sorted({t['round_id'] for t in row['rounds']}):
            for turn in [t for t in row['rounds'] if t['round_id'] == rid]:
                agent = turn['agent_id']
                expected = previous.get(agent, 0.)
                if abs(turn['reputation_before']-expected) > 2e-6:
                    errors['between_question' if rid == 1 and index else 'within_or_initial'] += 1
                previous[agent] = turn['reputation_after']
    return {'questions': len(rows), 'errors': dict(errors), 'prefix_chain_matches': not errors}


def quantile(values, p):
    values = sorted(values)
    index = (len(values)-1)*p
    low = int(index); high = min(low+1, len(values)-1)
    return values[low]*(high-index)+values[high]*(index-low) if high != low else values[low]


def block_summary(sequences, length, draws=2000):
    if not sequences or type(length) is not int or length < 1 or draws < 1 or any(not seq or length > len(seq) for seq in sequences):
        raise ValueError('valid nonempty sequences and block length required')
    rng = random.Random(20261021+length)
    values = []
    for _ in range(draws):
        total = n = 0
        for seq in sequences:
            sample = []
            while len(sample) < len(seq):
                start = rng.randrange(len(seq)-length+1)
                sample.extend(seq[start:start+length])
            total += sum(sample[:len(seq)]); n += len(seq)
        values.append(total/n)
    return {'block_length': length, 'draws': draws,
        'observed_delta': sum(map(sum, sequences))/sum(map(len, sequences)),
        'pointwise_percentile_interval_95': [quantile(values, .025), quantile(values, .975)],
        'scope': 'post-hoc stratified moving-block sensitivity; not independent replication or corrected confirmatory p value'}


def pair_outcomes(pairs):
    if not pairs: raise ValueError('paired observations required')
    b = sum(not a and c for a, c in pairs)
    c = sum(a and not d for a, d in pairs)
    n = b+c
    p = min(1., 2*sum(comb(n, i) for i in range(min(b,c)+1))/2**n) if n else 1.
    return {'n_pairs': len(pairs), 'left_correct': sum(a for a, _ in pairs),
        'right_correct': sum(b for _, b in pairs), 'wrong_to_correct': b, 'correct_to_wrong': c,
        'delta': sum(int(b)-int(a) for a,b in pairs)/len(pairs),
        'descriptive_exact_mcnemar_p_unadjusted_IID_assumption': p}


def mainline():
    frozen = RESULTS/'mechanism/frozen'
    manifest = json.loads((frozen/'manifest.json').read_text())
    splits = json.loads((ROOT/'experiments/mechanism/splits.json').read_text())['datasets']
    checks = {}; bundles = []; unique_questions = set(); suite_ids = {}
    label_maps = {dataset: {str(q['question_id']): q['ground_truth']
        for q in json.loads((ROOT/info['source']).read_text())['questions']}
        for dataset,info in splits.items()}
    from experiments.core.trust import extract_stance
    record_checks = Counter(); tie_counts = Counter()
    for name in manifest['required_suites']:
        path = frozen/name/'question_level.jsonl'
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        expected = next(b for b in manifest['bundles'] if b['name'] == name)
        keys = [(r['experiment_id'], r['dataset'], r['condition'], r['question_id'], r['seed']) for r in rows]
        ids = {(r['dataset'], str(r['question_id'])) for r in rows}
        unique_questions.update(ids); suite_ids[name] = ids
        matrix = json.loads((frozen/name/'matrix.json').read_text())
        provenance = json.loads((frozen/name/'provenance.json').read_text())
        bad_files = [filename for filename, expected_hash in provenance['files'].items()
                     if fingerprint(frozen/name/filename) != expected_hash]
        record_checks['frozen_files_checked'] += len(provenance['files'])
        record_checks['frozen_file_hash_mismatches'] += len(bad_files)
        for row in rows:
            record_checks['label_mismatches'] += row['ground_truth'] != label_maps[row['dataset']][str(row['question_id'])]
            for turn in row['rounds']:
                record_checks['stance_mismatches'] += turn['stance'] != extract_stance(turn['report']['C'])
            last = max(t['round_id'] for t in row['rounds'])
            votes = [extract_stance(t['report']['C']) for t in row['rounds'] if t['round_id'] == last]
            ranked = Counter(votes).most_common()
            record_checks['final_decision_mismatches'] += ranked[0][0] != row['final_decision']
            record_checks['final_top_count_ties'] += len(ranked) > 1 and ranked[0][1] == ranked[1][1]
            if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
                tie_counts[name+'/'+row['condition']] += 1
        bundles.append({'suite': name, 'records': len(rows), 'unique_question_ids': len(ids),
            'split': matrix['defaults'].get('split'),
            'duplicate_keys': len(keys)-len(set(keys)),
            'completed_errors': sum(not r['completed'] or bool(r['error']) for r in rows),
            'provenance_hash_mismatches': bad_files,
            'hash_matches': fingerprint(path) == expected['question_level_sha256']})
    rows = [json.loads(l) for l in (frozen/'formal_v10/question_level.jsonl').read_text().splitlines() if l.strip()]
    groups = defaultdict(list)
    for row in rows: groups[row['dataset'], row['condition']].append(row)
    histories = {}; split_report = []; sequences = []; action_sequences = []
    legacy = []; text_checks = []; sensitivity_pairs = defaultdict(list); removals = []
    for dataset, split in splits.items():
        parts = {k: set(map(str, v)) for k, v in split['splits'].items()}
        split_report.append({'dataset': dataset, 'counts': {k: len(v) for k, v in parts.items()},
            'pair_overlaps': {a+'_'+b: len(parts[a]&parts[b]) for a,b in [('train','dev'),('train','test'),('dev','test')]}})
        cache_path = ROOT/split['source']
        cache = json.loads(cache_path.read_text())
        items = {str(q['question_id']): q for q in cache['questions']}
        formal_ids = list(map(str, split['splits']['test'][:150]))
        def normalized(qid):
            return ' '.join(unicodedata.normalize('NFKC', items[qid]['question']).casefold().split())
        text_sets = {k: {normalized(q) for q in ids} for k, ids in parts.items()}
        formal_texts = [normalized(q) for q in formal_ids]
        text_checks.append({'dataset': dataset,
            'formal_normalized_duplicate_texts': len(formal_texts)-len(set(formal_texts)),
            'split_normalized_text_overlaps': {a+'_'+b: len(text_sets[a]&text_sets[b])
                for a,b in [('train','dev'),('train','test'),('dev','test')]},
            'scope': 'NFKC, case and whitespace normalization only; no semantic paraphrase detection'})
        legacy.append({'dataset': dataset, 'formal_test_questions': len(formal_ids),
            'also_have_legacy_agent_outputs_in_source_cache': sum(bool(items[qid].get('agents')) for qid in formal_ids),
            'cache_hash_matches_split_manifest': fingerprint(cache_path) == split['source_sha256'],
            'interpretation': 'held-out for mechanism development split, not demonstrably never-used questions across whole project'})
        for condition in sorted({r['condition'] for r in rows if r['dataset'] == dataset}):
            histories[dataset+'/'+condition] = history_check(groups[dataset, condition], formal_ids)
        base = {str(r['question_id']): r for r in groups[dataset, 'independent__no_mechanism@0.2']}
        full = {str(r['question_id']): r for r in groups[dataset, 'independent__static_aeic@0.2']}
        seen_test = set()
        for q in formal_ids:
            pair = (base[q]['final_decision'] == base[q]['ground_truth'],
                    full[q]['final_decision'] == full[q]['ground_truth'])
            text = normalized(q)
            duplicate = text in seen_test
            prior_split = text in text_sets['train'] or text in text_sets['dev']
            seen_test.add(text)
            sensitivity_pairs['original'].append(pair)
            def untied_correct(row):
                final_round = max(t['round_id'] for t in row['rounds'])
                counts = Counter(extract_stance(t['report']['C']) for t in row['rounds'] if t['round_id'] == final_round).most_common()
                tied = len(counts) > 1 and counts[0][1] == counts[1][1]
                return row['final_decision'] == row['ground_truth'] and not tied
            sensitivity_pairs['tied_final_votes_count_as_failure'].append((untied_correct(base[q]), untied_correct(full[q])))
            if not duplicate: sensitivity_pairs['deduplicate_test_text_only'].append(pair)
            if not duplicate and not prior_split:
                sensitivity_pairs['deduplicate_and_exclude_train_dev_text_overlap'].append(pair)
            if duplicate or prior_split:
                removals.append({'dataset': dataset, 'question_id': q,
                    'duplicate_test_text': duplicate, 'train_dev_text_overlap': prior_split,
                    'selection_rule': 'first occurrence in frozen test order, then exclude normalized input seen in train/dev; no output labels used'})
        sequences.append([int(full[q]['final_decision'] == full[q]['ground_truth'])-
                          int(base[q]['final_decision'] == base[q]['ground_truth']) for q in formal_ids])
        def effort(row): return sum(t['chosen_action'] in ('aeic_effort','verify') for t in row['rounds'])/len(row['rounds'])
        action_sequences.append([effort(full[q])-effort(base[q]) for q in formal_ids])
    checks['bundle_integrity'] = all(b['hash_matches'] and not b['duplicate_keys'] and not b['completed_errors'] for b in bundles)
    checks['all_provenance_hashes_match'] = record_checks['frozen_file_hash_mismatches'] == 0
    checks['labels_stances_and_aggregation_match'] = all(record_checks[k] == 0 for k in
        ['label_mismatches', 'stance_mismatches', 'final_decision_mismatches'])
    checks['within_split_disjoint'] = all(not any(r['pair_overlaps'].values()) for r in split_report)
    checks['all_source_cache_hashes_match'] = all(r['cache_hash_matches_split_manifest'] for r in legacy)
    checks['reputation_sequence_matches_split_order'] = all(r['prefix_chain_matches'] for r in histories.values())
    checks['normalized_text_disjoint'] = all(not r['formal_normalized_duplicate_texts'] and
        not any(r['split_normalized_text_overlaps'].values()) for r in text_checks)
    extras = {}
    all_formal_ids = set(unique_questions)
    for name, path in [('interface', RESULTS/'mechanism/interface/report_interface_v1.jsonl'),
                       ('decomposition', RESULTS/'mechanism/decomposition/mechanism_decomposition_v1.jsonl')]:
        extra_rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        ids = {(r['dataset'], str(r['question_id'])) for r in extra_rows}
        extras[name] = {'records': len(extra_rows), 'unique_question_ids': len(ids),
                       'overlap_with_frozen_questions': len(ids & unique_questions)}
        all_formal_ids.update(ids)
    return {'checks': checks, 'bundles': bundles, 'record_replay_checks': dict(record_checks),
        'final_ties_by_suite_and_condition': dict(tie_counts),
        'aggregation_rule': 'historical Counter.most_common(1), including neutral; tie resolved by first occurrence, not a strict quorum rule',
        'frozen_unique_dataset_question_ids': len(unique_questions),
        'formal_6360_unique_dataset_question_ids': len(all_formal_ids), 'explanatory_question_overlaps': extras,
        'split_checks': split_report, 'normalized_text_checks': text_checks, 'legacy_exposure': legacy,
        'reputation_histories': histories,
        'input_deduplication_sensitivity': {name: pair_outcomes(pairs) for name,pairs in sensitivity_pairs.items()},
        'input_only_exclusion_manifest': removals,
        'input_sensitivity_scope': 'post-hoc filtered fixed trajectories; no regeneration or state reset, not a new contamination-free collection',
        'suite_overlap_with_formal_questions': {k: len(v&suite_ids['formal_v10']) for k,v in suite_ids.items()},
        'accuracy_sequence_sensitivity': [block_summary(sequences, length) for length in [1,5,10,20]],
        'effort_sequence_sensitivity': [block_summary(action_sequences, length) for length in [1,5,10,20]],
        'inference_limitations': ['condition histories are separate stateful trajectories; not IID paired trials',
            'original McNemar p values retained as historical IID-based analysis, not made exact by bootstrap',
            'single trajectory per dataset/condition cannot establish arbitrary-history robustness',
            'cached labels and past agent outputs preclude a never-seen-across-project claim',
            'legacy exposure is not proof current report prompts directly received answer keys']}


def run():
    main = mainline()
    essential = ['bundle_integrity', 'all_provenance_hashes_match',
                 'labels_stances_and_aggregation_match', 'within_split_disjoint',
                 'all_source_cache_hashes_match', 'reputation_sequence_matches_split_order']
    decision_record = {
        'status': 'offline_review_complete_with_declared_limits' if all(main['checks'][k] for k in essential) else 'integrity_failure_requires_resolution',
        'research_object': 'public process contract + action-bound resource execution + AEIC + fixed historical aggregation',
        'raw_and_input_filtered_joint_gain_reproduced': all(v['delta'] > 0 for v in main['input_deduplication_sensitivity'].values()),
        'normalized_input_isolation_passed_in_original_split': main['checks']['normalized_text_disjoint'],
        'new_api_calls_required_for_recomputing_existing_limited_claims': False,
        'not_established': ['never-used-question independent generalization',
            'pure equal-compute and equal-instruction accuracy advantage',
            'equilibrium or autonomous rational best response', 'unique AEIC format accuracy effect',
            'arbitrary-sequence causal effect or full novelty certification'],
        'conditional_experiment': 'only if stronger identification/generalization is required; freeze one protocol first, no automatic acquisition'}
    return {'version': 'current-process-contract-closeout-2026-10-21',
        'no_api_calls': True, 'historical_data_unchanged': True,
        'inventory': inventory(), 'git_history': git_history(), 'mainline': main,
        'decision': decision_record}


if __name__ == '__main__':
    result = run()
    out = RESULTS/'audit/research_closeout_2026-10-21'
    out.mkdir(parents=True, exist_ok=True)
    (out/'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    (out/'decision.json').write_text(json.dumps(result['decision'], ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'files': result['inventory']['n_files'],
        'dispositions': result['inventory']['dispositions'],
        'main_checks': result['mainline']['checks'],
        'legacy_exposure': result['mainline']['legacy_exposure'],
        'accuracy_block_sensitivity': result['mainline']['accuracy_sequence_sensitivity']}, ensure_ascii=False, indent=2))
