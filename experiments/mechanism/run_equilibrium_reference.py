"""Resumable sequence-score -> sealed readouts -> offline labels pipeline.

No API keys, network access, Qwen, or automatic model downloads. Use local
checkpoint or explicit complete score file. Both remain template adaptations,
not a replication of the paper's full dataset preprocessing/model setup.
"""
import argparse
from hashlib import sha256
import json
from math import comb, log
from pathlib import Path

from experiments.core.mechanism.equilibrium_ranking_reference import anchors, solve
from experiments.core.mechanism.sequence_likelihood import requests_for_case, score_tables, HuggingFaceSequenceScorer, checkpoint_digest
from experiments.mechanism.run_upgrade_pilot import freeze, lock


def digest_file(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def rank(ids, scores):
    order = sorted(range(len(ids)), key=lambda i: -scores[i])
    return ids[order[0]] if scores[order[0]] - scores[order[1]] > 1e-10 else None


def readout(case, scores):
    ids, g, d = score_tables(case, scores)
    ga, da = anchors(g, d)
    result = solve(g, d)
    # Raw likelihoods stay in log space; SC/D use the paper's cross-normalized anchors.
    tables = {'G': g[0], 'MI': tuple(g[0][i]+d[i][0] for i in range(len(ids))),
              'SC': ga[0], 'D': tuple(row[0] for row in da),
              'ER-G': result['generator'][0], 'ER-D': tuple(row[0] for row in result['discriminator'])}
    return {'answers': {k: rank(ids, v) for k, v in tables.items()},
            'scores': {k: dict(zip(ids, v)) for k, v in tables.items()},
            'solver': result, 'candidate_ids': ids}


def evaluate(readouts, labels):
    if set(readouts) != set(labels):
        raise ValueError('labels must match sealed case IDs exactly')
    for cid, row in readouts.items():
        if labels[cid] not in row['candidate_ids']:
            raise ValueError('invalid gold candidate ID')
    methods = tuple(next(iter(readouts.values()))['answers'])
    metrics = {m: {'n': len(labels), 'correct': sum(r['answers'][m] == labels[k] for k, r in readouts.items()),
                   'abstain': sum(r['answers'][m] is None for r in readouts.values())} for m in methods}
    pairs = {}
    for game in ('ER-G', 'ER-D'):
        for base in ('G', 'MI', 'SC', 'D'):
            b = sum(r['answers'][game] == labels[k] and r['answers'][base] != labels[k] for k,r in readouts.items())
            c = sum(r['answers'][game] != labels[k] and r['answers'][base] == labels[k] for k,r in readouts.items())
            n = b+c
            p = min(1., 2*sum(comb(n,i) for i in range(min(b,c)+1))/2**n) if n else 1.
            pairs[game+'/'+base] = {'corrections': b, 'harms': c, 'exact_mcnemar_p': p}
    ordered = sorted(pairs, key=lambda k: pairs[k]['exact_mcnemar_p'])
    previous = 0.
    for i, key in enumerate(ordered):
        previous = max(previous, min(1., (len(ordered)-i)*pairs[key]['exact_mcnemar_p']))
        pairs[key]['holm_p'] = previous
    return {'metrics': metrics, 'paired': pairs, 'all_cases_retained': True,
            'scope': 'offline paired comparison, no automatic superiority or novelty claim'}


def run(cases_path, output, scores_path=None, model_path=None, revision=None, labels_path=None):
    output = Path(output)
    cases = json.loads(Path(cases_path).read_text())
    if not isinstance(cases, list) or not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('nonempty unique-ID case list required')
    for case in cases: requests_for_case(case)
    if bool(scores_path) == bool(model_path):
        raise ValueError('exactly one score file or local model required')
    if model_path and (not revision or checkpoint_digest(model_path) != revision):
        raise ValueError('local checkpoint content digest must match --revision')
    with lock(output):
        code_paths = [Path(__file__), Path(__file__).parents[1]/'core/mechanism/equilibrium_ranking_reference.py',
                      Path(__file__).parents[1]/'core/mechanism/sequence_likelihood.py']
        protocol = {'cases_sha256': digest_file(cases_path), 'score_file_sha256': digest_file(scores_path) if scores_path else None,
                    'local_model_path': str(Path(model_path).resolve()) if model_path else None,
                    'revision': revision, 'code_sha256': {p.name: digest_file(p) for p in code_paths},
                    'eta': .1, 'lambda': .1, 'updates': 5000, 'new_api_calls': 0,
                    'scope': 'equation-level reference with explicit prompt adaptation; not full paper reproduction'}
        freeze(output/'protocol.json', protocol)
        source = json.loads(Path(scores_path).read_text()) if scores_path else None
        if source is not None and set(source) != {c['id'] for c in cases}:
            raise ValueError('score file cases mismatch')
        scorer = None
        sealed = {}
        import time
        timings = {}; scoring_seconds = 0.; target_tokens = 0; score_slots = 0
        for case in cases:
            # Hashed paths avoid user IDs acting as filesystem paths.
            key = sha256(case['id'].encode()).hexdigest()
            path = output/(key+'.scores.json')
            if path.exists():
                scores = json.loads(path.read_text())
                if source is not None and scores != source[case['id']]:
                    raise ValueError('cached score changed')
            elif source is not None:
                scores = source[case['id']]
                score_tables(case, scores)
                freeze(path, scores)
            else:
                if scorer is None: scorer = HuggingFaceSequenceScorer(model_path, revision)
                scores = {}
                for slot, req in requests_for_case(case).items():
                    slotpath = output/(key+'.'+sha256(slot.encode()).hexdigest()+'.json')
                    if slotpath.exists(): value = json.loads(slotpath.read_text())
                    else:
                        started = time.perf_counter(); row = scorer(req)
                        value = {'row': row, 'wall_seconds': time.perf_counter()-started}
                        freeze(slotpath, value)
                    scores[slot] = value['row']
                score_tables(case, scores)
                freeze(path, scores)
            started = time.perf_counter()
            sealed[case['id']] = readout(case, scores)
            timings[case['id']] = time.perf_counter()-started
            score_slots += len(scores)
            target_tokens += sum(len(row['token_logprobs']) for row in scores.values())
            if model_path:
                for slot in requests_for_case(case):
                    slotpath = output/(key+'.'+sha256(slot.encode()).hexdigest()+'.json')
                    scoring_seconds += json.loads(slotpath.read_text())['wall_seconds']
        freeze(output/'sealed_readouts.json', sealed)
        # Do not overwrite original runtime measurements on resume.
        if not (output/'solver_timing.json').exists(): freeze(output/'solver_timing.json', timings)
        freeze(output/'costs.json', {'new_api_calls':0, 'unique_sequence_score_slots':score_slots,
            'scored_target_tokens':target_tokens, 'local_scoring_seconds':scoring_seconds if model_path else None,
            'upstream_api_calls':None, 'upstream_tokens':None,
            'note':'imported score-file costs unknown; target tokens are not total prompt or billing tokens'})
        if labels_path:
            labels = json.loads(Path(labels_path).read_text())
            result = {**evaluate(sealed, labels), 'labels_sha256': digest_file(labels_path),
                      'readouts_sha256': digest_file(output/'sealed_readouts.json')}
            freeze(output/'result.json', result)
            return result
        return {'sealed_cases': len(sealed), 'new_api_calls': 0}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', required=True); p.add_argument('--output', required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--scores'); mode.add_argument('--local-model')
    p.add_argument('--revision'); p.add_argument('--labels')
    args = p.parse_args()
    print(json.dumps(run(args.cases, args.output, args.scores, args.local_model, args.revision, args.labels), indent=2))
