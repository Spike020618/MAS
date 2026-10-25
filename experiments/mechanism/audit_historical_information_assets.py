"""Bounded source-code eligibility review; NEVER import or execute collectors."""
import ast
from hashlib import sha256
import json

from experiments.mechanism.replay_budgeted_information import ROOT
from experiments.mechanism.run_upgrade_pilot import freeze

OUT = ROOT/'experiments/results/audit/historical_information_assets_2026-10-23'
CANDIDATES = [
    ('experiments/run_memory_diverse.py', 'generate_evidence_hints',
     'controlled_label_derived_hints',
     'Uses ground_truth to create confirming/misleading hints; valid controlled manipulation, not independently retrieved factual evidence.'),
    ('experiments/mechanism/collect.py', 'peer_context',
     'peer_report_visibility',
     'The reviewed function routes previous reports, majority and opposing evidence; it does not establish private external provider knowledge.'),
    ('experiments/mechanism/pilot_abstract_conflicts.py', 'build_case',
     'synthetic_local_observation_world',
     'Generated local record subsets and paid documents are controlled source observations; not native public-QA effectiveness or real ownership evidence.'),
    ('experiments/mechanism/pilot_verified_contract.py', 'expected_payoffs',
     'synthetic_fair_bit_lookup',
     'Explicit toy acquisition of a random hidden bit; legitimate synthetic information task, not permission to query native QA answer keys.'),
    ('experiments/mechanism/pilot_grounded_evidence.py', 'ranked_sources',
     'central_cached_public_source_extraction',
     'Real source material is useful, but extraction reads controller-accessible snapshots; this interface alone establishes no private cost or information advantage.'),
    ('experiments/mechanism/pilot_external_evidence.py', 'quote_provenance',
     'public_source_quote_provenance',
     'Checks quotes against retrieved source excerpts; provenance is not independent truth verification or a proof of private resource ownership.'),
]


def inspect_function(path, name):
    text = path.read_text()
    tree = ast.parse(text)
    node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    segment = ast.get_source_segment(text, node)
    return {'start_line': node.lineno, 'end_line': node.end_lineno,
            'source_sha256': sha256(path.read_bytes()).hexdigest(),
            'function_sha256': sha256(segment.encode()).hexdigest(), 'text': segment}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for rel, function, classification, meaning in CANDIDATES:
        inspected = inspect_function(ROOT/rel, function)
        rows.append({'path': rel, 'function': function,
            **{k: v for k, v in inspected.items() if k != 'text'},
            'classification': classification, 'interpretation': meaning,
            'directly_qualifies_as_real_private_information_for_current_effort_game': False})
    result = {'scope': 'six inspected historical entry points only, not a complete project-wide nonexistence proof',
        'new_model_calls': 0, 'collectors_imported_or_executed': False,
        'reviewed_entries': rows,
        'preserve_old_results': True,
        'do_not_infer': ['all historical experiments are invalid',
            'formal450 had this hint construction', 'private evidence does not exist anywhere else',
            'synthetic observation tests are fabricated results', 'source provenance guarantees factual truth'],
        'gate': 'locate an actual owned/permissioned or genuinely costly local resource before proposing a new private-information mechanism'}
    freeze(OUT/'result.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__': main()
