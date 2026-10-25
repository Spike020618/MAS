"""No-API integration audit on existing source snapshots; no quality/equilibrium experiment."""
import argparse
import hashlib
from itertools import permutations, product
import json
from pathlib import Path
from experiments.core.mechanism.source_sharing import AcquisitionLedger, SourceSharingGame


def audit(root):
    path = root/'experiments/results/mechanism/stackelberg/external_evidence_pilot_v2/retrieval.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    results = []
    for row in rows:
        sources = {}
        for source in row['sources']:
            snapshot = (root/source['snapshot_path']).resolve()
            if not snapshot.is_relative_to(root.resolve()):
                raise ValueError('snapshot outside repository')
            data = snapshot.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if snapshot.stem != digest:
                raise ValueError('snapshot filename hash mismatch')
            if hashlib.sha256(source['excerpt'].encode()).hexdigest() != source['excerpt_sha256']:
                raise ValueError('excerpt hash mismatch')
            sources[digest] = data
        if not sources:
            raise ValueError('missing source bank')
        ids = sorted(sources)
        agents = ['a', 'b', 'c']
        # Public index allocation; independent of answers and all report outcomes.
        bundles = {'a': ids[:1], 'b': ids[1:2], 'c': ids}
        weights = {s: 1/len(ids) for s in ids}
        contracts = []
        for budget in [0, .5, 1, 2, 4]:
            # DIAGNOSTIC units, not a calibrated cost estimate or preregistration.
            game = SourceSharingGame(agents, bundles, weights, {a: [0, .75] for a in agents}, budget)
            runs = []
            for order in permutations(agents):
                for initial in product((0, 1), repeat=3):
                    run = game.run_local_updates(initial, order)
                    ledger = AcquisitionLedger(row['key'], game, sources)
                    for i, agent in enumerate(agents):
                        if run['terminal']['profile'][i]:
                            for source in bundles[agent]:
                                ledger.acquire(agent, source)
                    settlement = ledger.settle(run['terminal']['profile'])
                    if max(run['regrets']) > 1e-10 or settlement['paid'] > budget+1e-10:
                        raise AssertionError('runtime mismatch')
                    runs.append({'initial': initial, 'order': order, **run,
                                 'receipt_count': len(settlement['receipts'])})
            contracts.append({'budget': budget, 'equilibria': game.equilibria(), 'runs': runs})
        results.append({'question_id': row['key'], 'source_hashes': ids, 'contracts': contracts})
    return {'scope': 'engineering integration on previously exposed sources; no new QA or LLM behavior evidence',
            'api_calls': 0, 'calibrated_costs': False, 'leader_quality_optimization_performed': False,
            'n_questions': len(results), 'n_contracts': 5,
            'n_executions': sum(len(c['runs']) for r in results for c in r['contracts']),
            'checks_passed': True, 'results': results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(Path(__file__).resolve().parents[2])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'results'}, ensure_ascii=False))


if __name__ == '__main__': main()
