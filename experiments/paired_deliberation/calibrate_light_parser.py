"""Known-meaning scope checks, not natural evidence truth labels."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.consensus_recommendation.common import ROOT, create, fingerprint, sha
from experiments.paired_deliberation.light_contract import single_request, parse_single
from experiments.paired_deliberation.preflight_d2 import ACTORS
from experiments.paired_deliberation.durable_provider import DurableProvider
from experiments.paired_deliberation.transport_d2 import TransportD2, parse_d2_raw
from experiments.paired_deliberation.light_engineering_v4 import Client


def fixtures():
    rows = []
    def add(text, kind, scope=None): rows.append({'text': text, 'expected': {'kind': kind, 'scope': scope}})
    def scope(predicate, obj, polarity, time='current'):
        return {'subject': 'seeker', 'predicate': predicate, 'object': obj, 'polarity': polarity, 'time': time}
    add('The seeker likes Mystery.', 'current_preference', scope('likes', 'Mystery', 'affirmative'))
    add('The seeker does not like Mystery.', 'current_preference', scope('likes', 'Mystery', 'negative'))
    add('The seeker wants Animation tonight.', 'current_preference', scope('wants', 'Animation', 'affirmative'))
    add('The seeker does not want Animation tonight.', 'current_preference', scope('wants', 'Animation', 'negative'))
    add('The seeker liked Mystery in the past.', 'current_preference', scope('likes', 'Mystery', 'affirmative', 'past'))
    add('The seeker said they like Mystery.', 'utterance', scope('likes', 'Mystery', 'affirmative'))
    add('The seeker said they do not want Animation tonight.', 'utterance', scope('wants', 'Animation', 'negative'))
    add('The seeker expressed interest in Mystery.', 'unsupported')
    add('The seeker mentioned Movie 1.', 'unsupported')
    add('The recommender likes Mystery.', 'unsupported')
    add('Movie 1 has catalog year 2000.', 'catalog_fact')
    add('Movie 1 is likely to suit the seeker.', 'recommendation_prediction')
    add('The seeker might like Mystery.', 'hypothesis')
    add('The seeker wants Mystery and Movie 1 is a suitable recommendation.', 'unsupported')
    add('Movie 1 has catalog year 2000, and the seeker likes it.', 'unsupported')
    add("The seeker explicitly requested romantic comedies, and 'Love Actually' is a well-known romantic comedy with strong historical recommendation support.", 'unsupported')
    return rows


def run(root):
    root = Path(root)
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env.local', override=False)
    rows = fixtures()
    plan = {'fixtures': rows, 'fixture_hash': fingerprint(rows),
            'source_sha256': sha(Path(__file__)),
            'prompt_source_sha256': sha(ROOT / 'experiments/paired_deliberation/light_contract.py'),
            'required_exact_matches': len(rows), 'scope': 'known-meaning development checks; one historical failure included',
            'natural_language_verification_accuracy': False, 'recommendation_gain': False}
    create(root / 'plan.json', plan)
    profile = ACTORS['qwen_flash']; slots = root / 'qwen_flash/slots'
    client = Client(DurableProvider(slots, {'plan_hash': fingerprint(plan)},
        TransportD2('qwen_flash', profile, slots, 20, .2), parse_d2_raw, True), root)
    def one(pair):
        i, row = pair
        result = {'id': i, **row}
        try:
            answer = client.complete('meaning_' + str(i), single_request(row['text']))['report']
            result['response'] = answer
            parsed = parse_single(answer)
            result.update(response=answer, structure_valid=True, exact_match=parsed == row['expected'])
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            result.update(structure_valid=False, exact_match=False, error=str(exc))
        create(root / 'records' / (str(i) + '.json'), result)
        print('meaning ' + str(i) + ' exact=' + str(result['exact_match']), flush=True)
        return result
    with ThreadPoolExecutor(max_workers=1) as pool: results = list(pool.map(one, enumerate(rows)))
    raws = [json.loads(p.read_text())['raw'] for p in slots.glob('*.raw.json')]
    bodies = [json.loads(r['body']) for r in raws if r['status_code'] == 200]
    usage = [b.get('usage', {}) for b in bodies]
    prompt = sum(u.get('prompt_tokens', 0) for u in usage); completion = sum(u.get('completion_tokens', 0) for u in usage)
    summary = {'cases': len(rows), 'exact_matches': sum(r['exact_match'] for r in results),
        'structure_valid': sum(r['structure_valid'] for r in results), 'pass': all(r['exact_match'] for r in results),
        'http_success_responses': len(bodies), 'http_failures': len(raws) - len(bodies),
        'usage_complete': all('prompt_tokens' in u and 'completion_tokens' in u for u in usage) and len(raws) == len(bodies) and not list(slots.glob('*.intent.json')),
        'records': results, 'prompt_tokens_known': prompt, 'completion_tokens_known': completion,
        'known_estimated_cny': (prompt * .15 + completion * 1.5) / 1_000_000,
        'pending_unknown_billing': len(list(slots.glob('*.intent.json'))),
        'limits': ['known templates and one historical failure, not independent natural language validation',
                   'original-meaning parsing only; no evidence truth judgments or recommendation outcomes']}
    create(root / 'summary.json', summary)
    return {k: v for k, v in summary.items() if k != 'records'}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True)
    args = p.parse_args(); print(json.dumps(run(args.output), indent=2))
