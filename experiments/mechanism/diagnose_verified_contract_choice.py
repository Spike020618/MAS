"""Post-hoc four-call arithmetic elicitation; never merged with frozen pilot."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.core.llm import LLMClient
from experiments.mechanism.collect import _call_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('pilot', type=Path)
    args = p.parse_args()
    target = args.pilot/'posthoc_arithmetic.jsonl'
    previous = {}
    if target.exists():
        previous = {x['key']: x for x in map(json.loads, target.read_text().splitlines())}
    client = LLMClient(backend='deepseek')
    for line in (args.pilot/'calls.jsonl').read_text().splitlines():
        row = json.loads(line)
        if not row['key'].startswith('record-01/') or not row['key'].endswith('/choose'):
            continue
        if row['key'] in previous:
            continue
        messages = row['messages'][:-1] + [{'role':'user','content':row['messages'][-1]['content']+
            '\nFor this separate diagnostic, calculate the expected FINAL wallet balances '
            'of both actions before choosing. Output JSON with exactly '
            'low_expected_wallet, verify_expected_wallet, explanation, ACTION. '
            'The explanation must justify the comparison; ACTION must maximize the computed expectation.'}]
        raw, trace = _call_json(client, messages, 0, 500, 'posthoc_arithmetic')
        item = {'key':row['key'], 'scope':'post-hoc elicitation, not original pilot',
                'messages':messages, 'response':raw, 'trace':trace}
        with target.open('a') as f:
            f.write(json.dumps(item,ensure_ascii=False)+'\n')
        print(row['key'],raw,flush=True)


if __name__ == '__main__':
    main()
