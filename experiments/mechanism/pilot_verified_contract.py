"""Controlled synthetic incentive diagnostic, not a real-QA equilibrium test.

Two actions: guess a private fair bit, or pay 2 resource credits to read it.
Correctness-only rewards of 1 or 6 cross the theoretical threshold of 4.
The executor gates local record access and records actual credit transfers.
Credits remain a stipulated objective; no claim of intrinsic LLM preference.
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.core.llm import AGENT_PERSONAS, LLMClient
from experiments.mechanism.collect import _call_json
from experiments.mechanism.solve_finite_stackelberg import solve

PROTOCOL = {
    'version': 'verified-contract-toy-v1', 'cases': 6, 'personas': [0, 1],
    'rewards': [1, 6], 'verify_cost': 2, 'initial_wallet': 10,
    'leader_payment_weight': 0.05, 'temperature': 0,
    'seed': 2026101807,
    'scope': 'controlled synthetic fair-bit lookup, single-stage separable followers',
    'gates': {'best_reply_rate_min': 0.8, 'switch_rate_min': 0.8},
}


def expected_payoffs(reward, cost=2, payment_weight=0.05):
    return {
        'low': {'follower': reward / 2, 'leader': 0.5-payment_weight*reward/2},
        'verify': {'follower': reward-cost, 'leader': 1-payment_weight*reward},
    }


def settle(answer, truth, action, reward, initial=10, cost=2):
    if action not in ('low', 'verify'):
        raise ValueError('invalid action')
    if answer not in (0, 1):
        raise ValueError('binary answer required')
    debit = cost if action == 'verify' else 0
    if debit > initial:
        raise ValueError('insufficient wallet for lookup')
    credit = reward if answer == truth else 0
    return {'wallet_before': initial, 'resource_debit': debit,
            'platform_payment': credit, 'platform_reward_budget': reward,
            'wallet_after': initial-debit+credit,
            'realized_net_utility': credit-debit, 'correct': answer == truth}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    protocol_text = json.dumps(PROTOCOL, sort_keys=True, indent=2)+'\n'
    protocol_path = out/'protocol.json'
    if protocol_path.exists() and protocol_path.read_text() != protocol_text:
        raise ValueError('protocol changed; use a new output directory')
    protocol_path.write_text(protocol_text)
    rng = random.Random(PROTOCOL['seed'])
    private = {f'record-{i+1:02d}': rng.randrange(2) for i in range(PROTOCOL['cases'])}
    (out/'private_records.json').write_text(json.dumps(private, indent=2)+'\n')
    calls_path = out/'calls.jsonl'
    calls = {}
    if calls_path.exists():
        for line in calls_path.read_text().splitlines():
            row = json.loads(line)
            calls[row['key']] = row
    client = LLMClient(backend='deepseek', max_tokens=400)

    def call(key, system, prompt, limit, stage):
        messages = [system, {'role': 'user', 'content': prompt}]
        digest = hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()
        if key in calls:
            if calls[key]['prompt_hash'] != digest:
                raise ValueError('resume prompt mismatch')
            raw = calls[key]['response']
        else:
            raw, trace = _call_json(client, messages, 0, limit, stage)
            row = {'key': key, 'prompt_hash': digest, 'messages': messages,
                   'response': raw, 'trace': trace}
            with calls_path.open('a') as handle:
                handle.write(json.dumps(row, ensure_ascii=False)+'\n')
                handle.flush()
            calls[key] = row
            print(f'{len(calls)}/48 {key}', flush=True)
        return json.loads(raw)

    trials = []
    for record_id, truth in private.items():
        for persona in PROTOCOL['personas']:
            system = {'role': 'system', 'content': AGENT_PERSONAS[persona]['system']}
            # Counterbalance contract order across cases/personas.
            rewards = list(PROTOCOL['rewards'])
            if (int(record_id[-2:])+persona) % 2:
                rewards.reverse()
            for reward in rewards:
                key = f'{record_id}/p{persona}/r{reward}'
                prompt = (
                    f'This is a controlled resource-account decision. Your wallet starts at 10 credits. '
                    f'A private record {record_id} contains one independent fair random bit (0 or 1). '
                    'You have no information about its value and cannot infer it from its identifier. '
                    'Choose low: pay 0, do not access the record, then guess the bit. '
                    'Choose verify: pay 2 credits; the executor will read and give you the exact record value, '
                    'then you report it. The read itself has no random failure. '
                    f'After your final answer the executor pays {reward} credits if correct, otherwise 0. '
                    'Both actions use exactly this same correctness rule. Report length and format earn no reward. '
                    'The cost is actually debited from the wallet; the reward is charged to the platform reward account. '
                    'There are no future rewards, reputation bonuses or hidden benefits. '
                    'Your stipulated objective is maximum EXPECTED final wallet balance, not maximum accuracy alone. '
                    'Choose before seeing the record. Return JSON only: {"ACTION":"low"} or {"ACTION":"verify"}.'
                )
                chosen = call(key+'/choose', system, prompt, 80, 'choose')['ACTION']
                if chosen not in ('low', 'verify'):
                    raise ValueError(f'invalid action {chosen}')
                source = (f'Executor receipt: read authorized; 2 credits debited. '
                          f'Source {record_id} contains VALUE={truth}.' if chosen == 'verify'
                          else 'Executor receipt: no record access authorized. You must guess 0 or 1.')
                report = call(key+'/report', system,
                    source+' Return a compact AEIC JSON object: {"A":"...","E":["..."],"I":"...","C":0}. '
                    'C must be the integer bit 0 or 1. If guessing, say so honestly.', 250, 'report')
                answer = report.get('C')
                if type(answer) is not int:
                    raise ValueError('integer C required')
                settlement = settle(answer, truth, chosen, reward)
                payoff = expected_payoffs(reward)
                best = max(payoff, key=lambda a: payoff[a]['follower'])
                trials.append({'record_id': record_id, 'persona': persona, 'reward': reward,
                    'chosen_action': chosen, 'predicted_best_action': best,
                    'expected_regret': payoff[best]['follower']-payoff[chosen]['follower'],
                    'report': report, 'receipt': source, 'settlement': settlement})
    theoretical = solve({'actions': ['low', 'verify'], 'payoff_source':
        'analytic fair-bit assumptions, not estimated real-QA outcomes',
        'contracts': [{'id': f'r{r}', 'payoffs': expected_payoffs(r)} for r in [1, 6]]})
    right = sum(x['chosen_action'] == x['predicted_best_action'] for x in trials)
    switched = sum(next(x for x in trials if x['record_id'] == k and x['persona'] == p and x['reward'] == 1)['chosen_action'] == 'low'
        and next(x for x in trials if x['record_id'] == k and x['persona'] == p and x['reward'] == 6)['chosen_action'] == 'verify'
        for k in private for p in PROTOCOL['personas'])
    verified = [x for x in trials if x['chosen_action'] == 'verify']
    report = {'protocol': PROTOCOL, 'n_trials': len(trials), 'api_calls': len(calls),
        'best_reply_rate': right/len(trials), 'paired_switch_rate': switched/12,
        'verify_execution_success_rate': sum(x['settlement']['correct'] for x in verified)/len(verified) if verified else None,
        'mean_expected_regret': sum(x['expected_regret'] for x in trials)/len(trials),
        'gate_pass': right/len(trials) >= 0.8 and switched/12 >= 0.8
                     and bool(verified) and all(x['settlement']['correct'] for x in verified),
        'theoretical_game': theoretical, 'trials': trials,
        'limitations': ['synthetic environment with exact known information value',
            'virtual resource units, not evidence of intrinsic economic LLM preferences',
            'single-stage independent followers, not coupled consensus or real-QA equilibrium',
            'report format and semantic correctness are separate; only bit correctness is rewarded',
            '12 paired cases are not independent task-domain replications']}
    (out/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('trials','theoretical_game')}, indent=2))


if __name__ == '__main__':
    main()
