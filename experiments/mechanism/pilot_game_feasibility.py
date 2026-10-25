"""Small diagnostic of payoff arithmetic versus autonomous contract response.

Uses three dev questions, two personas, three contracts. A single execution per
action measures diagnostic payoffs, not population expected payoffs or equilibrium.
Writes every API call immediately and resumes successful units.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.core.llm import LLMClient, AGENT_PERSONAS
from experiments.core.mechanism.contracts import CONTRACTS, scale_rewards, scale_costs, render_contract, parse_action
from experiments.core.mechanism.schema import AgentAction
from experiments.core.mechanism.controller import MechanismController
from experiments.mechanism.collect import load_local_questions, execution_prompt, build_choice_prompt, parse_aeic, _call_json

ACTIONS = list(AgentAction)
CONTRACT_GRID = [(0.05, 1.0), (0.2, 1.5), (0.5, 0.5)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = args.out_dir / 'calls.jsonl'
    completed = {}
    if raw.exists():
        for line in raw.read_text().splitlines():
            item = json.loads(line)
            completed[item['key']] = item
    client = LLMClient(backend='deepseek', max_tokens=1000)

    def call(key, messages, limit, stage):
        if key in completed:
            return completed[key]['response']
        response, trace = _call_json(client, messages, 0.0, limit, stage)
        item = {'key': key, 'prompt_sha256': hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest(),
                'messages': messages, 'response': response, 'trace': trace}
        with raw.open('a') as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + '\n')
            handle.flush()
        completed[key] = item
        print(f'{len(completed)}/66 {key}', flush=True)
        return response

    outcomes = []
    for dataset in ('StrategyQA', 'MMLU-Hard', 'TruthfulQA'):
        question = load_local_questions(dataset, 1, 'dev')[0]
        context = 'No peer information is visible. Reason independently.'
        for persona_id in (0, 1):
            system = {'role': 'system', 'content': AGENT_PERSONAS[persona_id]['system']}
            base = f'{dataset}/{question["question_id"]}/p{persona_id}'
            reports = {}
            for action in ACTIONS:
                prompt = execution_prompt(question['question'], context, action)
                draft = call(base + '/' + action.value, [system, {'role': 'user', 'content': prompt}],
                             350 if action == AgentAction.LOW_EFFORT else 1000, action.value)
                if action == AgentAction.VERIFY:
                    draft = call(base + '/verify_revision', [system, {'role': 'user', 'content':
                        execution_prompt(question['question'], context, action, draft)}], 1000, 'verify_revision')
                reports[action.value] = parse_aeic(draft)
            for reward, cost in CONTRACT_GRID:
                spec = scale_costs(scale_rewards(CONTRACTS['static_aeic'], reward), cost)
                controller = MechanismController(1)
                payoffs = {}
                for action in ACTIONS:
                    result = controller.settle(0, spec.to_offer('pilot'), action, reports[action.value], None)
                    payoffs[action.value] = result['utility']
                tag = f'{base}/r{reward}_c{cost}'
                choice = parse_action(call(tag + '/autonomous', [system, {'role': 'user', 'content':
                    build_choice_prompt(question['question'], render_contract(spec), context, 0.0)}], 80, 'choose'))
                table = '\n'.join(f'{a}: {v}' for a,v in payoffs.items())
                informed = parse_action(call(tag + '/explicit_payoffs', [system, {'role': 'user', 'content':
                    'Diagnostic decision task. Net utilities are given below; there are no other rewards or future effects. '
                    'Choose an action with the largest value. Return JSON {"ACTION":"action_name"}.\n' + table}], 80, 'choose_table'))
                maximum = max(payoffs.values())
                outcomes.append({'dataset': dataset, 'question_id': question['question_id'], 'persona': persona_id,
                    'reward': reward, 'cost': cost, 'diagnostic_payoffs': payoffs,
                    'autonomous_action': choice.value, 'explicit_payoff_action': informed.value,
                    'autonomous_diagnostic_regret': maximum-payoffs[choice.value],
                    'explicit_payoff_regret': maximum-payoffs[informed.value]})
    report = {'scope': 'three dev questions, two personas, one forced sample/action; diagnostic, no equilibrium claim',
              'execution_backend': 'deepseek-chat for every action; verify adds revision; no actual peer data for imitate',
              'n_decisions': len(outcomes), 'api_calls': len(completed),
              'autonomous_best_reply_rate': sum(x['autonomous_diagnostic_regret'] <= 1e-9 for x in outcomes)/len(outcomes),
              'explicit_payoff_best_reply_rate': sum(x['explicit_payoff_regret'] <= 1e-9 for x in outcomes)/len(outcomes),
              'outcomes': outcomes}
    (args.out_dir/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'outcomes'}, indent=2))


if __name__ == '__main__':
    main()
