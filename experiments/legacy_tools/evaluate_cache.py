import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from experiments.core.orchestrator import _aggregate_yes_no

import json

# Load MMLU-Hard star and check how _aggregate_yes_no evaluates it
with open("experiments/results/cache/MMLU-Hard_star.json", "r") as f:
    data = json.load(f)
    
q = data["questions"][0]
print(f"GT: {q.get('ground_truth')}")
agents = q.get('agents', [])

# Manually extract conclusions
yes_votes = 0
no_votes = 0
for a in agents:
    conclusion = a.get('aeic', {}).get('C', '').lower()
    if 'yes' in conclusion and 'no' not in conclusion:
        yes_votes += 1
    elif 'no' in conclusion and 'yes' not in conclusion:
        no_votes += 1
    elif 'yes' in conclusion:
        yes_votes += 1
        
print(f"Yes: {yes_votes}, No: {no_votes}")
if yes_votes > no_votes:
    pred = "yes"
elif no_votes > yes_votes:
    pred = "no"
else:
    pred = "unknown"
print(f"Pred: {pred}")
print(f"Correct? {pred == q.get('ground_truth', '').lower()}")
