import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.core.reputation import ReputationSystem, weighted_majority_vote

def evaluate_repeated_games():
    print("=== Direction 4: Long-term Trust & Repeated Games ===\n")
    
    cache_dir = "experiments/results/cache"
    dataset = "MMLU-Hard"
    topo = "star" # Changing to star since parallel doesn't exist
    
    file_path = os.path.join(cache_dir, f"{dataset}_{topo}.json")
    if not os.path.exists(file_path):
        print(f"Required file {file_path} not found.")
        return
        
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    questions = data.get("questions", [])
    if not questions:
        return
        
    n_agents = len(questions[0].get("agents", []))
    print(f"Simulating streaming evaluation on {dataset}_{topo} with {n_agents} agents ({len(questions)} rounds)...")
    
    # Track metrics
    default_maj_correct = 0
    reputation_correct = 0
    
    # Initialize reputation system
    rep = ReputationSystem(n_agents=n_agents, learning_rate=0.15)
    
    for round_idx, q in enumerate(questions):
        gt = str(q.get("ground_truth", "")).lower().strip()
        
        # Extract votes
        agent_votes = []
        for a in q.get("agents", []):
            c = a.get("aeic", {}).get("C", "").lower()
            if 'yes' in c and 'no' not in c:
                agent_votes.append('yes')
            elif 'no' in c and 'yes' not in c:
                agent_votes.append('no')
            elif c.strip().startswith('yes'):
                agent_votes.append('yes')
            elif c.strip().startswith('no'):
                agent_votes.append('no')
            else:
                agent_votes.append('unknown')
                
        # 1. Default Majority Vote
        yes_votes = agent_votes.count('yes')
        no_votes = agent_votes.count('no')
        def_pred = 'yes' if yes_votes > no_votes else 'no' if no_votes > yes_votes else 'unknown'
        
        if def_pred == gt:
            default_maj_correct += 1
            
        # 2. Reputation-Weighted Vote
        weights = rep.get_weights()
        rep_pred = weighted_majority_vote(agent_votes, weights)
        
        if rep_pred == gt:
            reputation_correct += 1
            
        # 3. Update Reputation for next round
        rep.update(agent_votes, gt)
        
    print(f"\nFinal Results after {len(questions)} rounds:")
    print(f"  Default Majority Acc:   {default_maj_correct / len(questions) * 100:.1f}%")
    print(f"  Reputation-Weighted Acc: {reputation_correct / len(questions) * 100:.1f}%")
    
    print("\nFinal Agent Trust Scores:")
    for i, s in enumerate(rep.trust_scores):
        print(f"  Agent {i}: {s:.2f}")
        
    if reputation_correct > default_maj_correct:
        print(f"\nSuccess! Reputation tracking improved accuracy by +{(reputation_correct - default_maj_correct) / len(questions) * 100:.1f}pp")
    else:
        print(f"\nNote: In this specific dataset ({dataset}), agents might have similar underlying reliabilities, so reputation tracking yielded similar results to uniform majority vote.")

if __name__ == "__main__":
    evaluate_repeated_games()
