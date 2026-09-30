import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.core.reputation import ReputationSystem, weighted_majority_vote

def test_reputation():
    print("Initializing reputation system for 3 agents...")
    rep = ReputationSystem(n_agents=3, learning_rate=0.2)
    
    # Agent 0: Always correct
    # Agent 1: Random
    # Agent 2: Always wrong
    
    questions = [
        {"ans": ["yes", "no", "no"], "gt": "yes"},
        {"ans": ["no", "yes", "yes"], "gt": "no"},
        {"ans": ["yes", "yes", "no"], "gt": "yes"},
        {"ans": ["no", "no", "yes"], "gt": "no"},
    ]
    
    for i, q in enumerate(questions):
        print(f"\nRound {i+1}:")
        print(f"Current Trust Scores: {[f'{s:.2f}' for s in rep.trust_scores]}")
        print(f"Current Weights: {[f'{w:.2f}' for w in rep.get_weights()]}")
        
        # Test voting with current weights
        vote = weighted_majority_vote(q["ans"], rep.get_weights())
        print(f"Agents answered: {q['ans']}")
        print(f"Weighted vote result: {vote} (GT: {q['gt']})")
        
        # Update reputation
        rep.update(q["ans"], q["gt"])
        
    print("\nFinal Trust Scores:")
    for i, score in enumerate(rep.trust_scores):
        print(f"Agent {i}: {score:.2f}")
        
    if rep.trust_scores[0] > rep.trust_scores[2]:
        print("\nSuccess: Reputation system correctly identified the reliable agent over the unreliable one.")

if __name__ == "__main__":
    test_reputation()
