class ReputationSystem:
    def __init__(self, n_agents, initial_trust=1.0, learning_rate=0.1):
        self.n_agents = n_agents
        self.trust_scores = [initial_trust] * n_agents
        self.learning_rate = learning_rate
        
    def update(self, agent_answers, correct_answer):
        """Update trust scores based on correctness."""
        for i, ans in enumerate(agent_answers):
            # Simple binary correctness for now
            is_correct = (str(ans).lower().strip() == str(correct_answer).lower().strip())
            
            if is_correct:
                # Increase trust (bounded at 2.0)
                self.trust_scores[i] = min(2.0, self.trust_scores[i] + self.learning_rate)
            else:
                # Decrease trust (bounded at 0.1)
                self.trust_scores[i] = max(0.1, self.trust_scores[i] - self.learning_rate)
                
    def get_weights(self):
        """Return normalized weights for voting."""
        total = sum(self.trust_scores)
        if total == 0:
            return [1.0 / self.n_agents] * self.n_agents
        return [score / total for score in self.trust_scores]

def weighted_majority_vote(answers, weights):
    """Aggregate answers using agent reputation weights."""
    scores = {}
    for ans, w in zip(answers, weights):
        ans = str(ans).lower().strip()
        scores[ans] = scores.get(ans, 0.0) + w
        
    if not scores:
        return None
        
    # Return answer with highest combined weight
    return max(scores.items(), key=lambda x: x[1])[0]
