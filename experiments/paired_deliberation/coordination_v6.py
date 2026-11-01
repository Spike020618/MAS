"""Label-free bounded scheduling, deterministic ties, no LLM value estimator."""
from experiments.paired_deliberation.contracts_v6 import OWNERS


class Coordinator:
    def __init__(self, mode, initial, max_rounds=3):
        if mode not in {"fixed", "adaptive"} or type(max_rounds) is not int or max_rounds < 0:
            raise ValueError("invalid coordination configuration")
        self.mode, self.initial, self.max_rounds = mode, initial, max_rounds
        self.processed, self.trace = set(), []
        owners = [owner for owner in OWNERS if owner in initial] + sorted(set(initial) - set(OWNERS))
        self.fixed = [owner + ":" + initial[owner]["claims"][i]["claim_id"]
                      for i in range(max((len(report["claims"]) for report in initial.values()), default=0))
                      for owner in owners if i < len(initial[owner]["claims"])]

    def features(self, target, state):
        row = state.claims[target]
        claim = row["claim"]
        mids = set(claim["candidate_ids"])
        opposition = any(other["owner"] != row["owner"] and other["status"] != "retracted"
                         and mids.intersection(other["claim"]["candidate_ids"])
                         and {claim["stance"], other["claim"]["stance"]} == {"support", "oppose"}
                         for other in state.claims.values())
        ranks = [set(report["ranking"]) for report in self.initial.values()]
        rank_disagreement = any(any(mid in ranking for ranking in ranks) and
                               not all(mid in ranking for ranking in ranks) for mid in mids)
        return {"opposing_stances": int(opposition), "uncertain_stance": int(claim["stance"] == "unknown"),
                "ranking_disagreement": int(rank_disagreement),
                "has_dependents": int(any(target in other["claim"]["depends_on"] for other in state.claims.values()))}

    def next(self, state):
        if len(self.trace) >= self.max_rounds:
            return None
        available = [key for key, row in state.claims.items() if key not in self.processed
                     and row["own_status"] != "retracted"
                     and row["claim"]["claim_type"] in {"item_fact", "explicit_preference"}]
        if not available:
            return None
        scores = {}
        for target in available:
            f = self.features(target, state)
            # Same unit action-cost assumption; actual compound cost reserved separately.
            scores[target] = {"features": f, "priority": 2 * f["opposing_stances"] + f["uncertain_stance"]
                              + f["ranking_disagreement"] + f["has_dependents"]}
        if self.mode == "fixed":
            order = {target: i for i, target in enumerate(self.fixed)}
            chosen = min(available, key=lambda key: (order.get(key, len(order)), key))
        else:
            chosen = min(available, key=lambda key: (-scores[key]["priority"], key))
        self.processed.add(chosen)
        self.trace.append({"target": chosen, "mode": self.mode, "candidates": scores,
                           "cost_assumption": "one bounded review action; conservative compound reservation"})
        return chosen
