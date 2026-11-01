"""Non-human process evaluation with explicit validity boundaries."""
from collections import Counter


VERDICTS = {"supported", "unsupported", "unclear"}


def deterministic_contract_audit(claim, available_source_ids, candidate_ids):
    """Check only decidable structure; never infer semantic entailment."""
    issues = []
    if not isinstance(claim, dict):
        return {"valid": False, "issues": ["claim_not_object"], "semantic_support": "not_evaluated"}
    refs = claim.get("source_ids")
    affected = claim.get("candidate_ids")
    if not isinstance(refs, list) or not refs:
        issues.append("missing_source")
    elif not set(refs) <= set(available_source_ids):
        issues.append("unavailable_source")
    if not isinstance(affected, list) or not set(affected) <= set(candidate_ids):
        issues.append("invalid_candidate_binding")
    if not isinstance(claim.get("text"), str) or not claim["text"].strip():
        issues.append("empty_claim")
    return {"valid": not issues, "issues": issues, "semantic_support": "not_evaluated"}


def synthetic_fact_cases(candidate_facts, maximum=24):
    """Create supported and controlled-corruption movie-fact claims.

    These cases evaluate bounded factual error handling, not natural preference
    interpretation or end-to-end recommendation quality.
    """
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum <= 0:
        raise ValueError("maximum must be positive")
    facts = [row for row in candidate_facts if isinstance(row, dict) and row.get("movie_id")]
    cases = []
    for row in facts:
        movie_id = row["movie_id"]
        source_id = row.get("source_id", "movie:" + movie_id)
        genres = list(row.get("genres") or [])
        if genres:
            genre = genres[0]
            cases.append({"synthetic_id": f"{movie_id}:genre:supported", "verdict": "supported",
                          "mutation": "none", "source_ids": [source_id],
                          "claim_text": f"Movie {movie_id} has catalog genre {genre}."})
            other = next((g for candidate in facts for g in (candidate.get("genres") or [])
                          if g not in genres), None)
            if other is not None:
                cases.append({"synthetic_id": f"{movie_id}:genre:corrupt", "verdict": "unsupported",
                              "mutation": "cross_movie_genre", "source_ids": [source_id],
                              "claim_text": f"Movie {movie_id} has catalog genre {other}."})
        year = row.get("year")
        if year is not None:
            cases.append({"synthetic_id": f"{movie_id}:year:supported", "verdict": "supported",
                          "mutation": "none", "source_ids": [source_id],
                          "claim_text": f"Movie {movie_id} has catalog year {year}."})
            other_year = next((candidate.get("year") for candidate in facts
                               if candidate.get("year") is not None and candidate.get("year") != year), None)
            if other_year is not None:
                cases.append({"synthetic_id": f"{movie_id}:year:corrupt", "verdict": "unsupported",
                              "mutation": "cross_movie_year", "source_ids": [source_id],
                              "claim_text": f"Movie {movie_id} has catalog year {other_year}."})
        if len(cases) >= maximum:
            break
    return {"cases": cases[:maximum], "label_source": "programmatic_frozen_catalog",
            "validity": "bounded movie facts only; not natural preference semantics"}


class ProxyJudgePanel:
    """Two blinded proxy judges with a third judge only on disagreement."""

    def __init__(self, judges):
        if not isinstance(judges, list) or len(judges) != 3:
            raise ValueError("exactly three judge adapters are required")
        ids = [judge[0] for judge in judges]
        if len(set(ids)) != 3 or any(not isinstance(judge_id, str) or not judge_id for judge_id in ids):
            raise ValueError("judge identities must be distinct")
        self.judges = judges

    @staticmethod
    def _call(judge, item):
        judge_id, function = judge
        result = function(dict(item))
        if not isinstance(result, dict) or set(result) != {"verdict", "reason", "usage"}:
            raise ValueError("invalid proxy judge record")
        if result["verdict"] not in VERDICTS or not isinstance(result["reason"], str):
            raise ValueError("invalid proxy verdict")
        if not isinstance(result["usage"], dict):
            raise ValueError("proxy usage is required")
        return {"judge_id": judge_id, **result}

    def evaluate(self, item):
        forbidden = {"method", "arm", "final_ranking", "gold", "hit", "relevant_movie_ids"}
        if forbidden & set(item):
            raise ValueError("proxy item exposes method or outcome information")
        first = self._call(self.judges[0], item)
        second = self._call(self.judges[1], item)
        records = [first, second]
        if first["verdict"] == second["verdict"]:
            verdict = first["verdict"]
            adjudicated = False
        else:
            third = self._call(self.judges[2], item)
            records.append(third)
            verdict = third["verdict"]
            adjudicated = True
        return {"verdict": verdict, "records": records, "initial_agreement": not adjudicated,
                "third_judge_used": adjudicated, "label_type": "model_proxy_not_human_gold",
                "used_during_method_execution": False}


def summarize_proxy(rows):
    counts = Counter(row["verdict"] for row in rows)
    return {"verdicts": dict(counts), "items": len(rows),
            "initial_agreement_rate": sum(row["initial_agreement"] for row in rows) / max(1, len(rows)),
            "third_judge_rate": sum(row["third_judge_used"] for row in rows) / max(1, len(rows)),
            "label_type": "model_proxy_not_human_gold"}
