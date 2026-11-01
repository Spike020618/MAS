import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/results/information_roles_pilot_v1"
CFG = {"version": "information-role-engineering.v1", "cases": 8, "history": 20,
       "candidates_per_base": 20, "top_k": 5, "rrf_k": 60,
       "temperature": 0, "max_tokens": 1200, "calls_per_case": 4,
       "messages_byte_cap_per_call": 24000, "model": "deepseek-chat",
       "budget_per_method": {"calls": 2, "output_token_cap": 2400,
                             "messages_byte_cap": 48000},
       "scope": "previously opened development users; engineering only; no game learning"}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, obj):
    path = Path(path)
    content = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise ValueError("immutable artifact mismatch: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(content)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_rank(ranking, candidates):
    if not isinstance(ranking, list) or any(not isinstance(mid, str) for mid in ranking):
        raise ValueError("ranking must contain string IDs")
    if len(ranking) != len(candidates) or len(set(ranking)) != len(ranking) or set(ranking) != set(candidates):
        raise ValueError("ranking is not a complete candidate permutation")
    return ranking


def rrf(left, right):
    validate_rank(left, right)
    ranks = [{mid: rank for rank, mid in enumerate(order, 1)} for order in (left, right)]
    return sorted(left, key=lambda mid: (-sum(1 / (CFG["rrf_k"] + r[mid]) for r in ranks), mid))


def messages(case, role, previous=None):
    # No case/user identity, timestamps or evaluation labels go to the provider.
    source = {"candidates": case["catalog"]}
    if role in ("P", "U1", "U2"):
        source["observed_history"] = case["history_packet"]
    if role in ("C", "U1", "U2"):
        source["collaborative_predictions"] = case["cf_packet"]
    instruction = {
        "P": "Estimate personal relevance using the observed ratings and movie genres. Unseen is not disliked.",
        "C": "Examine collaborative model predictions and similarity to observed history. Scores are uncertain predictions, not facts.",
        "U1": "Use both observed history and collaborative predictions to rank movies by personal relevance.",
        "U2": "Reconsider the prior ranking using both sources. Change it only if justified; keeping it is allowed."}[role]
    if previous is not None:
        source["prior_ranking"] = previous
    system = ("You rank a fixed movie candidate set for a person. "
              "Do not infer that unobserved movies are disliked. Do not invent actors or plots. "
              "Return JSON only: {\"ranking\":[\"movie_id\",...]}. Include each candidate exactly once. "
              "Movie IDs are identifiers, not relevance scores. No prose or extra fields.")
    result = [{"role": "system", "content": system},
              {"role": "user", "content": instruction + "\n" + json.dumps(source, ensure_ascii=False)}]
    if sum(len(m["content"].encode()) for m in result) > CFG["messages_byte_cap_per_call"]:
        raise ValueError("prompt exceeds frozen byte budget")
    return result


def code_dependencies():
    files = list(Path(__file__).parent.glob("*.py")) + [
        ROOT / "experiments/slate_choice/baselines.py", ROOT / "experiments/slate_choice/common.py"]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}
