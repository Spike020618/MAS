"""MovieLens-backed catalog with deterministic, evidence-traceable retrieval."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .domain import CandidateSource, MovieCandidate, SearchRequest, normalize_genres


TOKEN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+", re.I)


def tokens(text: str) -> set[str]:
    values: set[str] = set()
    for chunk in TOKEN.findall(text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
            values.update(chunk[i:i + 2] for i in range(max(1, len(chunk) - 1)))
        else:
            values.add(chunk)
    return values


class JsonlMovieCatalog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.movies: Dict[str, dict] = {}
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    self.movies[str(item["movie_id"])] = item
        if not self.movies:
            raise ValueError(f"empty movie catalog: {self.path}")

    def get(self, movie_id: str) -> Optional[dict]:
        return self.movies.get(str(movie_id))

    @staticmethod
    def _text(item: dict) -> str:
        tags = " ".join(x["tag"] for x in item.get("semantic_tags", []))
        return " ".join([item.get("title", ""), " ".join(item.get("genres", [])), tags]).lower()

    def search(self, request: SearchRequest, limit: int = 50) -> List[MovieCandidate]:
        query_tokens = tokens(request.query)
        preferred = normalize_genres(request.preferred_genres)
        required = normalize_genres(request.required_genres)
        excluded = normalize_genres(request.excluded_genres)
        ranked = []
        max_count = max(int(x.get("rating_count", 0)) for x in self.movies.values())
        for item in self.movies.values():
            year = item.get("year")
            genres = item.get("genres", [])
            genre_lower = normalize_genres(genres)
            if preferred and not preferred.intersection(genre_lower):
                continue
            if required and not required.issubset(genre_lower):
                continue
            if request.min_year and (year is None or year < request.min_year):
                continue
            if request.max_year and (year is None or year > request.max_year):
                continue
            if excluded & genre_lower:
                continue
            text = self._text(item)
            hits = sum(token in text for token in query_tokens if len(token) > 1)
            pref_hits = len(preferred & genre_lower)
            if hits == 0 and pref_hits == 0:
                continue
            popularity = math.log1p(item.get("rating_count", 0)) / math.log1p(max_count)
            rating = float(item.get("rating_mean", 0.0)) / 5.0
            score = min(1.0, 0.12 + 0.08 * hits + 0.12 * pref_hits + 0.08 * popularity + 0.08 * rating)
            sources = [CandidateSource.LEXICAL]
            if pref_hits:
                sources.append(CandidateSource.PREFERENCE)
            ranked.append(MovieCandidate(
                movie_id=str(item["movie_id"]), title=item["title"], year=year or 0,
                genres=genres, sources=sources, retrieval_score=score,
                metadata={
                    "rating": item.get("rating_mean"), "rating_count": item.get("rating_count"),
                    "semantic_tags": item.get("semantic_tags", []), "imdb_id": item.get("imdb_id"),
                    "evidence_refs": item.get("evidence_refs", {}),
                },
            ))
        return sorted(ranked, key=lambda x: (-x.retrieval_score, x.movie_id))[:limit]


class HybridMovieCatalog(JsonlMovieCatalog):
    """Lexical catalog retrieval augmented with train-only ReDial co-occurrence."""

    def __init__(self, path: str | Path, association_path: str | Path,
                 association_weight: float = 0.85):
        super().__init__(path)
        payload = json.loads(Path(association_path).read_text(encoding="utf-8"))
        self.associations = payload.get("associations") or {}
        if not 0.0 <= association_weight <= 1.0:
            raise ValueError("association_weight must be in [0, 1]")
        self.association_weight = association_weight

    @staticmethod
    def _allowed(item: dict, request: SearchRequest) -> bool:
        year = item.get("year")
        genres = normalize_genres(item.get("genres", []))
        preferred = normalize_genres(request.preferred_genres)
        required = normalize_genres(request.required_genres)
        excluded = normalize_genres(request.excluded_genres)
        if preferred and not preferred.intersection(genres):
            return False
        if required and not required.issubset(genres):
            return False
        if excluded & genres:
            return False
        if request.min_year and (year is None or year < request.min_year):
            return False
        if request.max_year and (year is None or year > request.max_year):
            return False
        return str(item["movie_id"]) not in set(map(str, request.excluded_movie_ids))

    def search(self, request: SearchRequest, limit: int = 50) -> List[MovieCandidate]:
        lexical = super().search(request, limit=max(limit, 50))
        merged = {candidate.movie_id: candidate for candidate in lexical}
        lexical_scores = {candidate.movie_id: candidate.retrieval_score for candidate in lexical}
        raw_scores: Dict[str, float] = {}
        for anchor in map(str, request.liked_movie_ids):
            for neighbor in self.associations.get(anchor, []):
                movie_id = str(neighbor["movie_id"])
                raw_scores[movie_id] = max(raw_scores.get(movie_id, 0.0),
                                           float(neighbor.get("score", 0.0)))
        maximum = max(raw_scores.values(), default=0.0)
        lexical_maximum = max(lexical_scores.values(), default=1.0)
        for movie_id, raw_score in raw_scores.items():
            item = self.movies.get(movie_id)
            if item is None or not self._allowed(item, request):
                continue
            existing = merged.get(movie_id)
            if existing is not None:
                if CandidateSource.PREFERENCE not in existing.sources:
                    existing.sources.append(CandidateSource.PREFERENCE)
                existing.metadata["train_cooccurrence_score"] = raw_score
                continue
            merged[movie_id] = MovieCandidate(
                movie_id=movie_id, title=item["title"], year=item.get("year") or 0,
                genres=item.get("genres", []), sources=[CandidateSource.PREFERENCE],
                retrieval_score=0.0,
                metadata={
                    "rating": item.get("rating_mean"), "rating_count": item.get("rating_count"),
                    "semantic_tags": item.get("semantic_tags", []), "imdb_id": item.get("imdb_id"),
                    "evidence_refs": item.get("evidence_refs", {}),
                    "train_cooccurrence_score": raw_score,
                },
            )
        for movie_id, candidate in merged.items():
            lexical_score = lexical_scores.get(movie_id, 0.0) / lexical_maximum
            association_score = raw_scores.get(movie_id, 0.0) / maximum if maximum else 0.0
            candidate.retrieval_score = min(
                1.0,
                (1.0 - self.association_weight) * lexical_score
                + self.association_weight * association_score,
            )
        return sorted(merged.values(), key=lambda x: (-x.retrieval_score, x.movie_id))[:limit]


class CooccurrenceMovieCatalog(HybridMovieCatalog):
    """Train-only co-occurrence retrieval without lexical fallback."""

    def __init__(self, path: str | Path, association_path: str | Path):
        super().__init__(path, association_path, association_weight=1.0)

    def search(self, request: SearchRequest, limit: int = 50) -> List[MovieCandidate]:
        candidates = super().search(request, limit=max(200, limit))
        candidates = [item for item in candidates
                      if "train_cooccurrence_score" in item.metadata]
        candidates.sort(key=lambda item: (
            -float(item.metadata["train_cooccurrence_score"]), item.movie_id
        ))
        return candidates[:limit]
