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
