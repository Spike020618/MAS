"""Small in-memory catalog used by the executable vertical slice.

Production storage is intentionally behind this interface so PostgreSQL or an
external search service can replace it without changing the consensus layer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List

from .domain import CandidateSource, MovieCandidate, SearchRequest, normalize_genres


_TOKEN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+", re.I)


def _tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for chunk in _TOKEN.findall(text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
            tokens.update(chunk[i:i + 2] for i in range(max(1, len(chunk) - 1)))
        else:
            tokens.add(chunk)
    return tokens


@dataclass(frozen=True)
class Movie:
    movie_id: str
    title: str
    year: int
    genres: List[str]
    director: str
    description: str
    rating: float


DEFAULT_MOVIES = [
    Movie("m1", "星际穿越", 2014, ["科幻", "剧情", "冒险"], "克里斯托弗·诺兰", "宇航员穿越虫洞寻找人类新家园。", 9.4),
    Movie("m2", "盗梦空间", 2010, ["科幻", "悬疑", "动作"], "克里斯托弗·诺兰", "盗梦团队进入多层梦境执行植入任务。", 9.4),
    Movie("m3", "流浪地球", 2019, ["科幻", "灾难"], "郭帆", "人类推动地球逃离太阳危机。", 7.9),
    Movie("m4", "银翼杀手2049", 2017, ["科幻", "悬疑", "剧情"], "丹尼斯·维伦纽瓦", "复制人警探追查可能改变社会秩序的秘密。", 8.3),
    Movie("m5", "降临", 2016, ["科幻", "剧情"], "丹尼斯·维伦纽瓦", "语言学家尝试理解外星文明的语言与时间观。", 7.8),
    Movie("m6", "机器人总动员", 2008, ["科幻", "动画", "爱情"], "安德鲁·斯坦顿", "清洁机器人踏上太空旅程并唤醒人类。", 9.3),
    Movie("m7", "霸王别姬", 1993, ["剧情", "历史"], "陈凯歌", "两位京剧演员跨越时代的命运。", 9.6),
    Movie("m8", "疯狂的麦克斯：狂暴之路", 2015, ["动作", "科幻", "冒险"], "乔治·米勒", "末日荒原中的高速逃亡与反抗。", 8.7),
]


class InMemoryCatalog:
    def __init__(self, movies: Iterable[Movie] = DEFAULT_MOVIES):
        self.movies: Dict[str, Movie] = {movie.movie_id: movie for movie in movies}

    def get(self, movie_id: str) -> Movie | None:
        return self.movies.get(movie_id)

    def search(self, request: SearchRequest, limit: int = 12) -> List[MovieCandidate]:
        query_tokens = _tokens(request.query)
        preferred = normalize_genres(request.preferred_genres)
        required = normalize_genres(request.required_genres)
        excluded = normalize_genres(request.excluded_genres)
        ranked = []
        for movie in self.movies.values():
            if request.min_year and movie.year < request.min_year:
                continue
            if request.max_year and movie.year > request.max_year:
                continue
            movie_genres = normalize_genres(movie.genres)
            if excluded.intersection(movie_genres):
                continue
            if required and not required.issubset(movie_genres):
                continue
            if preferred and not preferred.intersection(movie_genres):
                continue
            text = " ".join([movie.title, movie.director, movie.description, *movie.genres]).lower()
            lexical_hits = sum(token in text for token in query_tokens if len(token) > 1)
            preference_hits = len(preferred.intersection(movie_genres))
            score = min(0.96, 0.10 + min(4, lexical_hits) * 0.10
                        + preference_hits * 0.15 + movie.rating / 100)
            if lexical_hits == 0 and preference_hits == 0:
                continue
            sources = [CandidateSource.LEXICAL]
            if preference_hits:
                sources.append(CandidateSource.PREFERENCE)
            ranked.append(MovieCandidate(
                movie_id=movie.movie_id,
                title=movie.title,
                year=movie.year,
                genres=list(movie.genres),
                sources=sources,
                retrieval_score=score,
                metadata={"director": movie.director, "description": movie.description, "rating": movie.rating},
            ))
        return sorted(ranked, key=lambda item: (-item.retrieval_score, item.movie_id))[:limit]
