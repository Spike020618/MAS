from __future__ import annotations

import json

from .domain import SearchRequest
from .workflow import ConsensusWorkflow


def main() -> None:
    response = ConsensusWorkflow.from_environment().run(SearchRequest(
        query="推荐2010年以后有思想性的科幻电影",
        top_k=3,
        preferred_genres=["科幻", "剧情"],
        min_year=2010,
    ))
    print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
