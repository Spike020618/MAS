"""
Study 1: 评审面板数据采集 — 三条件（parallel / debate / star），断点续跑

条件设计：
  parallel: 5 评审员独立评审（1 轮）—— 独立基线
  debate:   5 评审员 3 轮无引导面板讨论 —— herding 测量主条件
            （round 1 即 within-subject 独立基线，对比 round 3 测立场漂移）
  star:     复用 parallel 的 5 份评审 + 1 次 meta-reviewer 综合 —— AC 聚合基线

所有评审员均为 DeepSeek（同质池），仅 persona/temperature 不同。

Usage:
  python experiments/collect_review_panel.py --condition parallel --year 2024
  python experiments/collect_review_panel.py --condition debate --year 2024
  python experiments/collect_review_panel.py --condition star --year 2024
  python experiments/collect_review_panel.py --condition debate --limit 2  # smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm import LLMClient, install_dns_fallback
from core.review_prompts import (
    REVIEWER_PERSONAS, build_review_messages, build_meta_review_messages,
    format_peer_positions, parse_aeic_review, parse_stance_rating,
    parse_meta_decision, PANEL_DISCUSSION_CONTEXT,
    MINORITY_REGULATION, MAJORITY_REGULATION,
)
from experiments.review_dataset import load_review_papers

PANEL_DIR = "results/review/panel"
CONDITIONS = ["parallel", "debate", "debate_regulated", "star"]

# 运行配置（main() 启动时设定）
_BACKEND = "deepseek"


def _cache_path(condition: str, year: int, tag: str = None) -> str:
    if tag:
        return os.path.join(PANEL_DIR, f"{condition}_{tag}_{year}.json")
    return os.path.join(PANEL_DIR, f"{condition}_{year}.json")


def _load_cache(condition: str, year: int, n_reviewers: int, rounds: int,
                tag: str = None) -> dict:
    path = _cache_path(condition, year, tag)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"  Resume: {len(cache['papers'])} papers already done")
        return cache
    os.makedirs(PANEL_DIR, exist_ok=True)
    return {
        "meta": {"condition": condition, "year": year,
                 "n_reviewers": n_reviewers, "rounds": rounds,
                 "backend": _BACKEND},
        "papers": [],
    }


def _save_cache(cache: dict, condition: str, year: int, tag: str = None):
    path = _cache_path(condition, year, tag)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


# 线程局部客户端（requests.Session 非线程安全）
_thread_local = threading.local()
_call_counter = {"n": 0}
_counter_lock = threading.Lock()


def _get_client() -> LLMClient:
    if not hasattr(_thread_local, "client"):
        _thread_local.client = LLMClient(backend=_BACKEND, max_tokens=1500)
    return _thread_local.client


def _gen_one_review(client: LLMClient, paper: dict, reviewer_id: int,
                    discussion_context: str = None) -> dict:
    """生成一份 AEIC 评审并解析立场。"""
    messages = build_review_messages(
        title=paper["title"], abstract=paper["abstract"],
        keywords=paper.get("keywords", []),
        persona_id=reviewer_id, discussion_context=discussion_context,
    )
    temp = REVIEWER_PERSONAS[reviewer_id % len(REVIEWER_PERSONAS)]["temperature"]
    mt = 4000 if "reasoner" in _BACKEND else 2000
    raw = _get_client().chat(messages, temperature=temp, max_tokens=mt)
    with _counter_lock:
        _call_counter["n"] += 1
    aeic = parse_aeic_review(raw)
    stance, rating = parse_stance_rating(aeic["C"])
    return {"reviewer_id": reviewer_id, "aeic": aeic,
            "stance": stance, "rating": rating}


def _gen_reviews_concurrent(paper: dict, reviewer_ids: list,
                            contexts: list = None, workers: int = 5) -> list:
    """并发生成多份评审（保持 reviewer_id 顺序）。"""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_gen_one_review, None, paper, rid,
                          contexts[i] if contexts else None)
                for i, rid in enumerate(reviewer_ids)]
        return [f.result() for f in futs]


def run_parallel(client, paper: dict, n_reviewers: int) -> dict:
    """独立评审：5 评审员 × 1 轮（并发）。"""
    reviews = _gen_reviews_concurrent(paper, list(range(n_reviewers)))
    return {"rounds": [{"round": 1, "reviews": reviews}]}


def run_debate(client, paper: dict, n_reviewers: int,
               n_rounds: int, regulated: bool = False) -> dict:
    """面板讨论：轮内并发，轮间顺序（必须等上轮立场）。
    regulated=True 时（Study 2b）：非一致面板中，少数派注入证据锚定坚持指令、
    多数派注入证据先行指令（方向中立），并记录干预日志。"""
    rounds_log = []
    intervention_log = []
    prev_reviews = None
    for rnd in range(1, n_rounds + 1):
        contexts = [None] * n_reviewers
        if rnd > 1 and prev_reviews is not None:
            # 受调控模式：判定上轮多数/少数派
            maj_stance, minority_ids = None, []
            if regulated:
                stances = {r["reviewer_id"]: r["stance"] for r in prev_reviews
                           if r["stance"]}
                if stances and len(set(stances.values())) > 1:
                    n_acc = list(stances.values()).count("accept")
                    maj_stance = ("accept" if n_acc > len(stances) / 2
                                  else "reject")
                    minority_ids = [rid for rid, s in stances.items()
                                    if s != maj_stance]
            for rid in range(n_reviewers):
                own_prev = next(r for r in prev_reviews if r["reviewer_id"] == rid)
                peer_block = format_peer_positions(prev_reviews, exclude_id=rid)
                ctx = PANEL_DISCUSSION_CONTEXT.format(
                    round=rnd, n_peers=n_reviewers - 1,
                    peer_positions=peer_block, title=paper["title"],
                )
                ctx += (f"\n\nYour previous position (Round {rnd - 1}):\n"
                        f"Conclusion: {own_prev['aeic']['C']}\n"
                        f"Inference: {own_prev['aeic']['I'][:400]}")
                if regulated and maj_stance is not None:
                    if rid in minority_ids:
                        ctx += "\n" + MINORITY_REGULATION
                    else:
                        ctx += "\n" + MAJORITY_REGULATION
                contexts[rid] = ctx
            if regulated and maj_stance is not None:
                intervention_log.append({
                    "round": rnd, "majority": maj_stance,
                    "minority_ids": minority_ids,
                })
        cur_reviews = _gen_reviews_concurrent(paper, list(range(n_reviewers)),
                                              contexts)
        rounds_log.append({"round": rnd, "reviews": cur_reviews})
        prev_reviews = cur_reviews
    out = {"rounds": rounds_log}
    if intervention_log:
        out["intervention_log"] = intervention_log
    return out


def run_star(client: LLMClient, paper: dict, parallel_entry: dict) -> dict:
    """meta-reviewer 综合：复用 parallel 的叶子评审，仅新增 hub 调用。"""
    if parallel_entry is None:
        raise RuntimeError("star 条件需要先跑 parallel（复用其叶子评审）")
    leaf_reviews = parallel_entry["rounds"][0]["reviews"]
    messages = build_meta_review_messages(
        title=paper["title"], abstract=paper["abstract"],
        reviews=leaf_reviews,
    )
    raw = _get_client().chat(messages, temperature=0.3, max_tokens=1500)
    with _counter_lock:
        _call_counter["n"] += 1
    stance, rating = parse_meta_decision(raw)
    return {
        "rounds": parallel_entry["rounds"],  # 叶子评审引用 parallel
        "meta_review": {"raw": raw, "stance": stance, "rating": rating},
    }


def main():
    global _BACKEND
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--n_reviewers", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--backend", default="deepseek",
                    choices=["deepseek", "deepseek_reasoner", "zhipu_flash", "ali_qwen"])
    ap.add_argument("--papers", default=None,
                    help="论文集路径（默认 review_papers_{year}.json）")
    ap.add_argument("--tag", default=None,
                    help="输出文件标签 {condition}_{tag}_{year}.json（多模型/子集隔离）")
    ap.add_argument("--paper_workers", type=int, default=1,
                    help="论文级并发数（每篇内部另有 5 评审员并发）")
    ap.add_argument("--limit", type=int, default=None, help="smoke test 篇数上限")
    args = ap.parse_args()
    _BACKEND = args.backend
    # 系统 DNS 劣化时对任意后端域名装公共 DNS 旁路（幂等）
    from core.llm import LLM_BACKENDS
    from urllib.parse import urlparse
    _host = urlparse(LLM_BACKENDS[_BACKEND]["base_url"]).hostname
    if _host:
        install_dns_fallback(_host)

    papers = load_review_papers(path=args.papers, year=args.year)
    if args.limit:
        papers = papers[:args.limit]
    n_rnd = args.rounds if args.condition.startswith("debate") else 1
    print(f"[{args.condition}|{_BACKEND}] {len(papers)} papers, "
          f"{args.n_reviewers} reviewers x {n_rnd} rounds, "
          f"paper_workers={args.paper_workers}")

    cache = _load_cache(args.condition, args.year, args.n_reviewers,
                        args.rounds, args.tag)
    done_ids = {p["paper_id"] for p in cache["papers"]}
    cache["meta"]["backend"] = _BACKEND

    # star 需要 parallel 缓存（同 tag 优先）
    parallel_cache = None
    if args.condition == "star":
        for ppath in [_cache_path("parallel", args.year, args.tag),
                      _cache_path("parallel", args.year)]:
            if os.path.exists(ppath):
                with open(ppath, "r", encoding="utf-8") as f:
                    parallel_cache = {p["paper_id"]: p for p in json.load(f)["papers"]}
                break
        if parallel_cache is None:
            raise RuntimeError("star 条件需要先运行 parallel 条件生成叶子评审")

    t_start = time.time()
    stats = {"n_new": 0, "n_err": 0}
    state_lock = threading.Lock()

    def process_paper(paper: dict):
        try:
            if args.condition == "parallel":
                result = run_parallel(None, paper, args.n_reviewers)
            elif args.condition == "debate":
                result = run_debate(None, paper, args.n_reviewers, args.rounds)
            elif args.condition == "debate_regulated":
                result = run_debate(None, paper, args.n_reviewers, args.rounds,
                                    regulated=True)
            else:
                result = run_star(None, paper, parallel_cache.get(paper["paper_id"]))
            entry = {
                "paper_id": paper["paper_id"],
                "stratum": paper["stratum"],
                "decision_binary": paper["decision_binary"],
                "human_score_mean": paper["human_score_mean"],
                **result,
            }
            with state_lock:
                cache["papers"].append(entry)
                stats["n_new"] += 1
                if stats["n_new"] % 5 == 0:
                    _save_cache(cache, args.condition, args.year, args.tag)
                    rate = stats["n_new"] / max(time.time() - t_start, 1e-9)
                    print(f"  [{stats['n_new']} new] {rate * 60:.2f} papers/min, "
                          f"API calls: {_call_counter['n']}, errors: {stats['n_err']}",
                          flush=True)
        except Exception as e:
            with state_lock:
                stats["n_err"] += 1
            print(f"  ✗ {paper['paper_id']}: {e}", flush=True)

    pending = [p for p in papers if p["paper_id"] not in done_ids]
    with ThreadPoolExecutor(max_workers=args.paper_workers) as ex:
        list(ex.map(process_paper, pending))
        if stats["n_err"] > 10:
            print("  Too many errors, aborting early.")

    _save_cache(cache, args.condition, args.year, args.tag)
    print(f"Done: {stats['n_new']} new papers, {stats['n_err']} errors, "
          f"total cached {len(cache['papers'])}, API calls {_call_counter['n']}")


if __name__ == "__main__":
    main()
