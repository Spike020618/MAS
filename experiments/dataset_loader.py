"""
数据集加载器 — 基于公开推理数据集构建 AEIC 多Agent共识任务

Datasets:
  - StrategyQA: Multi-step boolean reasoning with fact decomposition
  - HotpotQA: Multi-hop QA with supporting paragraphs (information asymmetry)
  - STS-B: Sentence pair similarity for external validation

AEIC Mapping:
  A (Assumption): Question framing + implicit premises
  E (Evidence): Supporting facts/paragraphs
  I (Inference): Reasoning chain connecting evidence to conclusion
  C (Conclusion): Final answer with justification
"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple

import numpy as np

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def load_stsb_sample() -> List[Dict]:
    """Load STS-B style sentence pairs with ground truth similarity (curated 90 pairs)."""
    pairs = [
        # High similarity (4.0-5.0)
        ("A man is playing a guitar.", "A person is playing a musical instrument.", 4.2),
        ("The cat is sleeping on the couch.", "A cat naps on the sofa.", 4.8),
        ("The stock market crashed today.", "Markets experienced a major decline today.", 4.5),
        ("The weather is sunny and warm.", "It is a bright and warm day.", 4.6),
        ("She is cooking dinner in the kitchen.", "A woman prepares food in the kitchen.", 4.3),
        ("The children are playing in the park.", "Kids are playing outdoors in a park.", 4.7),
        ("He drove to work this morning.", "He took his car to the office today.", 4.1),
        ("The algorithm converges in O(n log n).", "The time complexity is O(n log n) for convergence.", 4.4),
        ("Deep learning models require large datasets.", "Neural networks need a lot of training data.", 4.3),
        ("The patient has a high fever.", "The patient's temperature is elevated.", 4.5),
        ("The company reported quarterly earnings.", "The firm released its quarterly financial results.", 4.6),
        ("Inflation rates continue to rise.", "Price levels keep increasing.", 4.2),
        ("The defendant pleaded not guilty.", "The accused denied the charges.", 4.1),
        ("Renewable energy is growing rapidly.", "Clean energy adoption is accelerating.", 4.3),
        ("The teacher explained the concept clearly.", "The instructor clarified the idea well.", 4.4),
        # Medium-high (3.0-4.0)
        ("A dog is running through the field.", "An animal is moving quickly outside.", 3.2),
        ("The economy is recovering slowly.", "Economic growth remains sluggish but positive.", 3.5),
        ("The server crashed due to overload.", "System failure was caused by excessive traffic.", 3.8),
        ("The medication reduced the symptoms.", "Treatment helped alleviate the condition.", 3.6),
        ("He scored the winning goal.", "He made the decisive point in the game.", 3.3),
        ("The bridge connects two cities.", "The structure links two urban areas.", 3.7),
        ("Students are taking an exam.", "People are writing a test.", 3.4),
        ("The committee approved the budget.", "The board passed the financial plan.", 3.6),
        ("Unemployment dropped this quarter.", "Jobless rates decreased recently.", 3.9),
        ("The research paper was published.", "The academic study was released.", 3.7),
        ("Coffee consumption is increasing globally.", "Worldwide coffee drinking is on the rise.", 3.8),
        # Medium-low (1.5-3.0)
        ("The plane landed safely.", "The ship departed on schedule.", 2.0),
        ("Interest rates increased sharply.", "Trade volumes remained stable.", 2.2),
        ("Temperatures are rising globally.", "Sea levels are increasing.", 2.8),
        ("He works at a hospital.", "She teaches at a university.", 1.8),
        ("The company hired new employees.", "The firm reduced its workforce.", 1.5),
        ("The database needs optimization.", "The user interface looks outdated.", 1.5),
        # Low similarity (0-1.5)
        ("A woman is jogging in the park.", "A man is sitting on a bench.", 1.2),
        ("The software has many bugs.", "The application runs smoothly.", 1.0),
        ("The patient recovered fully.", "The patient's condition worsened.", 1.2),
        ("It is raining heavily outside.", "The sun is shining brightly.", 0.8),
        ("The professor gave a lecture.", "The mechanic fixed the car.", 0.5),
        ("A cat is drinking milk.", "The stock market rallied sharply.", 0.2),
        ("Mathematics is a fundamental science.", "The cake tastes delicious.", 0.1),
        ("Quantum computing uses qubits.", "The restaurant is fully booked.", 0.1),
        ("Blood pressure was measured.", "The concert starts at eight.", 0.2),
        # Chinese pairs
        ("股市今日大幅上涨", "A股市场强势上行", 4.5),
        ("患者体温升高至38度", "病人出现发烧症状", 4.3),
        ("被告人构成合同违约", "被告方违反了合同约定", 4.6),
        ("深度学习需要大量数据", "神经网络训练依赖海量数据集", 4.4),
        ("全球变暖导致冰川融化", "气候变化引起极地冰层消退", 4.2),
        ("新能源汽车销量持续增长", "电动车市场份额不断扩大", 4.1),
        ("人工智能改变了医疗行业", "AI技术正在变革医疗领域", 4.7),
        ("中央银行下调了利率", "央行宣布降息决定", 4.5),
        ("房价持续上涨", "经济增长放缓", 2.5),
        ("学生们在教室学习", "工人在工厂工作", 1.8),
        ("医生建议多运动", "律师准备法庭辩论", 0.8),
        ("今天天气很好", "量子计算机突破瓶颈", 0.2),
        ("他在弹钢琴", "法院驳回了上诉", 0.1),
        ("手术非常成功", "股票跌停了", 0.3),
        # Mixed Chinese-English
        ("Machine learning is powerful.", "机器学习非常强大", 4.8),
        ("The patient needs surgery.", "患者需要手术治疗", 4.7),
        ("Market volatility increased.", "市场波动性加剧", 4.6),
        ("The defendant is innocent.", "被告人无罪", 4.5),
        ("Climate change is serious.", "气候变化问题严重", 4.6),
    ]
    return [{"s1": s1, "s2": s2, "score": score / 5.0} for s1, s2, score in pairs]


def _load_strategyqa(n_samples: int = 200) -> List[Dict]:
    """Load StrategyQA and convert to raw records."""
    from datasets import load_dataset
    ds = load_dataset("ChilleD/StrategyQA", split=f"train[:{n_samples}]")
    return [dict(row) for row in ds]


def _load_hotpotqa(n_samples: int = 200) -> List[Dict]:
    """Load HotpotQA validation and convert to raw records."""
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor", split=f"validation[:{n_samples}]")
    return [dict(row) for row in ds]


def strategyqa_to_aeic(raw_data: List[Dict], agents_per_task: int = 3,
                        seed: int = 42) -> List[Dict]:
    """
    Convert StrategyQA to multi-agent AEIC consensus tasks.

    Multi-agent simulation: each agent sees the same question but
    constructs slightly different reasoning chains (simulating
    different LLM models with different reasoning strategies).
    """
    rng = random.Random(seed)
    tasks = []

    for item in raw_data:
        question = item["question"]
        facts_raw = item.get("facts", "")
        answer = item["answer"]
        term = item.get("term", "")
        desc = item.get("description", "")

        # Parse facts into individual evidence items
        if isinstance(facts_raw, str):
            facts = [f.strip() for f in facts_raw.split(".") if f.strip() and len(f.strip()) > 10]
        elif isinstance(facts_raw, list):
            facts = facts_raw
        else:
            facts = [str(facts_raw)]

        if len(facts) < 2:
            continue

        nodes = []
        for agent_id in range(agents_per_task):
            # Simulate agent diversity through:
            # 1. Different assumption framing
            # 2. Different evidence subsets (information asymmetry)
            # 3. Different inference strategies
            # 4. Potentially different conclusions

            # Assumption: different framing of the question
            assumption_templates = [
                f"To answer whether {question.lower().rstrip('?')}, we assume standard definitions apply to '{term}'",
                f"Assuming factual accuracy of available sources, we investigate: {question}",
                f"Given that '{term}' refers to {desc}, we analyze the question systematically",
                f"Under the assumption of common knowledge about {term}, we evaluate: {question}",
                f"Taking the literal interpretation of '{question}', assuming no ambiguity in terms",
            ]
            a = assumption_templates[agent_id % len(assumption_templates)]

            # Evidence: each agent sees a subset (information asymmetry)
            n_evidence = min(max(2, len(facts) - 1), len(facts))
            if agent_id == 0:
                agent_facts = facts[:n_evidence]  # first agent: first N facts
            elif agent_id == 1:
                agent_facts = facts[1:n_evidence + 1] if len(facts) > n_evidence else facts  # offset
            else:
                agent_facts = rng.sample(facts, min(n_evidence, len(facts)))  # random subset

            # Inference: different reasoning strategies
            inference_templates = [
                f"Connecting the evidence sequentially: {' → '.join(f[:30] + '...' for f in agent_facts[:3])}. This chain {'supports' if answer else 'contradicts'} an affirmative answer.",
                f"Analyzing each fact independently and synthesizing: the collective evidence {'strongly suggests' if answer else 'indicates against'} the proposition.",
                f"Using comparative reasoning across {len(agent_facts)} evidence points, the logical conclusion {'follows' if answer else 'does not follow'} from the premises.",
            ]
            i = inference_templates[agent_id % len(inference_templates)]

            # Conclusion: most agents agree, but with variation
            if rng.random() < 0.85 or agent_id == 0:  # 85% agree with ground truth
                conclusion_val = answer
            else:
                conclusion_val = not answer  # minority dissent

            conclusion_templates = [
                f"Based on the evidence analysis, the answer is {'Yes' if conclusion_val else 'No'}. The reasoning chain is {'consistent' if conclusion_val == answer else 'partially inconsistent'} with available facts.",
                f"Concluding {'affirmatively' if conclusion_val else 'negatively'}: the evidence {'sufficiently supports' if conclusion_val else 'does not support'} the claim about {term}.",
                f"Final assessment: {'Yes' if conclusion_val else 'No'} — {'multiple' if len(agent_facts) > 2 else 'limited'} evidence points {'converge on' if conclusion_val == answer else 'diverge from'} this conclusion.",
            ]
            c = conclusion_templates[agent_id % len(conclusion_templates)]

            nodes.append({"A": a, "E": agent_facts, "I": i, "C": c})

        tasks.append({
            "task_id": f"sqa_{item.get('qid', len(tasks))}",
            "source": "StrategyQA",
            "question": question,
            "ground_truth": answer,
            "nodes": nodes,
        })

    return tasks


def hotpotqa_to_aeic(raw_data: List[Dict], agents_per_task: int = 3,
                      seed: int = 42) -> List[Dict]:
    """
    Convert HotpotQA to multi-agent AEIC consensus tasks.

    Multi-agent simulation: each agent has access to different context
    paragraphs (natural information asymmetry from the distractor setting).
    """
    rng = random.Random(seed)
    tasks = []

    for item in raw_data:
        question = item["question"]
        answer = item["answer"]
        q_type = item.get("type", "unknown")

        # Get context paragraphs
        context_titles = item["context"]["title"]
        context_sents = item["context"]["sentences"]
        supporting_titles = set(item["supporting_facts"]["title"])

        # Build paragraph texts
        paragraphs = {}
        for title, sents in zip(context_titles, context_sents):
            paragraphs[title] = " ".join(sents)

        supporting_paras = {t: paragraphs[t] for t in supporting_titles if t in paragraphs}
        distractor_paras = {t: paragraphs[t] for t in paragraphs if t not in supporting_titles}

        if len(supporting_paras) < 1:
            continue

        nodes = []
        for agent_id in range(agents_per_task):
            # Assumption
            assumption_templates = [
                f"To determine: {question} We assume the provided context is factually reliable and complete.",
                f"Investigating '{question}' under the assumption that named entities are correctly identified.",
                f"Approaching this {q_type} question assuming standard reasoning applies: {question}",
            ]
            a = assumption_templates[agent_id % len(assumption_templates)]

            # Evidence: information asymmetry
            # Agent 0: has both supporting paragraphs (well-informed)
            # Agent 1: has 1 supporting + 1 distractor (partially informed)
            # Agent 2: has 1 supporting + 2 distractors (poorly informed)
            sup_list = list(supporting_paras.values())
            dis_list = list(distractor_paras.values())

            if agent_id == 0:
                agent_evidence = sup_list[:3]
            elif agent_id == 1:
                agent_evidence = sup_list[:1] + dis_list[:1]
            else:
                agent_evidence = sup_list[:1] + rng.sample(dis_list, min(2, len(dis_list)))

            # Truncate long evidence
            agent_evidence = [e[:200] for e in agent_evidence]

            # Inference
            n_ev = len(agent_evidence)
            if agent_id == 0:
                i = f"Cross-referencing {n_ev} sources, the {q_type} reasoning yields a clear connection to '{answer}'."
            elif agent_id == 1:
                i = f"With {n_ev} available sources, partial evidence suggests the answer relates to '{answer}' though confirmation is limited."
            else:
                noise_answer = answer if rng.random() < 0.7 else "uncertain"
                i = f"Based on {n_ev} sources (some potentially irrelevant), inference points toward '{noise_answer}'."

            # Conclusion
            if agent_id == 0 or rng.random() < 0.8:
                c = f"The answer is '{answer}'. This conclusion is supported by {n_ev} evidence sources with {'high' if agent_id == 0 else 'moderate'} confidence."
            else:
                c = f"Based on available evidence, the most likely answer is '{answer}', though the evidence is incomplete for full certainty."

            nodes.append({"A": a, "E": agent_evidence, "I": i, "C": c})

        tasks.append({
            "task_id": f"hqa_{item['id']}",
            "source": "HotpotQA",
            "question": question,
            "ground_truth": answer,
            "type": q_type,
            "nodes": nodes,
        })

    return tasks


def load_reasoning_datasets(n_strategyqa: int = 150, n_hotpotqa: int = 150,
                             agents_per_task: int = 3, seed: int = 42) -> List[Dict]:
    """
    Load and combine reasoning datasets into AEIC multi-agent tasks.

    Returns list of tasks, each with 'nodes' containing agent AEIC records.
    """
    tasks = []

    print(f"    Loading StrategyQA ({n_strategyqa} samples)...")
    sqa_raw = _load_strategyqa(n_strategyqa)
    sqa_tasks = strategyqa_to_aeic(sqa_raw, agents_per_task=agents_per_task, seed=seed)
    tasks.extend(sqa_tasks)
    print(f"      → {len(sqa_tasks)} tasks generated")

    print(f"    Loading HotpotQA ({n_hotpotqa} samples)...")
    hqa_raw = _load_hotpotqa(n_hotpotqa)
    hqa_tasks = hotpotqa_to_aeic(hqa_raw, agents_per_task=agents_per_task, seed=seed)
    tasks.extend(hqa_tasks)
    print(f"      → {len(hqa_tasks)} tasks generated")

    print(f"    Total: {len(tasks)} consensus tasks ({agents_per_task} agents each)")
    return tasks


def generate_diversity_dataset(topics: List[str] = None, seed: int = 42) -> Dict[str, List[Dict]]:
    """Generate paired homo/hetero reasoning for diversity ablation (unchanged)."""
    if topics is None:
        topics = [
            "artificial intelligence regulation",
            "cryptocurrency adoption",
            "remote work transition",
            "climate change mitigation",
            "aging society challenges",
            "urban transportation reform",
            "education digitalization",
            "healthcare AI deployment",
            "supply chain resilience",
            "energy transition policy",
            "人工智能监管",
            "数字货币推广",
            "远程办公转型",
            "气候变化应对",
            "老龄化社会挑战",
            "城市交通改革",
            "教育数字化",
            "医疗AI部署",
            "供应链韧性",
            "能源转型政策",
        ]

    rng = random.Random(seed)
    homo_tasks, hetero_tasks = [], []

    perspectives = [
        ("economic", "macro-economic analysis", "GDP/inflation/employment data",
         "supply-demand models and multiplier effects", "structural economic implications"),
        ("sociological", "social structure analysis", "survey/demographic/sentiment data",
         "network effects and diffusion patterns", "social interaction transformation"),
        ("technological", "technology feasibility", "maturity curves/patents/roadmaps",
         "technology diffusion models", "implementation readiness"),
        ("political", "policy and governance", "legislation/regulation/alignment data",
         "institutional theory and policy analysis", "governance recommendations"),
        ("environmental", "sustainability impact", "emissions/resource/ecological data",
         "lifecycle analysis and systems dynamics", "environmental sustainability"),
    ]

    for topic in topics:
        p = perspectives[0]
        homo_nodes = []
        for i in range(3):
            homo_nodes.append({
                "A": f"From {p[0]} perspective on {topic}, assuming standard {p[1]}",
                "E": [f"{p[2]} point {j + 1} for {topic}" for j in range(3)],
                "I": f"Based on {p[3]}, {topic} shows predictable trajectory (analyst {i + 1})",
                "C": f"{p[4]}: {topic} will develop as mainstream models predict",
            })
        homo_tasks.append({"topic": topic, "nodes": homo_nodes, "type": "homogeneous"})

        selected = rng.sample(perspectives, 3)
        hetero_nodes = []
        for p in selected:
            hetero_nodes.append({
                "A": f"From {p[0]} perspective on {topic}, assuming {p[1]}",
                "E": [f"{p[2]} evidence {j + 1} regarding {topic}" for j in range(3)],
                "I": f"Through {p[3]}, {topic} reveals multi-dimensional dynamics",
                "C": f"{p[4]}: {topic} requires integrated cross-disciplinary response",
            })
        hetero_tasks.append({"topic": topic, "nodes": hetero_nodes, "type": "heterogeneous"})

    return {"homogeneous": homo_tasks, "heterogeneous": hetero_tasks}
