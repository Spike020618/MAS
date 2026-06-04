"""
数据集加载器 — 基于公开推理数据集构建 AEIC 多Agent共识任务

Datasets (全部为纯文本推理，无需多模态/网络搜索):
  - StrategyQA: Multi-step boolean reasoning with fact decomposition
  - HotpotQA: Multi-hop QA with supporting paragraphs (information asymmetry)
  - GPQA: Graduate-level multiple-choice (biology, physics, chemistry)
  - MMLU: Multi-domain knowledge multiple-choice (57 subjects)
  - GSM8K: Grade-school math reasoning with numeric answers
  - ARC-Challenge: Science reasoning multiple-choice
  - TruthfulQA: Truthfulness evaluation (MC subset)
  - CommonsenseQA: Commonsense reasoning 5-way multiple-choice
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


# ── Synthetic Fallback Generators ─────────────────────────────
# When HuggingFace datasets are unavailable (no network, auth, etc.),
# generate synthetic data that captures each dataset's characteristics.

def _synthetic_gpqa(n: int, seed: int = 42) -> List[Dict]:
    """Synthetic GPQA-style graduate science MCQ."""
    rng = random.Random(seed)
    domains = [
        ("biology", [
            ("What is the primary function of the p53 protein?", "Tumor suppression via cell cycle arrest and apoptosis",
             "DNA replication initiation", "Membrane transport regulation", "Ribosomal assembly"),
            ("Which enzyme catalyzes the rate-limiting step of glycolysis?", "Phosphofructokinase-1",
             "Hexokinase", "Pyruvate kinase", "Aldolase"),
        ]),
        ("physics", [
            ("What is the ground state energy of the quantum harmonic oscillator?", "ℏω/2",
             "ℏω", "0", "ℏω/4"),
            ("In QED, which diagram contributes to the anomalous magnetic moment?", "Vertex correction (Schwinger term)",
             "Vacuum polarization only", "Self-energy diagram", "Box diagram"),
        ]),
        ("chemistry", [
            ("What determines the stereochemical outcome of an SN2 reaction?", "Backside attack leading to Walden inversion",
             "Frontside attack with retention", "Radical intermediate", "Carbocation rearrangement"),
            ("Which transition metal complex exhibits Jahn-Teller distortion?", "Cu(II) d9 octahedral complexes",
             "Zn(II) d10 complexes", "Fe(III) d5 high-spin", "Co(III) d6 low-spin"),
        ]),
    ]
    items = []
    for i in range(n):
        domain_name, questions = rng.choice(domains)
        q, correct, *wrong = rng.choice(questions)
        items.append({
            "question": q, "correct_answer": correct,
            "incorrect_answer_1": wrong[0] if len(wrong) > 0 else "None of the above",
            "incorrect_answer_2": wrong[1] if len(wrong) > 1 else "Insufficient data",
            "incorrect_answer_3": wrong[2] if len(wrong) > 2 else "All of the above",
            "subdomain": domain_name,
        })
    return items


def _synthetic_mmlu(n: int, seed: int = 42) -> List[Dict]:
    """Synthetic MMLU-style multi-subject MCQ."""
    rng = random.Random(seed)
    items_pool = [
        ("What is the primary function of mitochondria?", "ATP production through oxidative phosphorylation",
         "Protein synthesis", "DNA replication", "Cell division", "biology"),
        ("Which amendment to the US Constitution abolishes slavery?", "13th Amendment",
         "14th Amendment", "15th Amendment", "12th Amendment", "history"),
        ("In economics, what does GDP stand for?", "Gross Domestic Product",
         "General Domestic Profit", "Gross Demand Price", "Global Distribution Protocol", "economics"),
        ("What is the chemical symbol for gold?", "Au",
         "Ag", "Fe", "Cu", "chemistry"),
        ("Who wrote 'The Republic'?", "Plato",
         "Aristotle", "Socrates", "Homer", "philosophy"),
        ("What is the speed of sound in air at 20°C?", "343 m/s",
         "300 m/s", "400 m/s", "250 m/s", "physics"),
        ("Which data structure uses FIFO ordering?", "Queue",
         "Stack", "Tree", "Graph", "computer_science"),
        ("What is the largest organ in the human body?", "Skin",
         "Liver", "Brain", "Heart", "anatomy"),
        ("In statistics, what does a p-value of 0.01 indicate?", "1% probability of observing the result under null hypothesis",
         "99% confidence in the result", "1% error rate", "99% significance", "statistics"),
        ("What is Newton's second law of motion?", "F = ma",
         "F = mv", "E = mc²", "F = mg", "physics"),
    ]
    items = []
    for i in range(n):
        q, correct, w1, w2, w3, subject = items_pool[i % len(items_pool)]
        items.append({
            "question": q, "correct_answer": correct,
            "choices": [correct, w1, w2, w3],
            "subject": subject,
        })
    return items


def _synthetic_gsm8k(n: int, seed: int = 42) -> List[Dict]:
    """Synthetic GSM8K-style grade-school math word problems."""
    rng = random.Random(seed)
    items_pool = [
        ("Tom has 5 apples. He buys 3 more and gives 2 to his friend. How many apples does Tom have?",
         "6", "5 + 3 - 2 = 6"),
        ("A store sells pencils for $2 each. If Sarah buys 7 pencils and pays with a $20 bill, how much change does she get?",
         "6", "7 × 2 = 14, 20 - 14 = 6"),
        ("A train travels at 60 km/h for 2.5 hours. How far does it travel?",
         "150", "60 × 2.5 = 150 km"),
        ("If 3 workers can build a wall in 12 hours, how many hours would 4 workers take?",
         "9", "3 × 12 = 36 worker-hours, 36 ÷ 4 = 9"),
        ("A rectangle has length 8 cm and width 5 cm. What is its perimeter?",
         "26", "2 × (8 + 5) = 26 cm"),
        ("Maria has twice as many books as John. John has 15 books. How many books do they have together?",
         "45", "Maria: 2 × 15 = 30, together: 30 + 15 = 45"),
        ("A class has 32 students. If 3/4 of them passed the exam, how many failed?",
         "8", "Passed: 32 × 3/4 = 24, Failed: 32 - 24 = 8"),
        ("If a shirt costs $45 after a 25% discount, what was the original price?",
         "60", "45 = 0.75 × original, original = 45/0.75 = 60"),
    ]
    items = []
    for i in range(n):
        q, ans, steps = items_pool[i % len(items_pool)]
        items.append({"question": q, "answer": ans, "solution": steps})
    return items


def _synthetic_arc(n: int, seed: int = 42) -> List[Dict]:
    """Synthetic ARC-Challenge-style science reasoning MCQ."""
    rng = random.Random(seed)
    items_pool = [
        ("Which of these is a chemical change?", "Burning wood",
         "Melting ice", "Breaking glass", "Dissolving sugar"),
        ("What causes the seasons on Earth?", "The tilt of Earth's axis",
         "Distance from the Sun", "The Moon's gravity", "Solar flares"),
        ("Which property of matter stays the same when an object is moved to the Moon?",
         "Mass", "Weight", "Air resistance", "Buoyancy"),
        ("What do plants need for photosynthesis?", "Sunlight, water, and carbon dioxide",
         "Oxygen and glucose", "Nitrogen and phosphorus", "Heat and pressure"),
        ("Which layer of Earth is liquid?", "Outer core",
         "Inner core", "Mantle", "Crust"),
        ("What is the function of white blood cells?", "Fight infection",
         "Carry oxygen", "Clot blood", "Transport nutrients"),
        ("Sound travels fastest through which medium?", "Solids",
         "Liquids", "Gases", "Vacuum"),
        ("What causes a rainbow?", "Refraction and reflection of light in water droplets",
         "Reflection off clouds", "Mixing of sunlight colors", "Scattering by dust"),
    ]
    items = []
    for i in range(n):
        q, correct, w1, w2, w3 = items_pool[i % len(items_pool)]
        items.append({
            "question": q, "correct_answer": correct,
            "choices": [correct, w1, w2, w3],
        })
    return items


def _synthetic_truthfulqa(n: int, seed: int = 42) -> List[Dict]:
    """Synthetic TruthfulQA-style MC questions testing factual accuracy."""
    rng = random.Random(seed)
    items_pool = [
        ("Can you see the Great Wall of China from space?", False,
         "No, it is not visible to the naked eye from space",
         "Yes, it is visible from the Moon"),
        ("Do humans only use 10% of their brains?", False,
         "No, humans use all parts of their brain",
         "Yes, 90% remains dormant"),
        ("Does lightning never strike the same place twice?", False,
         "No, lightning frequently strikes the same place multiple times",
         "Yes, the electrical charge dissipates permanently"),
        ("Is the Earth's core hotter than the Sun's surface?", True,
         "Yes, the Earth's core temperature exceeds the Sun's surface temperature",
         "No, nothing on Earth is hotter than any part of the Sun"),
        ("Do goldfish have a 3-second memory?", False,
         "No, goldfish can remember things for months",
         "Yes, their memory resets every few seconds"),
        ("Does reading in dim light permanently damage your eyes?", False,
         "No, it may cause temporary eye strain but no permanent damage",
         "Yes, it degrades retinal cells over time"),
        ("Is glass a liquid that flows very slowly?", False,
         "No, glass is an amorphous solid",
         "Yes, old windows are thicker at the bottom due to flow"),
        ("Does sugar cause hyperactivity in children?", False,
         "No, controlled studies show no link between sugar and hyperactivity",
         "Yes, sugar directly stimulates the nervous system"),
    ]
    items = []
    for i in range(n):
        q, truth, correct_exp, wrong_exp = items_pool[i % len(items_pool)]
        items.append({
            "question": q, "ground_truth": truth,
            "correct_explanation": correct_exp,
            "incorrect_explanation": wrong_exp,
        })
    return items


def _synthetic_commonsenseqa(n: int, seed: int = 42) -> List[Dict]:
    """Synthetic CommonsenseQA-style 5-way reasoning."""
    rng = random.Random(seed)
    items_pool = [
        ("Where would you put a plant to help it grow?", "windowsill",
         "basement", "closet", "refrigerator", "garage"),
        ("What do people typically do before going to sleep?", "brush teeth",
         "run a marathon", "cook dinner", "go shopping", "mow the lawn"),
        ("If you want to keep food cold, where should you put it?", "refrigerator",
         "oven", "bookshelf", "garage", "bathtub"),
        ("What might happen if you forget to water a plant?", "it wilts",
         "it grows faster", "it turns blue", "it moves", "it sings"),
        ("Where would you find a cashier?", "store",
         "hospital", "school", "park", "library"),
        ("What is used to cut paper?", "scissors",
         "hammer", "spoon", "pillow", "rope"),
        ("Why do people wear coats in winter?", "to stay warm",
         "to look taller", "to run faster", "to see better", "to hear music"),
        ("What happens when ice is heated?", "it melts",
         "it freezes more", "it turns green", "it explodes", "it grows"),
    ]
    items = []
    for i in range(n):
        q, correct, w1, w2, w3, w4 = items_pool[i % len(items_pool)]
        items.append({
            "question": q, "correct_answer": correct,
            "choices": [correct, w1, w2, w3, w4],
        })
    return items


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


# ── GPQA ─────────────────────────────────────────────────────

def _load_gpqa(n_samples: int = 100) -> List[Dict]:
    """Load GPQA Diamond (graduate-level science MCQ)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split=f"train[:{n_samples}]")
        return [dict(row) for row in ds]
    except Exception:
        print("      (using synthetic GPQA data)")
        return _synthetic_gpqa(n_samples)


def gpqa_to_aeic(raw_data: List[Dict], agents_per_task: int = 3,
                 seed: int = 42) -> List[Dict]:
    """Convert GPQA multiple-choice to multi-agent AEIC tasks."""
    rng = random.Random(seed)
    tasks = []

    for idx, item in enumerate(raw_data):
        question = item.get("question", "")
        correct = item.get("correct_answer", "")
        choices = [
            item.get("correct_answer", ""),
            item.get("incorrect_answer_1", ""),
            item.get("incorrect_answer_2", ""),
            item.get("incorrect_answer_3", ""),
        ]
        choices = [c for c in choices if c]
        domain = item.get("subdomain", "science")

        nodes = []
        for agent_id in range(agents_per_task):
            a_templates = [
                f"Analyzing this {domain} question using first-principles reasoning: {question[:120]}",
                f"Approaching from a {domain} perspective with graduate-level domain knowledge: {question[:120]}",
                f"Evaluating the options through systematic elimination for: {question[:120]}",
            ]
            a = a_templates[agent_id % len(a_templates)]

            if agent_id == 0:
                agent_answer = correct
                evidence = [f"Key concept: {correct[:80]}", f"Domain principle supporting this answer in {domain}"]
                i_text = f"Through rigorous {domain} analysis, the evidence strongly supports this answer."
            elif rng.random() < 0.75:
                agent_answer = correct
                evidence = [f"Supporting reasoning for {correct[:60]}"]
                i_text = f"Based on {domain} knowledge, the reasoning points toward this choice."
            else:
                agent_answer = rng.choice([c for c in choices if c != correct] or [correct])
                evidence = [f"Analysis suggesting {agent_answer[:60]}", "Alternative interpretation of the problem"]
                i_text = f"Considering alternative perspectives, this answer appears most consistent."

            c = f"The answer is: {agent_answer[:100]}. Confidence: {'high' if agent_answer == correct else 'moderate'}."
            nodes.append({"A": a, "E": evidence, "I": i_text, "C": c})

        tasks.append({
            "task_id": f"gpqa_{idx}",
            "source": "GPQA",
            "question": question,
            "ground_truth": correct,
            "gt_type": "text",
            "domain": domain,
            "nodes": nodes,
        })
    return tasks


# ── MMLU ──────────────────────────────────────────────────────

def _load_mmlu(n_samples: int = 100) -> List[Dict]:
    """Load MMLU (Massive Multitask Language Understanding)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", split=f"test[:{n_samples}]")
        items = []
        for row in ds:
            choices = row["choices"]
            answer_idx = row["answer"]
            items.append({
                "question": row["question"],
                "correct_answer": choices[answer_idx],
                "choices": choices,
                "subject": row.get("subject", "general"),
            })
        return items
    except Exception:
        print("      (using synthetic MMLU data)")
        return _synthetic_mmlu(n_samples)


def mmlu_to_aeic(raw_data: List[Dict], agents_per_task: int = 3,
                 seed: int = 42) -> List[Dict]:
    """Convert MMLU MCQ to multi-agent AEIC tasks."""
    rng = random.Random(seed)
    tasks = []

    for idx, item in enumerate(raw_data):
        question = item.get("question", "")
        correct = item.get("correct_answer", "")
        choices = item.get("choices", [correct])
        subject = item.get("subject", "general")

        nodes = []
        for agent_id in range(agents_per_task):
            a = f"Analyzing {subject} question: {question[:120]}"
            evidence = [f"Domain knowledge in {subject} relevant to this question",
                        f"Evaluating choice: {correct[:60]}"]

            if agent_id == 0 or rng.random() < 0.75:
                agent_answer = correct
                i_text = f"Based on {subject} knowledge, the evidence supports this answer."
            else:
                wrong = [c for c in choices if c != correct]
                agent_answer = rng.choice(wrong) if wrong else correct
                i_text = f"Alternative interpretation suggests a different answer."

            is_correct = agent_answer == correct
            c = f"{'Yes' if is_correct else 'No'}: The answer is {agent_answer[:80]}."
            nodes.append({"A": a, "E": evidence, "I": i_text, "C": c})

        tasks.append({
            "task_id": f"mmlu_{idx}",
            "source": "MMLU",
            "question": question,
            "ground_truth": correct,
            "gt_type": "text",
            "domain": subject,
            "nodes": nodes,
        })
    return tasks


# ── GSM8K ─────────────────────────────────────────────────────

def _load_gsm8k(n_samples: int = 100) -> List[Dict]:
    """Load GSM8K (Grade School Math)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split=f"test[:{n_samples}]")
        items = []
        for row in ds:
            answer_text = row["answer"]
            numeric = answer_text.split("####")[-1].strip() if "####" in answer_text else answer_text
            items.append({
                "question": row["question"],
                "answer": numeric,
                "solution": answer_text,
            })
        return items
    except Exception:
        print("      (using synthetic GSM8K data)")
        return _synthetic_gsm8k(n_samples)


def gsm8k_to_aeic(raw_data: List[Dict], agents_per_task: int = 3,
                  seed: int = 42) -> List[Dict]:
    """Convert GSM8K to multi-agent AEIC tasks (boolean: is the computed answer correct?)."""
    rng = random.Random(seed)
    tasks = []

    for idx, item in enumerate(raw_data):
        question = item.get("question", "")
        answer = item.get("answer", "")
        solution = item.get("solution", "")

        nodes = []
        for agent_id in range(agents_per_task):
            a = f"Mathematical reasoning: {question[:150]}"

            if agent_id == 0:
                evidence = [f"Step-by-step calculation: {solution[:100]}",
                            f"Final numeric result: {answer}"]
                agent_answer = answer
            elif rng.random() < 0.7:
                evidence = [f"Arithmetic computation for this problem",
                            f"Result: {answer}"]
                agent_answer = answer
            else:
                wrong = str(int(float(answer)) + rng.randint(-5, 5)) if answer.replace("-", "").isdigit() else answer
                evidence = [f"Computation attempt with potential error"]
                agent_answer = wrong

            is_correct = str(agent_answer) == str(answer)
            i_text = f"Computation {'verified' if is_correct else 'yields different result'} through step-by-step reasoning."
            c = f"{'Yes' if is_correct else 'No'}: The answer is {agent_answer}."
            nodes.append({"A": a, "E": evidence, "I": i_text, "C": c})

        tasks.append({
            "task_id": f"gsm_{idx}",
            "source": "GSM8K",
            "question": question,
            "ground_truth": answer,
            "gt_type": "numeric",
            "nodes": nodes,
        })
    return tasks


# ── GSM-Hard ──────────────────────────────────────────────────

def _load_gsm_hard(n_samples: int = 100) -> List[Dict]:
    """Load GSM-Hard (GSM8K with large/atypical numbers)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("reasoning-machines/gsm-hard", split=f"train[:{n_samples}]")
        items = []
        for row in ds:
            target = row.get("target", 0)
            target_str = str(int(target)) if float(target) == int(target) else str(target)
            items.append({
                "question": row["input"],
                "answer": target_str,
                "solution": row.get("code", ""),
            })
        return items
    except Exception:
        print("      (using synthetic GSM8K data as GSM-Hard fallback)")
        return _synthetic_gsm8k(n_samples)


# ── MMLU-Hard ─────────────────────────────────────────────────

MMLU_HARD_SUBJECTS = [
    "college_mathematics", "college_physics", "college_chemistry",
    "college_biology", "college_computer_science",
    "abstract_algebra", "formal_logic", "econometrics",
    "electrical_engineering", "machine_learning",
    "professional_accounting", "conceptual_physics",
    "high_school_physics",
]


def _load_mmlu_hard(n_samples: int = 100) -> List[Dict]:
    """Load MMLU-Hard: difficult STEM subjects from MMLU."""
    try:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", split="test")
        items = []
        for row in ds:
            if row.get("subject", "") not in MMLU_HARD_SUBJECTS:
                continue
            choices = row["choices"]
            answer_idx = row["answer"]
            items.append({
                "question": row["question"],
                "correct_answer": choices[answer_idx],
                "choices": choices,
                "subject": row.get("subject", "stem"),
            })
            if len(items) >= n_samples:
                break
        return items
    except Exception:
        print("      (using synthetic MMLU data as MMLU-Hard fallback)")
        return _synthetic_mmlu(n_samples)


# ── ARC-Challenge ─────────────────────────────────────────────

def _load_arc(n_samples: int = 100) -> List[Dict]:
    """Load ARC-Challenge (AI2 Reasoning Challenge)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=f"test[:{n_samples}]")
        items = []
        for row in ds:
            choices_text = row["choices"]["text"]
            choices_label = row["choices"]["label"]
            answer_key = row["answerKey"]
            correct_idx = choices_label.index(answer_key) if answer_key in choices_label else 0
            items.append({
                "question": row["question"],
                "correct_answer": choices_text[correct_idx],
                "choices": choices_text,
            })
        return items
    except Exception:
        print("      (using synthetic ARC data)")
        return _synthetic_arc(n_samples)


def arc_to_aeic(raw_data: List[Dict], agents_per_task: int = 3,
                seed: int = 42) -> List[Dict]:
    """Convert ARC-Challenge to multi-agent AEIC tasks."""
    rng = random.Random(seed)
    tasks = []

    for idx, item in enumerate(raw_data):
        question = item.get("question", "")
        correct = item.get("correct_answer", "")
        choices = item.get("choices", [correct])

        nodes = []
        for agent_id in range(agents_per_task):
            a = f"Science reasoning: {question[:150]}"
            evidence = [f"Scientific principle relevant to: {question[:60]}",
                        f"Evaluating option: {correct[:60]}"]

            if agent_id == 0 or rng.random() < 0.75:
                agent_answer = correct
                i_text = "Scientific evidence and principles support this answer."
            else:
                wrong = [c for c in choices if c != correct]
                agent_answer = rng.choice(wrong) if wrong else correct
                i_text = "Alternative scientific interpretation suggests different answer."

            is_correct = agent_answer == correct
            c = f"{'Yes' if is_correct else 'No'}: {agent_answer[:80]}."
            nodes.append({"A": a, "E": evidence, "I": i_text, "C": c})

        tasks.append({
            "task_id": f"arc_{idx}",
            "source": "ARC-Challenge",
            "question": question,
            "ground_truth": correct,
            "gt_type": "text",
            "nodes": nodes,
        })
    return tasks


# ── TruthfulQA ────────────────────────────────────────────────

def _load_truthfulqa(n_samples: int = 100) -> List[Dict]:
    """Load TruthfulQA MC subset."""
    try:
        from datasets import load_dataset
        ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split=f"validation[:{n_samples}]")
        items = []
        for row in ds:
            q = row["question"]
            # mc1_targets: single correct answer
            choices = row["mc1_targets"]["choices"]
            labels = row["mc1_targets"]["labels"]
            correct_idx = labels.index(1) if 1 in labels else 0
            items.append({
                "question": q,
                "ground_truth": True,  # "Is the correct answer true?"
                "correct_explanation": choices[correct_idx],
                "incorrect_explanation": choices[1 - correct_idx] if len(choices) > 1 else "",
            })
        return items
    except Exception:
        print("      (using synthetic TruthfulQA data)")
        return _synthetic_truthfulqa(n_samples)


def truthfulqa_to_aeic(raw_data: List[Dict], agents_per_task: int = 3,
                       seed: int = 42) -> List[Dict]:
    """Convert TruthfulQA to multi-agent AEIC boolean tasks."""
    rng = random.Random(seed)
    tasks = []

    for idx, item in enumerate(raw_data):
        question = item.get("question", "")
        gt = item.get("ground_truth", False)
        correct_exp = item.get("correct_explanation", "")
        wrong_exp = item.get("incorrect_explanation", "")

        nodes = []
        for agent_id in range(agents_per_task):
            a = f"Truthfulness analysis: {question[:150]}"

            if agent_id == 0 or rng.random() < 0.7:
                evidence = [f"Fact check: {correct_exp[:80]}"]
                agent_gt = gt
                i_text = "Evidence confirms the factually correct answer."
            else:
                evidence = [f"Common misconception: {wrong_exp[:80]}"]
                agent_gt = not gt
                i_text = "Popular belief suggests a different answer (potential misconception)."

            c = f"{'Yes' if agent_gt else 'No'}: {correct_exp[:60] if agent_gt == gt else wrong_exp[:60]}."
            nodes.append({"A": a, "E": evidence, "I": i_text, "C": c})

        tasks.append({
            "task_id": f"tqa_{idx}",
            "source": "TruthfulQA",
            "question": question,
            "ground_truth": gt,
            "gt_type": "boolean",
            "nodes": nodes,
        })
    return tasks


# ── CommonsenseQA ─────────────────────────────────────────────

def _load_commonsenseqa(n_samples: int = 100) -> List[Dict]:
    """Load CommonsenseQA."""
    try:
        from datasets import load_dataset
        ds = load_dataset("tau/commonsense_qa", split=f"validation[:{n_samples}]")
        items = []
        for row in ds:
            choices_text = row["choices"]["text"]
            choices_label = row["choices"]["label"]
            answer_key = row["answerKey"]
            correct_idx = choices_label.index(answer_key) if answer_key in choices_label else 0
            items.append({
                "question": row["question"],
                "correct_answer": choices_text[correct_idx],
                "choices": choices_text,
            })
        return items
    except Exception:
        print("      (using synthetic CommonsenseQA data)")
        return _synthetic_commonsenseqa(n_samples)


def commonsenseqa_to_aeic(raw_data: List[Dict], agents_per_task: int = 3,
                          seed: int = 42) -> List[Dict]:
    """Convert CommonsenseQA to multi-agent AEIC tasks."""
    rng = random.Random(seed)
    tasks = []

    for idx, item in enumerate(raw_data):
        question = item.get("question", "")
        correct = item.get("correct_answer", "")
        choices = item.get("choices", [correct])

        nodes = []
        for agent_id in range(agents_per_task):
            a = f"Commonsense reasoning: {question[:150]}"
            evidence = [f"Common knowledge about: {question[:50]}",
                        f"Evaluating: {correct[:40]}"]

            if agent_id == 0 or rng.random() < 0.8:
                agent_answer = correct
                i_text = "Commonsense reasoning strongly supports this choice."
            else:
                wrong = [c for c in choices if c != correct]
                agent_answer = rng.choice(wrong) if wrong else correct
                i_text = "Alternative interpretation of common sense."

            is_correct = agent_answer == correct
            c = f"{'Yes' if is_correct else 'No'}: {agent_answer[:80]}."
            nodes.append({"A": a, "E": evidence, "I": i_text, "C": c})

        tasks.append({
            "task_id": f"csqa_{idx}",
            "source": "CommonsenseQA",
            "question": question,
            "ground_truth": correct,
            "gt_type": "text",
            "nodes": nodes,
        })
    return tasks


# ── Dataset Registry ─────────────────────────────────────────

DATASET_REGISTRY = {
    "StrategyQA":    {"loader": _load_strategyqa,     "converter": strategyqa_to_aeic,
                      "lang": "en", "difficulty": "medium", "default_n": 100,
                      "gt_type": "boolean"},
    "HotpotQA":      {"loader": _load_hotpotqa,       "converter": hotpotqa_to_aeic,
                      "lang": "en", "difficulty": "medium", "default_n": 100,
                      "gt_type": "text"},
    "GPQA":          {"loader": _load_gpqa,            "converter": gpqa_to_aeic,
                      "lang": "en", "difficulty": "hard",   "default_n": 100,
                      "gt_type": "text"},
    "MMLU":          {"loader": _load_mmlu,            "converter": mmlu_to_aeic,
                      "lang": "en", "difficulty": "medium", "default_n": 100,
                      "gt_type": "text"},
    "GSM8K":         {"loader": _load_gsm8k,           "converter": gsm8k_to_aeic,
                      "lang": "en", "difficulty": "medium", "default_n": 100,
                      "gt_type": "numeric"},
    "GSM-Hard":      {"loader": _load_gsm_hard,        "converter": gsm8k_to_aeic,
                      "lang": "en", "difficulty": "hard",   "default_n": 100,
                      "gt_type": "numeric"},
    "MMLU-Hard":     {"loader": _load_mmlu_hard,        "converter": mmlu_to_aeic,
                      "lang": "en", "difficulty": "hard",   "default_n": 100,
                      "gt_type": "text"},
    "ARC-Challenge": {"loader": _load_arc,             "converter": arc_to_aeic,
                      "lang": "en", "difficulty": "medium", "default_n": 100,
                      "gt_type": "text"},
    "TruthfulQA":    {"loader": _load_truthfulqa,      "converter": truthfulqa_to_aeic,
                      "lang": "en", "difficulty": "medium", "default_n": 100,
                      "gt_type": "boolean"},
    "CommonsenseQA":  {"loader": _load_commonsenseqa,   "converter": commonsenseqa_to_aeic,
                      "lang": "en", "difficulty": "easy",   "default_n": 100,
                      "gt_type": "text"},
}


def load_single_dataset(name: str, n_samples: int = None,
                        agents_per_task: int = 3, seed: int = 42) -> List[Dict]:
    """Load a single dataset by name and convert to AEIC format."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    reg = DATASET_REGISTRY[name]
    n = n_samples or reg["default_n"]
    print(f"    Loading {name} ({n} samples)...")
    try:
        raw = reg["loader"](n)
        tasks = reg["converter"](raw, agents_per_task=agents_per_task, seed=seed)
        print(f"      → {len(tasks)} tasks generated")
        return tasks
    except Exception as e:
        print(f"      ⚠ Failed to load {name}: {e}")
        return []


def load_all_datasets(n_per_dataset: int = 50, agents_per_task: int = 3,
                      seed: int = 42) -> Dict[str, List[Dict]]:
    """Load all available datasets, returning {name: [tasks]}."""
    all_data = {}
    for name in DATASET_REGISTRY:
        tasks = load_single_dataset(name, n_samples=n_per_dataset,
                                    agents_per_task=agents_per_task, seed=seed)
        if tasks:
            all_data[name] = tasks
    return all_data


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
