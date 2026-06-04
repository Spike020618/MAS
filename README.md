# AEIC-Stackelberg: Game-Theoretic Decentralized Semantic Consensus for Multi-Agent Systems

## Overview

A multi-agent semantic consensus framework that combines structured reasoning decomposition (AEIC) with Stackelberg game-theoretic incentives. Agents independently generate structured reasoning (Assumption-Evidence-Inference-Conclusion), which is evaluated via four trust dimensions (IC/EC/EQ/EV), then aggregated through a Stackelberg game to produce reliable collective decisions.

Key innovations:
- **AEIC Decomposition**: Decomposes reasoning chains into four orthogonal semantic layers for precise conflict attribution
- **Four-Component Trust Evaluation**: IC (inference-conclusion consistency), EC (evidence-conclusion support), EQ (evidence quality), EV (cross-agent evidence consensus)
- **Stackelberg Game Consensus**: Leader-Follower incentive structure with pure theta-normalized voting
- **Six Communication Modes**: Parallel, Sequential, Star, Debate, SelfCorrect, AgentVerse
- **17 Learnable Parameters**: Data-driven consensus optimization via differential evolution at zero API cost
- **Heterogeneous LLM Backends**: DeepSeek (strong) + GLM-4-Flash (weak/free) with round-robin assignment

## Project Structure

```
MAS/
├── core/                              # Core algorithm modules
│   ├── similarity.py                  # Similarity computation (Embedding/BM25/TF-IDF)
│   ├── consensus.py                   # AEIC consensus engine + adaptive weight learning
│   ├── stackelberg.py                 # Stackelberg game (Leader-Follower + energy convergence)
│   ├── trust.py                       # Trust evaluation (IC/EC/EQ/EV) + bid construction
│   ├── consensus_config.py            # 17-parameter ConsensusConfig dataclass
│   ├── llm.py                         # Multi-LLM pool (DeepSeek + GLM-4-Flash)
│   ├── orchestrator.py                # Mode-aware AEIC orchestrator
│   └── topology.py                    # Network topology definitions
│
├── experiments/                       # Experiment suite
│   ├── dataset_loader.py              # Dataset loading (10 benchmarks -> AEIC format)
│   ├── collect_cache.py               # Phase 1: AEIC cache collection + Phase 2: signal precomputation
│   ├── optimize_consensus.py          # Parameter optimization via differential evolution
│   ├── exp_orchestrator.py            # Full orchestrator experiment runner
│   ├── exp8_topology.py               # Topology analysis experiment
│   ├── exp9_cross_dataset.py          # Cross-dataset generalization experiment
│   ├── run_all_cache.sh               # Batch cache collection for all datasets
│   └── run_all.py                     # Run all experiments
│
├── results/                           # Cached AEIC outputs and precomputed signals
│   ├── aeic_cache_{dataset}_5a.json   # Raw AEIC outputs per agent per question
│   ├── signals_{dataset}_5a.json      # Precomputed IC/EC/EQ/EV/stance/agreement
│   └── optimized_config.json          # Optimized 17-parameter configuration
│
├── figures/                           # Generated figures (PDF + PNG)
├── papers/                            # Reference papers
│   ├── 2605.00914v1.pdf               # The Cost of Consensus (Bertalanic & Fortuna, 2026)
│   └── 2308.10848v3.pdf               # AgentVerse (Chen et al., 2023)
│
├── papers.tex                         # English paper (37 pages)
├── papers.pdf                         # Compiled English paper
├── 论文.tex                            # Chinese paper (39 pages)
├── 论文.pdf                            # Compiled Chinese paper
└── requirements.txt                   # Python dependencies
```

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Cache Collection Pipeline (Two-Phase)

```bash
# Phase 1: Collect AEIC outputs from LLM APIs (200 questions x 5 agents)
python experiments/collect_cache.py --dataset StrategyQA --n_samples 200 --n_agents 5

# Phase 2: Precompute trust signals (local embedding, no API cost)
python experiments/collect_cache.py --precompute results/aeic_cache_StrategyQA_5a.json

# Batch all datasets
bash experiments/run_all_cache.sh
```

### Parameter Optimization (Zero API Cost)

```bash
python experiments/optimize_consensus.py --signals results/signals_StrategyQA_5a.json
```

### Run Orchestrator Experiments

```bash
python experiments/exp_orchestrator.py --dataset StrategyQA --mode parallel --n_agents 5
```

## Core Architecture

### AEIC Semantic Decomposition

Each agent's reasoning chain R is decomposed into four orthogonal layers:
- **A** (Assumption): Premises and problem framing
- **E** (Evidence): Supporting facts and data points
- **I** (Inference): Reasoning process from evidence to conclusion
- **C** (Conclusion): Final judgment (Yes/No with justification)

### Trust Evaluation (IC/EC/EQ/EV)

| Component | What it measures | Detection target |
|-----------|-----------------|------------------|
| IC | Inference-Conclusion directional consistency | Sycophantic flips (conclusion changed but reasoning didn't) |
| EC | Evidence-Conclusion semantic support | Fabricated conclusions unsupported by evidence |
| EQ | Evidence quality (count, length, density, diversity) | Weak model low-quality evidence |
| EV | Cross-agent evidence consensus | Anomalous/fabricated evidence |

### Stackelberg Game -> Pure theta Voting

Trust signals (IC/EC/EQ/EV) feed into bid construction -> Stackelberg game computes equilibrium theta* -> theta normalized as sole voting weight. No double-counting of trust signals.

### Six Communication Modes

| Mode | Information Flow | API Calls | Baseline Source |
|------|-----------------|-----------|-----------------|
| Parallel | Independent -> consensus | N | - |
| Sequential | Chain: C_{i-1} -> A_i | N | - |
| Star | Leaves parallel -> Hub synthesis | N+1 | - |
| Debate | Multi-round peer exchange | N x R | - |
| SelfCorrect | Independent self-revision | N x R | Bertalanic & Fortuna, 2026 |
| AgentVerse | Solver -> Reviewers -> Revision | N+1 | Chen et al., 2023 |

## Datasets (10 Benchmarks)

| Dataset | Domain | Difficulty | Questions |
|---------|--------|-----------|-----------|
| StrategyQA | Multi-step implicit reasoning | Medium | 200 |
| GSM8K | Elementary math | Medium | 200 |
| GSM-Hard | Math (large numbers/traps) | Hard | 200 |
| MMLU | Multi-subject knowledge | Medium | 200 |
| MMLU-Hard | Hard STEM subset | Hard | 200 |
| TruthfulQA | Factual truthfulness | Medium | 200 |
| CommonsenseQA | Commonsense reasoning | Medium | 200 |
| ARC-Challenge | Science reasoning | Hard | 200 |
| HotpotQA | Multi-hop reasoning | Medium | 200 |
| GPQA | Graduate-level science | Hard | 200 |

## LLM Backends

- **DeepSeek** (deepseek-chat): Strong reasoning model, paid API
- **GLM-4-Flash** (glm-4-flash): Weak reasoning model, free API
- Round-robin assignment: Agent 0 -> DeepSeek, Agent 1 -> GLM-4-Flash, Agent 2 -> DeepSeek, ...

## Paper Compilation

```bash
# English paper
pdflatex papers.tex

# Chinese paper (requires CTeX)
xelatex 论文.tex
```
