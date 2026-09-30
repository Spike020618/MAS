import json
import os

cache_dir = "experiments/results/cache"
topologies = ["parallel", "sequential", "star", "debate", "hierarchical"]
datasets = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]

for dataset in datasets:
    questions = []
    # Load the base questions
    with open(os.path.join(cache_dir, f"{dataset}.json"), 'r') as f:
        data = json.load(f)
        questions = data.get("questions", [])
        print(f"{dataset}: Loaded {len(questions)} base questions")
    
    # We need to find which topology got it right
    # Let's check the cache files for each topology
    for topo in topologies:
        topo_file = os.path.join(cache_dir, f"{dataset}_{topo}.json")
        if os.path.exists(topo_file):
            with open(topo_file, 'r') as f:
                topo_data = json.load(f)
                correct_count = 0
                for q in topo_data.get("questions", []):
                    if q.get("is_correct", False):
                        correct_count += 1
                print(f"  {topo}: {correct_count}/{len(topo_data.get('questions', []))} correct")
