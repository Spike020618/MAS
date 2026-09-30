import json
import os

cache_dir = "experiments/results/cache"
topologies = ["parallel", "sequential", "star", "debate", "hierarchical"]
datasets = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]

# Use simple majority voting string match as a fallback
for dataset in datasets:
    with open(os.path.join(cache_dir, f"{dataset}.json"), 'r') as f:
        data = json.load(f)
        questions = data.get("questions", [])
        print(f"{dataset}: Loaded {len(questions)} base questions")
        
    for topo in topologies:
        topo_file = os.path.join(cache_dir, f"{dataset}_{topo}.json")
        if os.path.exists(topo_file):
            with open(topo_file, 'r') as f:
                topo_data = json.load(f)
                correct_count = 0
                for q in topo_data.get("questions", []):
                    gt = str(q.get("ground_truth", "")).lower().strip()
                    
                    yes_votes = 0
                    no_votes = 0
                    for a in q.get("agents", []):
                        c = a.get("aeic", {}).get("C", "").lower()
                        # Simple extraction
                        if "yes" in c and "no" not in c:
                            yes_votes += 1
                        elif "no" in c and "yes" not in c:
                            no_votes += 1
                        elif "yes" in c: # First word fallback
                            if c.strip().startswith("yes"):
                                yes_votes += 1
                            elif c.strip().startswith("no"):
                                no_votes += 1
                                
                    pred = "yes" if yes_votes > no_votes else "no"
                    if pred == gt:
                        correct_count += 1
                        
                print(f"  {topo}: {correct_count}/{len(topo_data.get('questions', []))} correct")
