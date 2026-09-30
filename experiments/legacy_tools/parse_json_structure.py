import json

with open("experiments/results/all_baselines.json", "r") as f:
    data = json.load(f)

for key in data.keys():
    print(f"Dataset: {key}")
    if isinstance(data[key], dict):
        for topo in data[key].keys():
            print(f"  Topology: {topo}")
            if isinstance(data[key][topo], list) and len(data[key][topo]) > 0:
                first_item = data[key][topo][0]
                print(f"    First item keys: {list(first_item.keys())}")
                if "is_correct" in first_item:
                    print(f"    is_correct format: {type(first_item['is_correct'])} = {first_item['is_correct']}")
                if "question" in first_item:
                    print(f"    question sample: {str(first_item['question'])[:50]}...")
            elif isinstance(data[key][topo], dict):
                print(f"    Keys: {list(data[key][topo].keys())}")
