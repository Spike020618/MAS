import json

with open("results/all_baselines.json", "r") as f:
    data = json.load(f)

for ds in data:
    for topo in data[ds]:
        if "AEIC_default" in topo:
            print(f"Dataset: {ds}, Topo: {topo}")
            print(type(data[ds][topo]))
            if isinstance(data[ds][topo], dict):
                print(data[ds][topo].keys())
            break
    break
