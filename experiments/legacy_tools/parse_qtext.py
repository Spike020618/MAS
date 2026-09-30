import json

with open("experiments/results/all_baselines.json", "r") as f:
    data = json.load(f)

for ds, ds_data in data.items():
    if isinstance(ds_data, dict):
        for topo, topo_data in ds_data.items():
            if isinstance(topo_data, list) and len(topo_data) > 0:
                print(f"[{ds}] {topo} sample:")
                keys = list(topo_data[0].keys())
                print(f"Keys: {keys}")
                # Print values of keys that might contain the text
                for k in keys:
                    if k in ['question', 'text', 'prompt', 'q', 'id']:
                        print(f"  {k}: {topo_data[0][k]}")
