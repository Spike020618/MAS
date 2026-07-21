import json

with open("results/adaptive_results.json", "r") as f:
    data = json.load(f)

for ds in data:
    print(f"Dataset: {ds}")
    print(list(data[ds].keys()))
    if "routing_history" in data[ds]:
        print(f"Num history items: {len(data[ds]['routing_history'])}")
        print(f"First item: {data[ds]['routing_history'][0]}")
    break
