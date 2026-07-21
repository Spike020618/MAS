import json

with open("results/cache/MMLU-Hard_star.json", "r") as f:
    data = json.load(f)
    print("Keys in top level:", data.keys())
    if "questions" in data and len(data["questions"]) > 0:
        q = data["questions"][0]
        print("Keys in question:", q.keys())
        if "agents" in q and len(q["agents"]) > 0:
            print("Keys in agent:", q["agents"][0].keys())
        if "group_conclusion" in q:
            print("group_conclusion:", q["group_conclusion"])
        if "ground_truth" in q:
            print("ground_truth:", q["ground_truth"])
