import json

with open("results/router_evaluation.json", "r") as f:
    data = json.load(f)

for key in data.keys():
    print(f"Key: {key}")
    if isinstance(data[key], list) and len(data[key]) > 0:
        first_item = data[key][0]
        print(f"  First item keys: {list(first_item.keys())}")
        print(f"  Sample text: {str(first_item.get('question_text', ''))[:50]}...")
    elif isinstance(data[key], dict):
        print(f"  Keys: {list(data[key].keys())}")
        
import os
for f in os.listdir("results"):
    if f.endswith(".json"):
        print(f"File: {f}")
