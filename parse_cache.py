import os
import json

cache_dir = "results/cache"
found_one = False
for ds in os.listdir(cache_dir):
    ds_path = os.path.join(cache_dir, ds)
    if os.path.isdir(ds_path):
        for qfile in os.listdir(ds_path):
            if qfile.endswith(".json"):
                qpath = os.path.join(ds_path, qfile)
                with open(qpath, 'r') as f:
                    data = json.load(f)
                    print(f"File: {qpath}")
                    print(f"Keys: {list(data.keys())}")
                    if "question" in data:
                        print(f"Question: {data['question'][:100]}...")
                    if "text" in data:
                        print(f"Text: {data['text'][:100]}...")
                found_one = True
                break
    if found_one:
        break
