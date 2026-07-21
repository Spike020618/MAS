import json

# How does eval_generalization.py evaluate these files? Let's check its logic.
with open("experiments/dataset_loader.py", "r") as f:
    for line in f.readlines():
        if "correct" in line or "score" in line:
            # Not helpful
            pass
