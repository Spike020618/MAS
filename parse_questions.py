import json
import os

# We need the actual questions to train a zero-shot model.
# The routing_features.npz has AEIC features but not the text itself.
# Let's check where the questions are stored.
for root, dirs, files in os.walk('datasets'):
    for file in files:
        if file.endswith('.json') or file.endswith('.jsonl'):
            print(os.path.join(root, file))
