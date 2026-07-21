import os
import json
import numpy as np

# We already have the npz file with routing_features and labels
# Let's inspect it directly to see if we can use it to map to text
data = np.load('results/routing_features.npz', allow_pickle=True)
print("Keys in NPZ:", data.files)
print(f"y shape: {data['y'].shape}")
print(f"y sample: {data['y'][:20]}")

topos = data['topos']
print(f"Topos: {topos}")

# We don't have the question index in the NPZ file, 
# so we'll need to extract labels from adaptive_results.json or similar
with open('results/adaptive_results.json', 'r') as f:
    adaptive = json.load(f)
    print("Adaptive keys:", adaptive.keys())
    
with open('results/final_results.json', 'r') as f:
    final = json.load(f)
    print("Final keys:", final.keys())
