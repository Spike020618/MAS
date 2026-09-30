import json
import os
import sys
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_zero_shot_experiment():
    cache_dir = "experiments/results/cache"
    
    if not os.path.exists(cache_dir):
        print(f"Directory {cache_dir} not found.")
        return
        
    topologies = ["parallel", "sequential", "star", "debate", "hierarchical"]
    datasets = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]
    
    X_texts = []
    y_labels = []
    
    for dataset in datasets:
        # Load base questions
        base_file = os.path.join(cache_dir, f"{dataset}.json")
        if not os.path.exists(base_file):
            continue
            
        with open(base_file, 'r') as f:
            base_data = json.load(f)
            questions = base_data.get("questions", [])
            
        # Map ID to question text
        q_texts = {q.get("question_id", i): q.get("question", "") for i, q in enumerate(questions)}
        
        # Track accuracy per question across topologies
        q_correct_by_topo = {q_id: {} for q_id in q_texts.keys()}
        
        for topo in topologies:
            topo_file = os.path.join(cache_dir, f"{dataset}_{topo}.json")
            if not os.path.exists(topo_file):
                continue
                
            with open(topo_file, 'r') as f:
                topo_data = json.load(f)
                
            for i, q in enumerate(topo_data.get("questions", [])):
                q_id = q.get("question_id", i)
                gt = str(q.get("ground_truth", "")).lower().strip()
                
                # Determine correct based on majority vote
                yes_votes = 0
                no_votes = 0
                for a in q.get("agents", []):
                    c = a.get("aeic", {}).get("C", "").lower()
                    if "yes" in c and "no" not in c:
                        yes_votes += 1
                    elif "no" in c and "yes" not in c:
                        no_votes += 1
                    elif c.strip().startswith("yes"):
                        yes_votes += 1
                    elif c.strip().startswith("no"):
                        no_votes += 1
                            
                pred = "yes" if yes_votes > no_votes else "no"
                is_correct = (pred == gt)
                q_correct_by_topo[q_id][topo] = is_correct
                
        # Determine best topology for each question
        for q_id, topo_results in q_correct_by_topo.items():
            correct_topos = [topo for topo, is_corr in topo_results.items() if is_corr]
            
            # If at least one topology gets it right, pick one as target
            if correct_topos and q_texts[q_id]:
                # In case of ties, prefer faster/cheaper topologies: parallel > star > seq > hierarchical > debate
                topo_priority = {"parallel": 1, "star": 2, "sequential": 3, "hierarchical": 4, "debate": 5}
                best_topo = min(correct_topos, key=lambda t: topo_priority.get(t, 99))
                
                X_texts.append(q_texts[q_id])
                y_labels.append(best_topo)
                
    print(f"Extracted {len(X_texts)} valid questions with optimal topologies.")
    
    if len(X_texts) < 10:
        print("Not enough data to train. Exiting.")
        return
        
    # Extract features (TF-IDF + Length)
    print("Extracting features and training model...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_tfidf = vectorizer.fit_transform(X_texts).toarray()
    
    X_lengths = np.array([len(t) for t in X_texts]).reshape(-1, 1)
    X_words = np.array([len(t.split()) for t in X_texts]).reshape(-1, 1)
    
    X_features = np.hstack([X_tfidf, X_lengths, X_words])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_labels, test_size=0.2, random_state=42
    )
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples...")
    
    # Train router
    classifier = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
    classifier.fit(X_train, y_train)
    
    # Evaluate
    predictions = classifier.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    print("\n--- Predictive Zero-Shot Router Evaluation ---")
    print(f"Accuracy (Predicting Optimal Topology): {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))
    
    # Baselines
    majority_class = max(set(y_train), key=y_train.count)
    majority_predictions = [majority_class] * len(y_test)
    majority_acc = accuracy_score(y_test, majority_predictions)
    print(f"\nBaseline: Always pick '{majority_class}': {majority_acc:.4f}")
    
    if acc > majority_acc:
        print(f"Success! The zero-shot router learns meaningful text features (+{(acc - majority_acc)*100:.1f}pp).")
    else:
        print(f"Note: Text features alone may not be sufficient for optimal routing without AEIC signals.")

if __name__ == "__main__":
    run_zero_shot_experiment()
