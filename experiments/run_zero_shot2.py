import json
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import openai

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def compute_end_to_end_accuracy(y_pred, q_correct_by_topo, q_ids_test):
    correct = 0
    total = len(y_pred)
    for i, pred_topo in enumerate(y_pred):
        q_id = q_ids_test[i]
        if q_correct_by_topo[q_id].get(pred_topo, False):
            correct += 1
    return correct / total if total > 0 else 0

def run_zero_shot_experiment():
    cache_dir = "experiments/results/cache"
    topologies = ["parallel", "sequential", "star", "debate", "hierarchical"]
    datasets = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]
    
    X_texts = []
    y_labels = []
    q_ids = []
    
    # Store evaluation data globally to compute true accuracy
    global_q_correct_by_topo = {}
    
    dataset_offset = 0
    for dataset in datasets:
        base_file = os.path.join(cache_dir, f"{dataset}.json")
        if not os.path.exists(base_file):
            continue
            
        with open(base_file, 'r') as f:
            base_data = json.load(f)
            questions = base_data.get("questions", [])
            
        q_texts = {i + dataset_offset: q.get("question", "") for i, q in enumerate(questions)}
        q_correct_by_topo = {i + dataset_offset: {} for i in range(len(questions))}
        
        for topo in topologies:
            topo_file = os.path.join(cache_dir, f"{dataset}_{topo}.json")
            if not os.path.exists(topo_file):
                continue
                
            with open(topo_file, 'r') as f:
                topo_data = json.load(f)
                
            for i, q in enumerate(topo_data.get("questions", [])):
                gt = str(q.get("ground_truth", "")).lower().strip()
                
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
                q_correct_by_topo[i + dataset_offset][topo] = is_correct
                
        # Merge into global map
        global_q_correct_by_topo.update(q_correct_by_topo)
        
        # Build training data
        for q_id, topo_results in q_correct_by_topo.items():
            correct_topos = [t for t, is_corr in topo_results.items() if is_corr]
            
            # Use all questions for evaluation, even if none are correct
            q_ids.append(q_id)
            X_texts.append(q_texts[q_id])
            
            if correct_topos:
                # Topo priority to break ties
                topo_priority = {"parallel": 1, "star": 2, "sequential": 3, "hierarchical": 4, "debate": 5}
                best_topo = min(correct_topos, key=lambda t: topo_priority.get(t, 99))
                y_labels.append(best_topo)
            else:
                y_labels.append("star") # default fallback
                
        dataset_offset += len(questions)
                
    # Train-test split
    indices = np.arange(len(X_texts))
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        indices, y_labels, test_size=0.2, random_state=42
    )
    
    # Text features
    X_train_texts = [X_texts[i] for i in X_train_idx]
    X_test_texts = [X_texts[i] for i in X_test_idx]
    test_q_ids = [q_ids[i] for i in X_test_idx]
    
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train_texts).toarray()
    X_test_tfidf = vectorizer.transform(X_test_texts).toarray()
    
    classifier = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
    classifier.fit(X_train_tfidf, y_train)
    
    predictions = classifier.predict(X_test_tfidf)
    
    # Compute end-to-end accuracies
    # 1. Zero-shot router
    zs_acc = compute_end_to_end_accuracy(predictions, global_q_correct_by_topo, test_q_ids)
    
    # 2. Fixed topologies
    fixed_accs = {}
    for topo in topologies:
        fixed_preds = [topo] * len(predictions)
        fixed_accs[topo] = compute_end_to_end_accuracy(fixed_preds, global_q_correct_by_topo, test_q_ids)
        
    best_fixed = max(fixed_accs.items(), key=lambda x: x[1])
    
    # 3. Oracle
    oracle_correct = 0
    for q_id in test_q_ids:
        if any(global_q_correct_by_topo[q_id].values()):
            oracle_correct += 1
    oracle_acc = oracle_correct / len(test_q_ids)
    
    print("\n=== Predictive Zero-Shot Routing Results ===")
    print(f"Test Set Size: {len(test_q_ids)} questions")
    print(f"Oracle Upper Bound: {oracle_acc:.4f}")
    print(f"Best Fixed Topology ({best_fixed[0]}): {best_fixed[1]:.4f}")
    print(f"Zero-Shot Router: {zs_acc:.4f}")
    
    if zs_acc > best_fixed[1]:
        print(f"\nSUCCESS: Zero-Shot Router outperforms best fixed topology by {(zs_acc - best_fixed[1])*100:.1f}pp!")
    else:
        print(f"\nNote: Zero-Shot Router ({zs_acc:.4f}) does not outperform fixed '{best_fixed[0]}' ({best_fixed[1]:.4f}).")
        print("This highlights why our original AEIC signal-based adaptive routing (Section 5) is necessary.")

if __name__ == "__main__":
    run_zero_shot_experiment()
