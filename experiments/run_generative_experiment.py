import json
import os
import sys
from collections import Counter
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.generative_aggregator import normalize_math_answer, aggregate_generative_answers

def run_generative_experiment():
    cache_dir = "results/cache"
    datasets = ["GSM8K", "GSM-Hard"]
    
    print("=== Direction 3: Generative Consensus Evaluation ===\n")
    
    for dataset in datasets:
        file_path = os.path.join(cache_dir, f"{dataset}.json")
        if not os.path.exists(file_path):
            print(f"File {file_path} not found. Skipping.")
            continue
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        questions = data.get("questions", [])
        print(f"Dataset: {dataset} ({len(questions)} questions)")
        
        # We'll compare:
        # 1. Single Agent (Agent 0)
        # 2. Majority Vote on Unstructured Inference (simulated by full AEIC text)
        # 3. AEIC Generative Majority (using C field)
        
        single_agent_correct = 0
        aeic_gen_correct = 0
        unstructured_correct = 0
        total_valid = 0
        
        for q in questions:
            gt_text = q.get("ground_truth", "")
            # In GSM datasets mapped to yes/no, ground_truth is 'yes' or 'no'
            # Let's check how the cache actually looks
            gt_norm = normalize_math_answer(gt_text)
            
            # Since these might be Boolean-formatted verification questions, 
            # let's extract the actual agent answers.
            answers_c = []
            answers_unstructured = []
            
            for a in q.get("agents", []):
                aeic = a.get("aeic", {})
                
                # Try to extract the C field
                c_val = aeic.get("C", "")
                answers_c.append(c_val)
                
                # Simulate unstructured by concatenating I and C
                unstructured = str(aeic.get("I", "")) + " " + str(c_val)
                answers_unstructured.append(unstructured)
                
            if not answers_c:
                continue
                
            total_valid += 1
            
            # 1. Single Agent
            sa_pred = normalize_math_answer(answers_c[0])
            if str(sa_pred).lower() == str(gt_norm).lower():
                single_agent_correct += 1
                
            # 2. AEIC Generative Aggregation
            aeic_pred = aggregate_generative_answers(answers_c)
            if str(aeic_pred).lower() == str(gt_norm).lower():
                aeic_gen_correct += 1
                
            # 3. Unstructured Aggregation
            unstructured_pred = aggregate_generative_answers(answers_unstructured)
            if str(unstructured_pred).lower() == str(gt_norm).lower():
                unstructured_correct += 1
                
        if total_valid > 0:
            print(f"  Single Agent Acc:        {single_agent_correct / total_valid * 100:.1f}%")
            print(f"  Unstructured Agg Acc:    {unstructured_correct / total_valid * 100:.1f}%")
            print(f"  AEIC Generative Acc:     {aeic_gen_correct / total_valid * 100:.1f}%")
            print(f"  Gap (AEIC vs Unstruct):  +{(aeic_gen_correct - unstructured_correct) / total_valid * 100:.1f}pp\n")
            
if __name__ == "__main__":
    run_generative_experiment()
