import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_generative_experiment():
    cache_dir = "experiments/results/cache"
    datasets = ["GSM8K", "GSM-Hard"]
    
    print("=== Direction 3: Generative Consensus Evaluation (Yes/No Format) ===\n")
    
    for dataset in datasets:
        file_path = os.path.join(cache_dir, f"{dataset}.json")
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        questions = data.get("questions", [])
        print(f"Dataset: {dataset} ({len(questions)} questions)")
        
        single_agent_correct = 0
        aeic_gen_correct = 0
        unstructured_correct = 0
        total_valid = 0
        
        for q in questions:
            gt = str(q.get("ground_truth", "")).lower().strip()
            
            answers_c = []
            answers_unstructured = []
            
            for a in q.get("agents", []):
                aeic = a.get("aeic", {})
                c_val = str(aeic.get("C", "")).lower()
                answers_c.append(c_val)
                unstructured = str(aeic.get("I", "")).lower() + " " + c_val
                answers_unstructured.append(unstructured)
                
            if not answers_c:
                continue
                
            total_valid += 1
            
            # Helper to extract yes/no
            def get_vote(text):
                if 'yes' in text and 'no' not in text: return 'yes'
                if 'no' in text and 'yes' not in text: return 'no'
                if text.strip().startswith('yes'): return 'yes'
                if text.strip().startswith('no'): return 'no'
                return 'unknown'
                
            # 1. Single Agent
            sa_pred = get_vote(answers_c[0])
            if sa_pred == gt:
                single_agent_correct += 1
                
            # 2. AEIC Structured
            yes_c = sum(1 for a in answers_c if get_vote(a) == 'yes')
            no_c = sum(1 for a in answers_c if get_vote(a) == 'no')
            aeic_pred = 'yes' if yes_c > no_c else 'no' if no_c > yes_c else 'unknown'
            if aeic_pred == gt:
                aeic_gen_correct += 1
                
            # 3. Unstructured
            yes_u = sum(1 for a in answers_unstructured if get_vote(a) == 'yes')
            no_u = sum(1 for a in answers_unstructured if get_vote(a) == 'no')
            u_pred = 'yes' if yes_u > no_u else 'no' if no_u > yes_u else 'unknown'
            if u_pred == gt:
                unstructured_correct += 1
                
        if total_valid > 0:
            print(f"  Single Agent Acc:        {single_agent_correct / total_valid * 100:.1f}%")
            print(f"  Unstructured Agg Acc:    {unstructured_correct / total_valid * 100:.1f}%")
            print(f"  AEIC Structured Acc:     {aeic_gen_correct / total_valid * 100:.1f}%")
            print(f"  Gap (AEIC vs Unstruct):  +{(aeic_gen_correct - unstructured_correct) / total_valid * 100:.1f}pp\n")
            
if __name__ == "__main__":
    run_generative_experiment()
