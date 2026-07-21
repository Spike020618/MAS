import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.generative_aggregator import normalize_math_answer, aggregate_generative_answers

def test_aggregation():
    # Simulated outputs from 5 agents for a math problem
    outputs = [
        "The answer is 42.",
        "42",
        "Therefore, we get 42 as the result.",
        "I calculate it to be 40.",
        "The final result is 42."
    ]
    
    print("Raw outputs:")
    for o in outputs:
        print(f"  - {o}")
        
    print("\nNormalized:")
    for o in outputs:
        print(f"  - {normalize_math_answer(o)}")
        
    aggregated = aggregate_generative_answers(outputs)
    print(f"\nAggregated majority answer: {aggregated}")
    
    if aggregated == "42":
        print("Success: Generative aggregation works correctly!")

if __name__ == "__main__":
    test_aggregation()
