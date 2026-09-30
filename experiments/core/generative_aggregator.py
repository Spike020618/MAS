import re
from collections import Counter

def normalize_math_answer(text):
    """Normalize mathematical answers to allow grouping."""
    if not text:
        return ""
    # Extract numbers if present
    numbers = re.findall(r'-?\d+\.?\d*', str(text))
    if numbers:
        return numbers[-1] # Usually the final answer is the last number
    return text.strip().lower()

def aggregate_generative_answers(answers):
    """Aggregates generative answers by majority voting after normalization."""
    if not answers:
        return None
        
    normalized = [normalize_math_answer(a) for a in answers if a]
    if not normalized:
        return None
        
    counts = Counter(normalized)
    majority_answer = counts.most_common(1)[0][0]
    return majority_answer
