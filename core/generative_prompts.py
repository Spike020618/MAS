def get_generative_prompt(question, topology="parallel", agent_id=None, previous_outputs=None):
    base_prompt = f"""You are participating in a multi-agent reasoning process.
Please solve the following problem and format your output strictly according to the AEIC structure.

Problem: {question}

Format your response exactly as follows:
{{"A": "Assumption - break down the key premises of the problem", 
  "E": ["Evidence 1", "Evidence 2"], 
  "I": "Inference - your step-by-step reasoning", 
  "C": "Conclusion - the final answer ONLY (e.g. a number, equation, or short phrase)"}}

"""
    
    if topology == "sequential" and previous_outputs:
        base_prompt += f"\nPrevious agents have provided these conclusions: {previous_outputs}\nYou may build upon or correct their reasoning."
        
    return base_prompt

def extract_generative_conclusion(aeic_dict):
    """Extracts the generative conclusion, attempting to parse math outputs."""
    if not aeic_dict or "C" not in aeic_dict:
        return None
    return str(aeic_dict["C"]).strip()
