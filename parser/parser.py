import json
import re
from pathlib import Path
import anthropic

def interpret_and_align_proofs(lean_path: Path, md_path: Path, api_key: str):
    """
    Independent processor to align formal Lean code with natural language Markdown.
    Uses the Anthropic API directly.
    """
    
    # 1. Initialize Anthropic Client
    client = anthropic.Anthropic(api_key=api_key)
    
    # 2. Load the source files
    if not (lean_path.exists() and md_path.exists()):
        raise FileNotFoundError("Could not find PROOF.lean or PROOF.md at the provided paths.")

    lean_text = lean_path.read_text()
    md_text = md_path.read_text()

    # 3. Define the Prompting Logic
    system_prompt = (
        "You are a Mathematical Logic Analyst specializing in Lean 4. "
        "Your goal is to decompose a proof into logical objects that align formal "
        "tactic code with natural language prose."
    )

    user_message = f"""
I have a verified Lean 4 proof and its natural language explanation.

[NATURAL LANGUAGE PROOF (PROOF.md)]
{md_text}

[LEAN 4 FORMAL PROOF (PROOF.lean)]
{lean_text}

TASK:
Decompose these into a sequence of 'Logical Objects' that show the correspondence between the two.

CRITERIA:
- Identify 'Primary' objects: High-level logical arguments (e.g., major 'have' blocks, induction steps).
- Identify 'Secondary' objects: Technical tactics (e.g., simp, ring, linarith, rw) or boilerplate.

OUTPUT FORMAT:
Return ONLY a JSON list. No preamble.
[
  {{
    "step_name": "Title",
    "importance": "Primary" | "Secondary",
    "informal_text": "Sentence(s) from PROOF.md",
    "formal_code": "The corresponding Lean code block",
    "explanation": "Mathematical insight for this step."
  }}
]
"""

    # 4. API Call
    print(f"Calling Anthropic to align {lean_path.name}...")
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620", # Or claude-3-opus-20240229
        max_tokens=4000,
        temperature=0,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

    raw_result = message.content[0].text

    # 5. Extract JSON and Generate Alignment Document
    json_match = re.search(r'\[\s*{.*}\s*\]', raw_result, re.DOTALL)
    if not json_match:
        print("Error: Could not find valid JSON array in response.")
        return None

    try:
        alignment_data = json.loads(json_match.group(0))
        output_path = lean_path.parent / "PROOF_ALIGNMENT.md"
        
        with open(output_path, "w") as f:
            f.write("# Proof Interpretation & Alignment Map\n\n")
            f.write("Generated via Anthropic Claude-3.5-Sonnet.\n\n")
            
            for item in alignment_data:
                header = "## 🟢" if item['importance'] == 'Primary' else "### ⚪"
                f.write(f"{header} {item['step_name']} ({item['importance']})\n")
                f.write(f"**Insight:** {item['explanation']}\n\n")
                f.write(f"**Prose:**\n> {item['informal_text']}\n\n")
                f.write(f"**Code:**\n```lean\n{item['formal_code']}\n```\n")
                f.write("---\n\n")
        
        print(f"Success! Alignment saved to {output_path}")
        return alignment_data

    except json.JSONDecodeError:
        print("JSON Decode Error. Check the raw response.")
        return None
    
if __name__ == "__main__":
    my_api_key = "your-anthropic-api-key-here"
    path_to_results = Path("./experiments/run_01/") 
    
    interpret_and_align_proofs(
        lean_path = path_to_results / "PROOF.lean",
        md_path   = path_to_results / "PROOF.md",
        api_key   = my_api_key
    )