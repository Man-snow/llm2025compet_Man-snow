import os
import pandas as pd
import requests
from datasets import Dataset
from tqdm import tqdm
import time
import json
import re

# --- Constants ---
# ★★★ New dataset URL ★★★
JSONL_URL = "https://raw.githubusercontent.com/tana114/vllm-api-structured/main/project/olym/data/TP_TO_maths_en_COMP.jsonl"
NUM_PROBLEMS = 5 # Number of problems to process

# --- OpenRouter API Settings ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "deepseek/deepseek-r1-0528:free"
YOUR_SITE_URL = "http://localhost"
APP_NAME = "Math Problem Evolver"

# --- Hugging Face Upload Settings ---
OUTPUT_DATASET_ID = "Man-snow/evolved-math-problems-OlympiadBench-from-deepseek-r1-0528-free"

# ★★★ Updated Prompt Template ★★★
UPWARD_EVOLUTION_PROMPT_TEMPLATE = """
You are an expert in mathematical problem design. Your primary task is to transform proof-based problems into computational problems. If the problem is already computational, your task is to make it more challenging.

#Instruction#
{problem}

Follow these steps precisely.
Step 1: First, determine if the "#Instruction#" is a "proof problem" (e.g., contains "Prove that...", "Show that...") or a "computational problem" (e.g., asks "Find...", "What is...").
- If it is a proof problem: Your main goal is to reformulate it into a computational problem that uses the same core mathematical concepts but asks for a specific value, formula, or example. This is your priority.
- If it is already a computational problem: Your goal is to make it more challenging.
Based on this goal, identify the key elements such as variables, conditions, or concepts that can be manipulated.

Step 2: Formulate a comprehensive plan.
- For a proof-to-computational transformation: The plan should detail how to introduce parameters or specific scenarios to create a question with a concrete answer.
- For increasing complexity: The plan should involve modifying or expanding at least three components. Consider adding more constraints, dependencies, or real-world context.

Step 3: Implement the plan to create the "#Rewritten Instruction#". The new problem must be solvable and logically sound. Ensure any new variables or conditions are clearly defined. The rewritten instruction should not exceed the original by more than 40 words.

Step 4: Review the "#Rewritten Instruction#" thoroughly. Ensure it fulfills the goal from Step 1 (either transformed or made more complex). Provide the "#Finally Rewritten Instruction#" without any supplementary explanation.

Please reply strictly in the following format:
Step 1
#Goal#: [Transform to Computational / Increase Complexity]
#Elements Identified#:
...
Step 2
#Plan#:
...
Step 3
#Rewritten Instruction#:
...
Step 4
#Finally Rewritten Instruction#:
...
"""

def get_problems_from_jsonl(url: str, num_problems: int):
    """Downloads and parses a .jsonl file from a URL."""
    print(f"🔄 Downloading dataset from {url}...")
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        lines = response.text.strip().splitlines()
        print(f"✅ Download complete. Found {len(lines)} problems.")
        
        problems = []
        for line in lines[:num_problems]:
            try:
                data = json.loads(line)
                # We will use 'question' as the problem text
                if 'id' in data and 'question' in data:
                    problems.append(data)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping a line due to JSON parsing error: {line}")
        
        print(f"✅ Successfully parsed {len(problems)} problems.")
        return pd.DataFrame(problems)

    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading the dataset: {e}")
        return None

def evolve_problem_with_openrouter(problem_text: str) -> tuple[str, str]:
    """Calls the OpenRouter API to evolve a problem statement."""
    # This function remains the same as the previous version
    if not OPENROUTER_API_KEY:
        return "failure", "❌ OPENROUTER_API_KEY environment variable not set."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": YOUR_SITE_URL,
        "X-Title": APP_NAME,
        "Content-Type": "application/json"
    }
    prompt = UPWARD_EVOLUTION_PROMPT_TEMPLATE.format(problem=problem_text)
    data = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}]}
    last_error = ""

    for attempt in range(3):
        try:
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            json_response = response.json()

            if 'choices' in json_response and len(json_response['choices']) > 0:
                content = json_response['choices'][0]['message']['content']
                return "success", content.strip()
            else:
                last_error = f"❌ API response missing valid content. Response: {json_response}"

        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [402, 429]:
                try:
                    error_details = e.response.json().get('error', {}).get('message', '')
                except json.JSONDecodeError:
                    error_details = e.response.text
                final_error_message = f"❌ Possible credit exhaustion or rate limit ({e.response.status_code}): {error_details}"
                return "failure", final_error_message
            else:
                last_error = f"❌ HTTP Error: {e}"
        except Exception as e:
            last_error = f"❌ An unknown error occurred: {e}"
        time.sleep(1)
    return "failure", last_error

def parse_final_instruction(response_text: str) -> str:
    """Extracts the final rewritten instruction from the full API response."""
    match = re.search(r'#Finally Rewritten Instruction#\s*:\s*(.*)', response_text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'#Finally Rewritten Instruction#\s*(.*)', response_text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return "Extraction Failed"


def main():
    """Main execution function."""
    problems_df = get_problems_from_jsonl(JSONL_URL, NUM_PROBLEMS)
    
    if problems_df is None or problems_df.empty:
        print("No problems to process. Exiting.")
        return

    results = []
    print(f"\n🚀 Starting upward evolution for {len(problems_df)} problems...")

    for index, row in tqdm(problems_df.iterrows(), total=len(problems_df), desc="Processing Problems"):
        original_problem = row['question']
        
        start_time = time.time()
        status, evolved_response = evolve_problem_with_openrouter(original_problem)
        end_time = time.time()
        processing_time = end_time - start_time
        
        evolved_problem = ""
        if status == 'success':
            evolved_problem = parse_final_instruction(evolved_response)
        
        results.append({
            "id": row.get('id', 'N/A'),
            "original_problem": original_problem,
            "evolved_problem": evolved_problem,
            "evolved_response": evolved_response,
            "status": status,
            "processing_time_seconds": round(processing_time, 2),
            "original_solution": str(row.get('solution', 'N/A')) # Ensure solution is string
        })
        
        time.sleep(1)
    
    # --- Save results and upload ---
    results_df = pd.DataFrame(results)
    column_order = [
        "id", "original_problem", "evolved_problem", "evolved_response", "status",
        "processing_time_seconds", "original_solution"
    ]
    results_df = results_df[[col for col in column_order if col in results_df.columns]]
    
    output_filename = "evolved_math_problems.csv"
    results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n✅ Results saved locally to '{output_filename}'")

    try:
        print(f"🚀 Uploading dataset to Hugging Face Hub: '{OUTPUT_DATASET_ID}'...")
        hf_dataset = Dataset.from_pandas(results_df)
        hf_dataset.push_to_hub(repo_id=OUTPUT_DATASET_ID, private=True)
        print(f"✅ Successfully uploaded dataset to '{OUTPUT_DATASET_ID}'.")
    except Exception as e:
        print(f"❌ Failed to upload to Hugging Face Hub: {e}")
        print("  Please ensure you are logged in (`huggingface-cli login`) and the repo_id is correct.")

    print("\n--- Processing complete ---")

if __name__ == "__main__":
    main()