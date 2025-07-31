import os
import pandas as pd
import requests
from datasets import load_dataset, Dataset
from tqdm import tqdm
import time
import json
import re

# --- Constants ---
DATASET_NAME = "SynthLabsAI/Big-Math-RL-Verified"
DATASET_SPLIT = "train"
NUM_PROBLEMS = 1

# --- OpenRouter API Settings ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "deepseek/deepseek-r1-0528:free"
YOUR_SITE_URL = "http://localhost"
APP_NAME = "BigMath Evolver"

# ★★★ Hugging Face Upload Settings ★★★
# ここは後で変更する必要あり
OUTPUT_DATASET_ID = "Man-snow/evolved-math-problems-from-deepseek-r1-0528-free"


UPWARD_EVOLUTION_PROMPT_TEMPLATE = """
You are an expert in creating complex mathematical problems. Your task is to rewrite the given instruction to make it more challenging.

#Instruction#
{problem}

Follow these steps precisely.
Step 1: Understand the core concept and structure of the "#Instruction#". Identify the key elements such as variables, conditions, participants, actions, or processes that can be manipulated to increase complexity. Also, recognize the theme of the instruction and ensure it remains consistent throughout the evolution.
Step 2: Formulate a comprehensive plan to increment the complexity of the "#Instruction#" based on the identified elements in Step 1. The plan should involve modifying or expanding at least three components from the list. It is crucial to ensure that all components in the instruction are logically interconnected and that the complexity increase is coherent and justified. The plan should avoid introducing variables or conditions without clear criteria for determining their values or without contributing to the overall complexity. In this step, consider adding more real-world constraints and dependencies between variables to make the problem more challenging. And you can also add more constraints, concretizing, increasing reasoning.
Step 3: Implement the plan step by step to create the "#Rewritten Instruction#". Ensure the rewritten instruction maintains a logical sequence and avoids ambiguity or confusion. If additional variables or conditions are introduced, provide clear and unambiguous methods or criteria for determining their values. The "#Rewritten Instruction#" should not exceed the original "#Instruction#" by more than 30 words to ensure readability and comprehension.
Step 4: Review the "#Rewritten Instruction#" thoroughly to identify any unreasonable elements or inconsistencies. Make sure the "#Rewritten Instruction#" is a more complex version of the "#Instruction#". and that it accurately reflects the intended increase in complexity. Adjust any part of the instruction that may lead to misunderstanding or ambiguity, and provide the "#Finally Rewritten Instruction#" without any supplementary explanation.
Please reply strictly in the following format:
Step 1
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

def get_sorted_problems():
    """Loads and sorts the dataset from Hugging Face."""
    print("🔄 Loading dataset from Hugging Face...")
    try:
        dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
        df = dataset.to_pandas()
        print(f"✅ Dataset loaded. Total {len(df)} problems.")
        
        print("🔃 Sorting problems...")
        df['llama8b_solve_rate'] = pd.to_numeric(df['llama8b_solve_rate'], errors='coerce')
        df.dropna(subset=['llama8b_solve_rate'], inplace=True)
        
        sorted_df = df.sort_values(by=['llama8b_solve_rate', 'problem'], ascending=[True, True])
        
        print(f"✅ Sort complete. Getting top {NUM_PROBLEMS} problems.")
        return sorted_df.head(NUM_PROBLEMS)
        
    except Exception as e:
        print(f"❌ Error getting or sorting dataset: {e}")
        return None

def evolve_problem_with_openrouter(problem_text: str) -> tuple[str, str]:
    """Calls the OpenRouter API to evolve a problem statement."""
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

        except json.JSONDecodeError as e:
            last_error = f"❌ Invalid API response format (JSONDecodeError): {e}"
        except requests.exceptions.RequestException as e:
            last_error = f"❌ API Request Error: {e}"
        except Exception as e:
            last_error = f"❌ An unknown error occurred: {e}"

        print(f"  (Attempt {attempt + 1}/3) API request failed. Retrying in 1 second...")
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
    problems_df = get_sorted_problems()
    
    if problems_df is None:
        return

    results = []
    print(f"\n🚀 Starting upward evolution for {len(problems_df)} problems...")

    for index, row in tqdm(problems_df.iterrows(), total=len(problems_df), desc="Processing Problems"):
        original_problem = row['problem']
        
        start_time = time.time()
        status, evolved_response = evolve_problem_with_openrouter(original_problem)
        end_time = time.time()
        processing_time = end_time - start_time
        
        evolved_problem = ""
        if status == 'success':
            evolved_problem = parse_final_instruction(evolved_response)
        
        results.append({
            "original_problem": original_problem,
            "evolved_problem": evolved_problem,
            "evolved_response": evolved_response,
            "status": status,
            "processing_time_seconds": round(processing_time, 2),
            "llama8b_solve_rate": row['llama8b_solve_rate'],
            "original_solution": row['answer']
        })
        
        time.sleep(1)

    # --- Save results to CSV and upload to Hugging Face Hub ---
    
    # 1. Convert to DataFrame and set column order
    results_df = pd.DataFrame(results)
    column_order = [
        "original_problem", "evolved_problem", "evolved_response", "status",
        "processing_time_seconds", "llama8b_solve_rate", "original_solution"
    ]
    final_columns = [col for col in column_order if col in results_df.columns]
    results_df = results_df[final_columns]
    
    # 2. Save to local CSV file
    output_filename = "evolved_math_problems.csv"
    results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n✅ Results saved locally to '{output_filename}'")

    # ★★★ 3. Upload to Hugging Face Hub ★★★
    try:
        print(f"🚀 Uploading dataset to Hugging Face Hub: '{OUTPUT_DATASET_ID}'...")
        hf_dataset = Dataset.from_pandas(results_df)
        hf_dataset.push_to_hub(
            repo_id=OUTPUT_DATASET_ID,
            private=False  # Creates the dataset as private
        )
        print(f"✅ Successfully uploaded dataset to '{OUTPUT_DATASET_ID}'.")
    except Exception as e:
        print(f"❌ Failed to upload to Hugging Face Hub: {e}")
        print("  Please ensure you are logged in (`huggingface-cli login`) and the repo_id is correct.")

    print("\n--- Processing complete ---")

if __name__ == "__main__":
    main()