import os
import time
import pandas as pd
from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams
import re

# --- 定数設定 ---

# 処理に利用するモデル
MODEL_ID = "deepseek-ai/DeepSeek-R1-0528"
# 入力データセット
SOURCE_DATASET_ID = "SynthLabsAI/Big-Math-RL-Verified"
# 出力データセット（!!ご自身のHugging Faceユーザー名に書き換えてください!!）
OUTPUT_DATASET_ID = "Man-snow/evolved-big-math-rl"
# Hugging Faceトークンを読み込むための環境変数名
HF_TOKEN_ENV = "HUGGING_FACE_TOKEN"

# 問題を上方修正するためのプロンプト
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

def parse_final_instruction(text: str) -> str | None:
    """
    モデルの出力から"#Finally Rewritten Instruction#"の部分を抽出する。
    見つからない場合はNoneを返す。
    """
    match = re.search(r"#Finally Rewritten Instruction#:\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def main():
    """
    メインの処理を実行する関数
    """
    # --- 1. Hugging Faceデータセットの準備 ---
    print("1. Loading and sorting dataset...")
    try:
        dataset = load_dataset(SOURCE_DATASET_ID, split="train")
        df = dataset.to_pandas()
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # llama8b_solve_rateの昇順、problemのアルファベット昇順でソート
    sorted_df = df.sort_values(by=["llama8b_solve_rate", "problem"], ascending=[True, True])
    
    # 上位5問を取得
    problems_to_process = sorted_df.head(5)
    print(f"Selected {len(problems_to_process)} problems to process.")

    # --- 2. vLLMモデルの初期化 ---
    print("2. Initializing VLLM model...")
    # 計算ノードのGPU数に応じてtensor_parallel_sizeを調整してください (例: 8)
    try:
        llm = LLM(model=MODEL_ID, tensor_parallel_size=8, trust_remote_code=True)
        sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)
    except Exception as e:
        print(f"Failed to initialize VLLM: {e}")
        return

    # --- 3. プロンプトの生成とモデルによる処理 ---
    print("3. Generating prompts and processing with the model...")
    prompts = [
        UPWARD_EVOLUTION_PROMPT_TEMPLATE.format(problem=row["problem"])
        for _, row in problems_to_process.iterrows()
    ]

    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    
    total_elapsed_time = end_time - start_time
    print(f"Finished processing in {total_elapsed_time:.2f} seconds.")

    # --- 4. 結果の集計 ---
    print("4. Aggregating results...")
    results = []
    for i, output in enumerate(outputs):
        original_problem = output.prompt.split("#Instruction#\n")[1].split("\n\nFollow these steps precisely.")[0].strip()
        generated_text = output.outputs[0].text
        
        evolved_problem = parse_final_instruction(generated_text)
        
        # 処理時間とトークン数はリクエスト毎に正確に取るのが難しいため、全体の平均を記録
        avg_time_per_problem = total_elapsed_time / len(outputs) if len(outputs) > 0 else 0
        
        results.append({
            "original_problem": original_problem,
            "evolved_problem": evolved_problem,
            "total_tokens": output.outputs[0].cumulative_logprobs is not None and len(output.outputs[0].cumulative_logprobs) or 0,
            "elapsed_time_avg": avg_time_per_problem,
            "success": evolved_problem is not None,
            "full_model_output": generated_text # デバッグ用に完全な出力も保存
        })

    # --- 5. Hugging Face Hubへのアップロード ---
    print("5. Uploading results to Hugging Face Hub...")
    hf_token = os.getenv(HF_TOKEN_ENV)
    if not hf_token:
        print(f"Error: Hugging Face token not found. Please set the '{HF_TOKEN_ENV}' environment variable.")
        return

    try:
        # 結果をDatasetオブジェクトに変換
        result_dataset = Dataset.from_pandas(pd.DataFrame(results))
        
        # Hugging Face Hubにプッシュ
        result_dataset.push_to_hub(
            repo_id=OUTPUT_DATASET_ID,
            token=hf_token,
            private=True # 必要に応じてFalseに変更
        )
        print(f"Successfully uploaded results to {OUTPUT_DATASET_ID}")
    except Exception as e:
        print(f"Failed to upload to Hugging Face Hub: {e}")

if __name__ == "__main__":
    main()
