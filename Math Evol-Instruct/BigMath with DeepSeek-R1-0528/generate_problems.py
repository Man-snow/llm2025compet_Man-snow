import os
import time
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
import re
from transformers import AutoTokenizer

# --- 定数設定 ---

# 計算ノードで実行する、軽量な量子化モデル
# Qwen/Qwen2.5-1.5B-Instruct-AWQ は存在しないため、同等の性能を持つ TheBloke/Qwen2-1.5B-Instruct-AWQ を利用します
MODEL_ID = "TheBloke/Qwen2-1.5B-Instruct-AWQ"
# 入力データセット
SOURCE_DATASET_ID = "SynthLabsAI/Big-Math-RL-Verified"
# 出力ファイル名
OUTPUT_CSV_FILENAME = "evolved_problems_output.csv"

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
    print("--- ステップ1: データセットの準備 ---")
    # Gated Datasetにアクセスするため、事前に `huggingface-cli login` が必要
    try:
        # !!修正!!: 不要になった trust_remote_code=True を削除
        dataset = load_dataset(SOURCE_DATASET_ID, split="train")
        df = dataset.to_pandas()
    except Exception as e:
        print(f"データセットの読み込みに失敗しました: {e}")
        return

    # llama8b_solve_rateの昇順、problemのアルファベット昇順でソート
    sorted_df = df.sort_values(by=["llama8b_solve_rate", "problem"], ascending=[True, True])
    
    # 上位100問を取得
    problems_to_process = sorted_df.head(100)
    print(f"データセットの準備が完了しました。処理対象: {len(problems_to_process)}問")

    # --- 2. vLLMモデルの初期化 ---
    print("--- ステップ2: vLLMモデルの初期化 ---")
    try:
        # 計算ノードのGPU数(8基)に合わせて並列化
        llm = LLM(
            model=MODEL_ID,
            quantization="awq",
            tensor_parallel_size=8, 
            trust_remote_code=True
        )
        sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"モデルの初期化に失敗しました: {e}")
        return
    print("モデルの初期化が完了しました。")

    # --- 3. プロンプトの生成とモデルによる処理 ---
    print("--- ステップ3: 問題生成と結果の集計 ---")
    
    # Qwen2 Instructモデルのチャットテンプレートに合わせてプロンプトを整形
    messages_list = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": UPWARD_EVOLUTION_PROMPT_TEMPLATE.format(problem=row["problem"])}
        ]
        for _, row in problems_to_process.iterrows()
    ]
    prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_list]

    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    total_elapsed_time = end_time - start_time
    print(f"問題生成が完了しました。処理時間: {total_elapsed_time:.2f}秒")

    # --- 4. 結果の集計 ---
    print("--- ステップ4: 結果の集計 ---")
    results = []
    for i, output in enumerate(outputs):
        original_problem_text = problems_to_process.iloc[i]['problem']
        generated_text = output.outputs[0].text
        evolved_problem = parse_final_instruction(generated_text)
        avg_time_per_problem = total_elapsed_time / len(outputs) if len(outputs) > 0 else 0
        
        results.append({
            "original_problem": original_problem_text,
            "evolved_problem": evolved_problem,
            "total_tokens": len(output.outputs[0].token_ids),
            "elapsed_time_avg": avg_time_per_problem,
            "success": evolved_problem is not None,
            "full_model_output": generated_text
        })
    print("結果の集計が完了しました。")

    # --- 5. 結果をCSVファイルとして保存 ---
    print(f"--- ステップ5: 結果を'{OUTPUT_CSV_FILENAME}'に保存 ---")
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV_FILENAME, index=False, encoding='utf-8-sig')
    print("正常に保存されました。")
    print("\n--- 生成データ プレビュー ---")
    print(results_df.head())

if __name__ == "__main__":
    main()