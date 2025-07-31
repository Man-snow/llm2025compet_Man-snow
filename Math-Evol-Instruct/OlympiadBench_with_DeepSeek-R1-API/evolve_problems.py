import os
import pandas as pd
import requests
from datasets import load_dataset
from tqdm import tqdm
import time
import json
import re # ★★★ 正規表現ライブラリをインポート ★★★

# --- 定数の設定 ---
DATASET_NAME = "SynthLabsAI/Big-Math-RL-Verified"
DATASET_SPLIT = "train"
NUM_PROBLEMS = 5

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "deepseek/deepseek-r1-0528:free"
YOUR_SITE_URL = "http://localhost"
APP_NAME = "BigMath Evolver"

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
    """Hugging Faceからデータセットをロードし、指定条件でソートして返す"""
    print("🔄 データセットをHugging Faceから読み込んでいます...")
    try:
        dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
        df = dataset.to_pandas()
        print(f"✅ データセットの読み込み完了。合計 {len(df)} 問。")
        
        print("🔃 問題をソートしています...")
        df['llama8b_solve_rate'] = pd.to_numeric(df['llama8b_solve_rate'], errors='coerce')
        df.dropna(subset=['llama8b_solve_rate'], inplace=True)
        
        sorted_df = df.sort_values(by=['llama8b_solve_rate', 'problem'], ascending=[True, True])
        
        print(f"✅ ソート完了。上位 {NUM_PROBLEMS} 問を取得します。")
        return sorted_df.head(NUM_PROBLEMS)
        
    except Exception as e:
        print(f"❌ データセットの取得またはソート中にエラーが発生しました: {e}")
        return None

def evolve_problem_with_openrouter(problem_text: str) -> tuple[str, str]:
    """OpenRouter APIを呼び出し、問題文を上方修正する。"""
    if not OPENROUTER_API_KEY:
        return "failure", "❌ 環境変数 'OPENROUTER_API_KEY' が設定されていません。"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": YOUR_SITE_URL, 
        "X-Title": APP_NAME,
        "Content-Type": "application/json"
    }
    prompt = UPWARD_EVOLUTION_PROMPT_TEMPLATE.format(problem=problem_text)
    data = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}]}

    for attempt in range(3):
        try:
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            json_response = response.json()
            
            if 'choices' in json_response and len(json_response['choices']) > 0:
                content = json_response['choices'][0]['message']['content']
                return "success", content.strip()
            else:
                last_error = f"❌ APIからのレスポンスに有効なコンテンツがありませんでした。 Response: {json_response}"

        except json.JSONDecodeError as e:
            last_error = f"❌ APIからの応答が不正な形式でした (JSONDecodeError): {e}"
        except requests.exceptions.RequestException as e:
            last_error = f"❌ APIリクエスト中にエラーが発生しました: {e}"
        except Exception as e:
            last_error = f"❌ 不明なエラーが発生しました: {e}"
        
        print(f"  (試行 {attempt + 1}/3) APIリクエストに失敗。1秒後に再試行します...")
        time.sleep(1)

    return "failure", last_error

# ★★★ 最終的な問題文を抽出する関数を新設 ★★★
def parse_final_instruction(response_text: str) -> str:
    """APIの完全な応答テキストから、最終的な問題文だけを抽出する。"""
    # 正規表現を使って、柔軟に最終的な問題文を抽出
    # re.IGNORECASE: 大文字・小文字を区別しない
    # re.DOTALL: 改行文字も「.」に含める
    match = re.search(r'#Finally Rewritten Instruction#\s*:\s*(.*)', response_text, re.IGNORECASE | re.DOTALL)
    if match:
        # マッチした部分の最初のキャプチャグループ（.*）を取得
        return match.group(1).strip()

    # コロンなしのフォールバックパターン
    match = re.search(r'#Finally Rewritten Instruction#\s*(.*)', response_text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # どちらのパターンにも一致しない場合
    return "抽出失敗"


def main():
    """メイン処理"""
    problems_df = get_sorted_problems()
    
    if problems_df is None:
        return

    results = []
    print(f"\n🚀 {len(problems_df)}問の問題の上方修正を開始します...")

    for index, row in tqdm(problems_df.iterrows(), total=len(problems_df), desc="問題を処理中"):
        original_problem = row['problem']
        
        start_time = time.time()
        status, evolved_response = evolve_problem_with_openrouter(original_problem)
        end_time = time.time()
        processing_time = end_time - start_time
        
        # ★★★ 抽出処理を追加 ★★★
        evolved_problem = ""
        if status == 'success':
            # 成功した場合のみ、最終的な問題文の抽出を試みる
            evolved_problem = parse_final_instruction(evolved_response)
        
        results.append({
            "original_problem": original_problem,
            "evolved_problem": evolved_problem, # 新しい列
            "evolved_response": evolved_response,
            "status": status,
            "processing_time_seconds": round(processing_time, 2),
            "llama8b_solve_rate": row['llama8b_solve_rate'],
            "original_solution": row['predicted_solution']
        })
        
        time.sleep(1)

    results_df = pd.DataFrame(results)
    
    # ★★★ 列の順序を指定 ★★★
    column_order = [
        "original_problem",
        "evolved_problem",
        "evolved_response",
        "status",
        "processing_time_seconds",
        "llama8b_solve_rate",
        "original_solution"
    ]
    # 存在する列のみで順序を再設定
    final_columns = [col for col in column_order if col in results_df.columns]
    results_df = results_df[final_columns]
    
    output_filename = "evolved_math_problems_v3.csv"
    results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 処理が完了しました！結果は '{output_filename}' に保存されました。")

if __name__ == "__main__":
    main()