from datasets import load_dataset
from huggingface_hub import HfApi

# --- 設定項目 ---
# 1. 上書きしたいHugging Face上のデータセットのリポジトリ名
repo_id = "Man-snow/evolved-math-problems-OlympiadBench-from-deepseek-r1-0528-free"

# 2. アップロードしたいローカルのCSVファイルのパス
local_csv_path = "../OlympiadBench_with_DeepSeek-R1-0528-free/evolved_math_problems_OE_TO_maths_en_COMP1～40.csv" # 👈 ご自身のファイルパスに書き換えてください

# --- 実行コード ---

# 1. ローカルのCSVファイルを読み込む
print(f"ローカルファイル '{local_csv_path}' を読み込んでいます...")
local_dataset = load_dataset("csv", data_files=local_csv_path)

# 2. Hugging Face Hubにプッシュして、既存のデータセットを上書きする
print(f"データセットをリポジトリ '{repo_id}' にプッシュしています。既存のデータは上書きされます。")
local_dataset.push_to_hub(repo_id)

print("\n✅ データセットの上書きが完了しました。")
print(f"Hugging Face Hubでデータセットを確認してください: https://huggingface.co/datasets/{repo_id}")