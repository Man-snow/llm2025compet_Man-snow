from datasets import load_dataset
import pandas as pd

# ダウンロードしたいデータセットの情報を指定
DATASET_ID = "Hothan/OlympiadBench"
CONFIG_NAME = "OE_TO_maths_en_COMP" #TP_TO_maths_en_COMP
SPLIT_NAME = "train"  # 通常は'train'スプリットにデータがあります

# 出力するCSVファイル名
OUTPUT_FILENAME = f"{CONFIG_NAME}.csv"

def download_as_csv():
    """Hugging Face Hubからデータセットを読み込み、CSVとして保存する"""
    try:
        # 1. Hugging Face Hubからデータセットを読み込む
        print(f"🔄 Downloading dataset '{DATASET_ID}' ({CONFIG_NAME})...")
        dataset = load_dataset(DATASET_ID, CONFIG_NAME, split=SPLIT_NAME)
        print("✅ Download complete.")

        # 2. pandasのDataFrameに変換する
        print("🔄 Converting to DataFrame...")
        df = dataset.to_pandas()
        print("✅ Conversion complete.")

        # 3. CSVファイルとして保存する
        print(f"💾 Saving to '{OUTPUT_FILENAME}'...")
        df.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8-sig')
        print(f"🎉 Successfully saved dataset to '{OUTPUT_FILENAME}'")

    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    download_as_csv()