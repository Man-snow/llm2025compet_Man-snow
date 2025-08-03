from datasets import load_dataset

def check_dataset_overlap():
    """
    Hugging Face上の2つのデータセットをロードし、
    問題文の重複をチェックして結果を表示する。
    """
    try:
        # データセット① (OlympiadBench) をロード
        print("データセット① (Hothan/OlympiadBench) をロード中...")
        dataset1 = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train")
        print("ロード完了。")

        # データセット② (Big-Math-RL-Verified) をロード
        print("\nデータセット② (SynthLabsAI/Big-Math-RL-Verified) をロード中...")
        dataset2 = load_dataset("SynthLabsAI/Big-Math-RL-Verified", split="train")
        print("ロード完了。")

        # データセットから問題文のリストを取得
        questions1 = dataset1['question']
        
        # 高速な検索のために、データセット②の問題文をセット(set)に変換
        # セットは重複する要素を持たず、要素の存在チェックが非常に速い
        print("\nデータセット②を検索用に準備中...")
        problems2_set = set(dataset2['problem'])
        print("準備完了。")

        print("\n重複チェックを開始します...")
        
        # 重複している問題の数をカウントする変数
        overlap_count = 0
        
        # データセット①の各問題がデータセット②に存在するかチェック
        for i, question in enumerate(questions1):
            if question in problems2_set:
                overlap_count += 1
            # 進捗表示
            if (i + 1) % 100 == 0:
                print(f"{i + 1} / {len(questions1)} 件チェック完了...")

        # 最終結果の表示
        print("\n--- チェック結果 ---")
        print(f"データセット① (OlympiadBench) の問題数: {len(questions1)}問")
        print(f"データセット② (Big-Math-RL-Verified) の問題数: {len(problems2_set)}問")
        print(f"重複していた問題の数: {overlap_count}問")

    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    check_dataset_overlap()