## 事前準備

1.  **必要なPythonライブラリのインストール**
    ```bash
    pip install datasets pandas requests tqdm huggingface_hub pyarrow
    ```

2.  **OpenRouter APIキーの設定**
    OpenRouterで取得したAPIキーを、環境変数として設定してください。
    * **Windowsの場合:**
        ```bash
        set OPENROUTER_API_KEY="ここにあなたのAPIキーを貼り付け"
        ```
    * **macOS / Linuxの場合:**
        ```bash
        export OPENROUTER_API_KEY="ここにあなたのAPIキーを貼り付け"
        ```

3.  **Hugging Faceへログイン**
    ```bash
    huggingface-cli login
    ```
    トークンを入力してください。(**注**: Hugging Faceへのアップロードには`write`権限が必要です)

4.  **Pythonファイル内の下記を変更**
    * `OUTPUT_DATASET_ID`: アップロード先のHugging FaceのデータセットID (例: `"TaroYamada/MyEvolvedMath"`)
    * `NUM_PROBLEMS`: 読み込む問題数

> **※注意**
> このスクリプトは `SynthLabsAI/Big-Math-RL-Verified` を読み込む前提のコードになっています。違うデータセットを読み込むときはコードも合わせて変更する必要があります。
