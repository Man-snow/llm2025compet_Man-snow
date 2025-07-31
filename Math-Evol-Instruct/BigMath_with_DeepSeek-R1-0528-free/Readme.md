## 事前準備
1. 必要なPythonライブラリのインストール
pip install datasets pandas requests tqdm huggingface_hub pyarrow

2. OpenRouter APIキーの設定
OpenRouterで取得したAPIキーを、環境変数として設定

Windowsの場合:
set OPENROUTER_API_KEY="ここにあなたのAPIキーを貼り付け"
macOS / Linuxの場合:
export OPENROUTER_API_KEY="ここにあなたのAPIキーを貼り付け"

3. HuggingFaceへログイン
    huggingface-cli login

トークンを入力（HuggingFaceへのアップロードはWrite権限必要）

4. Pythonファイル内の下記は変更してください。
OUTPUT_DATASET_ID：アップロード先のhuggingfaceのデータセットID
NUM_PROBLEMS：読み込む問題数

※今はSynthLabsAI/Big-Math-RL-Verifiedを読み込む前提のコードになっています。違うデータセットを読み込むときはコードも合わせて変更する必要あり。