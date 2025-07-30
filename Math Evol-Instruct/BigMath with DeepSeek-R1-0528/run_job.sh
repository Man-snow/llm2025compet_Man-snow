#!/bin/bash

#SBATCH --job-name=evolve-math-deepseek-r1
#SBATCH --partition=P02                 # !!ご自身のチームのパーティション名に書き換えてください!!
#SBATCH --nodes=3
#SBATCH --gpus-per-node=8               # vLLMで利用するGPU数
#SBATCH --time=24:00:00                 # モデルのロード時間を考慮し、最大実行時間を24時間に延長
#SBATCH --mem=1200G                     # 巨大モデルのロードに必要なシステムメモリを1TBに増量
#SBATCH --output=slurm_logs/%x-%j.out   # 標準出力ログの保存場所
#SBATCH --error=slurm_logs/%x-%j.err    # エラーログの保存場所

# --- リアルタイムでの進捗確認方法 ---
# このジョブを sbatch で投入した後、ログインノードで以下のコマンドを実行すると、
# 処理の進捗をリアルタイムで確認できます。
# ※ JOBID の部分は、sbatch実行時に表示されるジョブIDに置き換えてください。
#
# tail -f slurm_logs/evolve-math-deepseek-r1-JOBID.out
# -----------------------------------------

# --- 環境設定 ---
echo "ジョブ開始: $(date)"
echo "実行ノード: $(hostname)"

# GPU間のP2P通信を無効にし、初期化の安定性を向上させる
export NCCL_P2P_DISABLE=1
echo "NCCL_P2P_DISABLE=1 に設定しました。"

# ログ保存用ディレクトリの作成
mkdir -p slurm_logs

# 【重要】事前準備
# このスクリプトを実行する前に、ログインノードで一度 `huggingface-cli login` を実行し、
# 「Write」権限を持つトークンで認証を済ませておく必要があります。

# 共有リポジトリにある仮想環境を有効化する
VENV_PATH="$HOME/llm2025compet/.venv/bin/activate"
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
    echo "仮想環境 ($VENV_PATH) を有効化しました。"
else
    echo "エラー: 仮想環境が見つかりません。パスを確認してください: $VENV_PATH"
    exit 1
fi

# --- Pythonスクリプトの実行 ---
# 実行前にvLLMのコンパイルキャッシュを削除し、破損したキャッシュによるエラーを防ぐ
echo "古いvLLMコンパイルキャッシュを削除します..."
rm -rf ~/.cache/vllm/torch_compile_cache
echo "キャッシュを削除しました。"

echo "Pythonスクリプト (generate_problems.py) を実行します..."
# !!修正!!: srun と python -u を使い、出力をリアルタイムでログに送るようにする
srun python -u generate_problems.py

echo "ジョブ終了: $(date)"
