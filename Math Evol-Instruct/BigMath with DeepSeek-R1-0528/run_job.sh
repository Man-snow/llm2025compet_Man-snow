#!/bin/bash

#SBATCH --job-name=evolve-math-server
#SBATCH --partition=P02                 # !!ご自身のチームのパーティション名に書き換えてください!!
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8               # vLLMで利用するGPU数
#SBATCH --time=02:00:00                 # 最大実行時間 (HH:MM:SS)
#SBATCH --mem=64G                       # !!修正!!: ジョブが使用するメインメモリを64GBに指定
#SBATCH --output=slurm_logs/%x-%j.out   # 標準出力ログの保存場所
#SBATCH --error=slurm_logs/%x-%j.err    # エラーログの保存場所

# --- 環境設定 ---
echo "ジョブ開始: $(date)"
echo "実行ノード: $(hostname)"

# スクリプト自身の場所に移動して、パスの問題を解決する
cd "$(dirname "$0")"
echo "作業ディレクトリをスクリプトの場所に変更しました: $(pwd)"


# ログ保存用ディレクトリの作成
mkdir -p slurm_logs

# 【重要】事前準備
# このスクリプトを実行する前に、ログインノードで一度 `huggingface-cli login` を実行し、
# Hugging Faceアカウントの認証を済ませておく必要があります。

# 共有リポジトリにある仮想環境を有効化する
# パスを `Damin3927` のリポジトリ名に修正
VENV_PATH="$HOME/llm2025compet/.venv/bin/activate"
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
    echo "仮想環境 ($VENV_PATH) を有効化しました。"
else
    echo "エラー: 仮想環境が見つかりません。パスを確認してください: $VENV_PATH"
    exit 1
fi

# --- Pythonスクリプトの実行 ---
echo "Pythonスクリプト (generate_server.py) を実行します..."
python generate_server.py

echo "ジョブ終了: $(date)"
