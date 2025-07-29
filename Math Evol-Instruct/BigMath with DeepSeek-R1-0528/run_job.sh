#!/bin/bash

#SBATCH --job-name=evolve-math-problems
#SBATCH --partition=P02                 # !!ご自身のチームのパーティション名に書き換えてください!!
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8               # vLLMで利用するGPU数
#SBATCH --time=02:00:00                 # 最大実行時間 (HH:MM:SS)
#SBATCH --output=slurm_logs/%x-%j.out   # 標準出力ログの保存場所
#SBATCH --error=slurm_logs/%x-%j.err    # エラーログの保存場所

# --- 環境設定 ---
echo "Job started on $(hostname) at $(date)"

# ログ保存用ディレクトリの作成
mkdir -p slurm_logs

# uvで作成した仮想環境を有効化
# !!パスはご自身の環境に合わせて修正してください!!
source ~/llm2025compet/.venv/bin/activate
echo "Virtual environment activated."

# Hugging Faceのトークンを環境変数として設定
# このスクリプトを実行する前に、`export HF_TOKEN="hf_..."` のようにトークンを設定してください
if [ -z "${HUGGING_FACE_TOKEN}" ]; then
  echo "Error: HUGGING_FACE_TOKEN is not set."
  exit 1
fi
export HF_TOKEN=${HUGGING_FACE_TOKEN}
echo "Hugging Face token is set."


# --- Pythonスクリプトの実行 ---
echo "Running the Python script..."
python generate_problems.py

echo "Job finished at $(date)"
