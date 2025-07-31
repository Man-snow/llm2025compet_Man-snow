#!/bin/bash

#SBATCH --job-name=evolve-math-deepseek-3node
#SBATCH --partition=P02                 # !!ご自身のチームのパーティション名に書き換えてください!!
#SBATCH --nodes=3                       # 3ノードを要求
#SBATCH --ntasks-per-node=1             # 各ノードで1つのタスクを実行
#SBATCH --gpus-per-node=8               # 1ノードあたりのGPU数
#SBATCH --time=24:00:00
#SBATCH --mem=1024G                     # 要求メモリをサーバーの物理限界に近い1TBに増量
#SBATCH --cpus-per-task=32              # !!追加!!: 各タスクに割り当てるCPUコア数を32に指定
# !!修正!!: ログの出力先を、ジョブ投入ディレクトリからの絶対パスに修正
#SBATCH --output="$SLURM_SUBMIT_DIR/slurm_logs/%x-%j.out"
#SBATCH --error="$SLURM_SUBMIT_DIR/slurm_logs/%x-%j.err"

# --- チームメンバーの成功実績がある環境設定を全面的に導入 ---
echo "Loading modules..."
module purge
module load cuda/12.4
module load cudnn/9.6.0
module load nccl/2.24.3

echo "Setting environment variables for distributed run..."
# !!修正!!: 共有キャッシュの指定を削除。各ライブラリがデフォルトの安全な場所(~/.cache)を使うようにする
# export HF_HUB_CACHE="/home/Competition2025/P02/shareP02/.cache/huggingface/hub"
# export VLLM_CACHE_ROOT="/home/Competition2025/P02/shareP02/.cache/vllm"

# NCCL (GPU間通信ライブラリ) の詳細設定
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=^lo,docker,virbr
export NCCL_PROTO=Simple

# RayとTokenizerの並列化設定
export RAY_DISABLE_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false

# --------------------------------------------------------------------

echo "ジョブ開始: $(date)"
echo "実行ノード: $(hostname)"

# Slurmのジョブ投入ディレクトリに移動し、権限の問題を解決する
cd "$SLURM_SUBMIT_DIR"
echo "作業ディレクトリをジョブ投入時の場所に変更しました: $(pwd)"


# ログ保存用ディレクトリの作成
mkdir -p slurm_logs

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
# 実行前にvLLMのコンパイルキャッシュを削除
echo "古いvLLMコンパイルキャッシュを削除します..."
rm -rf ~/.cache/vllm/torch_compile_cache
echo "キャッシュを削除しました。"

echo "Pythonスクリプト (generate_problems.py) を実行します..."

# !!修正!!: srunで3つのタスクを起動し、親タスク(PROCID=0)からのみPythonを実行する
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}
srun --cpu-bind=v --accel-bind=gn bash -c '
if [[ $SLURM_PROCID -eq 0 ]]; then
    echo "--- Head Node (PROCID=$SLURM_PROCID) starting Python script ---"
    python -u generate_problems.py
else
    echo "--- Worker Node (PROCID=$SLURM_PROCID) is standing by ---"
    # ワーカーノードはvLLMによって自動的に利用されるため、ここでは待機するだけ
    sleep infinity
fi
'

echo "ジョブ終了: $(date)"