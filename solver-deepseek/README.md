# Math Problem Solver Agent (DeepSeek/OpenRouter版)

このプロジェクトは、論文「Gemini 2.5 Pro Capable of Winning Gold at IMO 2025」で提案された「自己検証パイプライン」を実装し、複雑な数学の問題を解くものです。

このバージョンは、DeepSeekモデルをOpenRouter API経由で利用し、特定のHugging Faceデータセットから問題を自動で取得するように改造されています。並列実行、堅牢なエラーハンドリング、そしてHugging Face Hubを介した共同作業のための結果マージ機能をサポートしています。

## 主な機能

- **問題の自動取得**: `Man-snow/evolved-math-problems-OlympiadBench-from-deepseek-r1-0528-free` データセットから、指定されたID範囲の問題を取得します。
- **自己検証パイプライン**: 論文で提案された「生成→自己改善→検証→修正」のループを実行し、解の精度を高めます。
- **並列処理**: 各問題に対し、複数のエージェントを同時に実行し、解を発見する確率を高めます。
- **堅牢なリトライ処理**: 初期解の生成が「不完全」と判断された場合に、自動でリトライします。
- **JSONL形式での出力**: 各問題の結果を、構造化されたJSON Lines (`.jsonl`) 形式で保存します。
- **共同作業のためのアップロード機能**: 共有されたHugging Faceデータセットに対し、新しいIDの結果を追加し、既存のIDの結果を上書きする、安全なアップロードを行います。

---

## セットアップ

### 1. 前提条件

- Python 3.7+
- [OpenRouter APIキー](https://openrouter.ai/)
- [Hugging Faceアカウント](https://huggingface.co/) と、書き込み(`write`)権限を持つAPIトークン

### 2. インストール

このリポジトリをクローンし、必要なPythonライブラリをインストールします。

```bash
git clone <your_repo_url>
cd <your_repo_name>
pip install -r requirements.txt
```

### 3. 認証設定

OpenRouterとHugging Faceの両方の認証情報を設定する必要があります。

#### OpenRouter APIキー:

プロジェクトのルートディレクトリに `.env` ファイルを作成し、キーを追加します。

```env
OPENROUTER_API_KEY="sk-or-..."
```

#### Hugging Face ログイン:

結果をアップロードするには、コマンドライン経由でHugging Faceアカウントにログインします。

```bash
huggingface-cli login
```

プロンプトに従って、ご自身のAPIトークンを入力してください。

---

## 実行方法

メインスクリプトは `run_solver.py` です。コマンドライン引数を使って動作を制御できます。

### 基本コマンド

```bash
python run_solver.py [OPTIONS]
```

### コマンドライン引数

- `--start_problem <ID>`: 取得を開始する問題の `new_id`。（デフォルト: 1）
- `--num_problems <N>`: 試行する問題の数。（デフォルト: 3）
- `--num_agents <N>`: 各問題に対して並列実行するエージェントの数。（デフォルト: 3）  
  ※ APIのレート制限を避けるため、1に設定することを推奨します。
- `--output_file <PATH>`: 結果を出力するファイルパス。（デフォルト: `results.jsonl`）
- `--upload_to_hf`: このフラグを付けると、Hugging Face Hubに結果をアップロードします。（デフォルト: 無効）
- `--hf_repo <REPO_ID>`: アップロード先のHugging FaceデータセットリポジトリID（例: `YourUsername/YourRepoName`）  
  ※ `--upload_to_hf` を使用する場合は必須です。

---

## 実行例

### 例1: 10問目から5問を解く

```bash
python run_solver.py --start_problem 10 --num_problems 5 --num_agents 1
```

### 例2: ID 50から2問を解き、結果をアップロードする

```bash
python run_solver.py --start_problem 50 --num_problems 2 --num_agents 1 --upload_to_hf --hf_repo "MyUsername/math-olympiad-solutions"
```

---

## 出力について

### ログファイル

`logs/` ディレクトリには、各エージェントの試行ごとの詳細なリアルタイムログが保存されます。

ファイル名は `problem_{ID}_agent_{AGENT_NUM}.log` の形式です。  
特定のエージェントの思考プロセスをデバッグするのに役立ちます。

### 結果ファイル (`results.jsonl`)

`results.jsonl` は、試行された各問題の最終結果を格納する主要な出力ファイルです。  
これは JSON Lines 形式のファイルで、各行が1つの完全なJSONオブジェクトになっています。

#### 各行の構造:

```json
{
    "id": 123, 
    "question": "問題の全文...",
    "output": "<think>成功した最初のエージェントの思考プロセスを含む、生の解答全文...</think>最終的な答え",
    "answer": "抽出された最終的な答え（例: '0'や'-2'、'解は存在しない'など）"
}
```

※ 問題が解けなかった場合、`output` の値は `"NO_SOLUTION_FOUND"` になります。

---

## 引用元 (Citation)

このプロジェクトの根幹となる手法は、以下の論文に基づいています。

```bibtex
@article{huang2025gemini,
  title={Gemini 2.5 Pro Capable of Winning Gold at IMO 2025},
  author={Huang, Yichen and Yang, Lin F},
  journal={arXiv preprint arXiv:2507.15855},
  year={2025}
}
```
