import os
import time
import csv # --- 変更点: csvライブラリをインポート ---
from openai import OpenAI

# --- 1. APIクライアントの設定 ---
try:
    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )
except Exception as e:
    print(f"エラー: APIキーが設定されていません。環境変数 'DEEPSEEK_API_KEY' を設定してください。")
    print(f"詳細: {e}")
    exit()

# --- 2. 上方進化プロンプトの定義 ---
UPWARD_EVOLUTION_PROMPT = """
Step 1: Understand the core concept and structure of the "#Instruction#". Identify the key elements such as variables, conditions, participants, actions, or processes that can be manipulated to increase complexity. Also, recognize the theme of the instruction and ensure it remains consistent throughout the evolution.
Step 2: Formulate a comprehensive plan to increment the complexity of the "#Instruction#" based on the identified elements in Step 1. The plan should involve modifying or expanding at least three components from the list. It is crucial to ensure that all components in the instruction are logically interconnected and that the complexity increase is coherent and justified. The plan should avoid introducing variables or conditions without clear criteria for determining their values or without contributing to the overall complexity. In this step, consider adding more real-world constraints and dependencies between variables to make the problem more challenging. And you can also add more constraints, concretizing, increasing reasoning.
Step 3: Implement the plan step by step to create the "#Rewritten Instruction#". Ensure the rewritten instruction maintains a logical sequence and avoids ambiguity or confusion. If additional variables or conditions are introduced, provide clear and unambiguous methods or criteria for determining their values. The "#Rewritten Instruction#" should not exceed the original "#Instruction#" by more than 30 words to ensure readability and comprehension.
Step 4: Review the "#Rewritten Instruction#" thoroughly to identify any unreasonable elements or inconsistencies. Make sure the "#Rewritten Instruction#" is a more complex version of the "#Instruction#". and that it accurately reflects the intended increase in complexity. Adjust any part of the instruction that may lead to misunderstanding or ambiguity, and provide the "#Finally Rewritten Instruction#" without any supplementary explanation.
Please reply strictly in the following format:
Step 1
#Elements Identified#:
...
Step 2
#Plan#:
...
Step 3
#Rewritten Instruction#:
...
Step 4
#Finally Rewritten Instruction#:
...
"""

# --- 3. 進化させる問題のリスト ---
problems_to_evolve = [
    "Let $T_{1}, T_{2}, T_{3}, T_{4}$ be pairwise distinct collinear points such that $T_{2}$ lies between $T_{1}$ and $T_{3}$, and $T_{3}$ lies between $T_{2}$ and $T_{4}$. Let $\\omega_{1}$ be a circle through $T_{1}$ and $T_{4}$; let $\\omega_{2}$ be the circle through $T_{2}$ and internally tangent to $\\omega_{1}$ at $T_{1}$; let $\\omega_{3}$ be the circle through $T_{3}$ and externally tangent to $\\omega_{2}$ at $T_{2}$; and let $\\omega_{4}$ be the circle through $T_{4}$ and externally tangent to $\\omega_{3}$ at $T_{3}$. A line crosses $\\omega_{1}$ at $P$ and $W, \\omega_{2}$ at $Q$ and $R, \\omega_{3}$ at $S$ and $T$, and $\\omega_{4}$ at $U$ and $V$, the order of these points along the line being $P, Q, R, S, T, U, V, W$. Prove that $P Q+T U=R S+V W$.",
    "A number of 17 workers stand in a row. Every contiguous group of at least 2 workers is a brigade. The chief wants to assign each brigade a leader (which is a member of the brigade) so that each worker's number of assignments is divisible by 4 . Prove that the number of such ways to assign the leaders is divisible by 17 .",
    "Let $A B C$ be a triangle, let $D$ be the touchpoint of the side $B C$ and the incircle of the triangle $A B C$, and let $J_{b}$ and $J_{c}$ be the incentres of the triangles $A B D$ and $A C D$, respectively. Prove that the circumcentre of the triangle $A J_{b} J_{c}$ lies on the bisectrix of the angle $B A C$.",
    "\nProve that every positive integer $n$ can be written uniquely in the form\n\n\n\n$$\n\nn=\\sum_{j=1}^{2 k+1}(-1)^{j-1} 2^{m_{j}}\n\n$$\n\n\n\nwhere $k \\geq 0$ and $0 \\leq m_{1}<m_{2}<\\cdots<m_{2 k+1}$ are integers.\n\n\n\nThis number $k$ is called the weight of $n$.",
    "Given a triangle $A B C$, let $H$ and $O$ be its orthocentre and circumcentre, respectively. Let $K$ be the midpoint of the line segment $A H$. Let further $\\ell$ be a line through $O$, and let $P$ and $Q$ be the orthogonal projections of $B$ and $C$ onto $\\ell$, respectively. Prove that $K P+K Q \\geq B C$.",
]

# --- 変更点: CSVファイルの設定 ---
csv_filename = 'evolved_problems_log.csv'
# UTF-8-sigにすることでExcelで開いた際の文字化けを防ぎます
with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    # ヘッダー（列名）を書き込む
    writer.writerow(['original instruction', 'updated instruction', 'total_tokens', 'process time'])

    print("🚀 問題の上方進化を開始し、結果をCSVに記録します...\n")

    for i, problem in enumerate(problems_to_evolve):
        problem_start_time = time.time() 

        print(f"--- 問題 {i+1}/{len(problems_to_evolve)} ---")
        print(f"元の問題:\n{problem}\n")

        try:
            chat_completion = client.chat.completions.create(
                model="deepseek-reasoner", 
                messages=[
                    {"role": "system", "content": UPWARD_EVOLUTION_PROMPT},
                    {"role": "user", "content": f"#Instruction#:\n{problem}"}
                ],
                max_tokens=10000,
                temperature=0.7,
                timeout=700.0,
            )
            
            # --- 変更点: API応答からデータを抽出 ---
            response_text = chat_completion.choices[0].message.content
            print("--- 進化した問題（APIの全応答） ---\n", chat_completion, "\n------------------------------------")
            total_tokens = chat_completion.usage.total_tokens
            
            # --- デバッグ用コードを追加 ---
            print("--- APIからの応答全文 ---\n", response_text, "\n------------------------") 
            
            # 応答テキストから最終的な問題文だけを抽出する
            final_instruction = ""
            if "#Finally Rewritten Instruction#:" in response_text:
                final_instruction = response_text.split("#Finally Rewritten Instruction#:")[1].strip()
            
            print("進化した問題:\n", final_instruction)
            
            # 処理時間を計算
            duration = time.time() - problem_start_time
            
            # --- 変更点: データをCSVファイルに書き込む ---
            writer.writerow([problem, final_instruction, total_tokens, f"{duration:.2f}"])

        except APITimeoutError as e:
            print(f"エラー: API呼び出しがタイムアウトしました。")
            print(f"詳細: {e}")
        except APIStatusError as e:
            print(f"エラー: APIサーバーからエラーステータスが返されました。")
            print(f"ステータスコード: {e.status_code}")
            print(f"応答内容: {e.response}")
        except APIConnectionError as e:
            print(f"エラー: APIサーバーへの接続に失敗しました。")
            print(f"詳細: {e.__cause__}")
        except Exception as e:
            print(f"エラー: API呼び出し中に問題が発生しました。")
            print(f"詳細: {e}")
            duration = time.time() - problem_start_time
            # エラーが発生した場合も記録を残す
            writer.writerow([problem, 'ERROR', 'N/A', f"{duration:.2f}"])
        
        finally:
            print(f"⏱️ この問題の処理時間: {duration:.2f} 秒")
            print("-" * 25 + "\n")

print(f"✅ すべての処理が完了し、'{csv_filename}' に結果を保存しました。")