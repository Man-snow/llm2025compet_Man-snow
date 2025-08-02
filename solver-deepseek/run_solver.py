import subprocess
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datasets import load_dataset

def fetch_problems():
    """
    Hugging Face Hubから数学問題を取得する
    """
    print("📚 Fetching problems from Hugging Face Hub...")
    try:
        # データセットをストリーミングモードでロード
        dataset = load_dataset("Man-snow/evolved-math-problems-OlympiadBench-from-deepseek-r1-0528-free", split='train', streaming=True)
        
        problems = []
        # 最初の3問を取得
        for i, example in enumerate(iter(dataset)):
            if i >= 3:
                break
            problems.append({
                "id": i + 1,
                "problem": example['evolved_problem']
            })
        
        print(f"✅ Successfully fetched {len(problems)} problems.")
        return problems
    except Exception as e:
        print(f"❌ Failed to fetch problems: {e}")
        sys.exit(1)

def run_single_agent_instance(agent_id, problem_id, problem_file, log_dir):
    """
    agent_modified.py の単一インスタンスを実行する
    """
    log_file = os.path.join(log_dir, f"problem_{problem_id}_agent_{agent_id:02d}.log")
    cmd = [
        sys.executable,
        "agent_modified.py",
        problem_file,
        "--log",
        log_file
    ]
    
    print(f"🚀 Starting agent {agent_id} for problem {problem_id}...")
    
    try:
        # agent_modified.pyは成功時にSUCCESS_SIGNAL.txtを生成する
        # ここではプロセスの完了を待つだけ
        subprocess.run(
            cmd,
            timeout=1800, # 30分間のタイムアウト
            check=False # エラーで停止しない
        )
        # 成功シグナルファイルがあるか確認
        if os.path.exists("SUCCESS_SIGNAL.txt"):
            os.remove("SUCCESS_SIGNAL.txt") # 他のプロセスに影響しないように削除
            return (problem_id, agent_id, True)
        return (problem_id, agent_id, False)

    except subprocess.TimeoutExpired:
        print(f"⌛ Agent {agent_id} for problem {problem_id} timed out.")
        return (problem_id, agent_id, False)
    except Exception as e:
        print(f"💥 Agent {agent_id} for problem {problem_id} failed with an error: {e}")
        return (problem_id, agent_id, False)

def solve_problem_in_parallel(problem_id, problem_text, num_agents=3, log_dir="logs"):
    """
    1つの問題に対して複数のエージェントを並列実行する
    """
    problem_file = f"problem_{problem_id}.txt"
    with open(problem_file, "w", encoding="utf-8") as f:
        f.write(problem_text)
        
    print("\n" + "="*60)
    print(f"🔬 Solving Problem {problem_id} with {num_agents} parallel agents...")
    print(f"📄 Problem statement saved to {problem_file}")
    print("="*60)
    
    solution_found = False
    
    with ProcessPoolExecutor(max_workers=num_agents) as executor:
        futures = [executor.submit(run_single_agent_instance, i, problem_id, problem_file, log_dir) for i in range(num_agents)]
        
        for future in futures:
            try:
                prob_id, agent_id, success = future.result()
                if success:
                    solution_found = True
                    print(f"\n🎉🎉🎉 Agent {agent_id} FOUND A SOLUTION for Problem {prob_id}! 🎉🎉🎉")
                    print(f"Check logs in '{log_dir}/problem_{prob_id}_agent_{agent_id:02d}.log' for details.")
                    # 1つでも成功したら他のエージェントはキャンセル（実際にはシャットダウンを試みる）
                    executor.shutdown(wait=False, cancel_futures=True)
                    break 
            except Exception as e:
                 print(f"A worker process failed: {e}")


    os.remove(problem_file) # 一時ファイルを削除
    
    if not solution_found:
        print(f"\n❌ No solution found for Problem {problem_id} after {num_agents} attempts.")
        
    return solution_found

def main():
    # 問題を取得
    problems = fetch_problems()
    
    # ログディレクトリを作成
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    start_time = time.time()
    
    solved_count = 0
    for problem in problems:
        if solve_problem_in_parallel(problem['id'], problem['problem'], num_agents=3, log_dir=log_dir):
            solved_count += 1
            
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "#"*60)
    print("### FINAL SUMMARY ###")
    print(f"Total problems attempted: {len(problems)}")
    print(f"Problems solved: {solved_count}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"All logs are available in the '{log_dir}' directory.")
    print("#"*60)

if __name__ == "__main__":
    main()