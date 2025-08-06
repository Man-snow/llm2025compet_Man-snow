import subprocess
import os
import sys
import time
import argparse # argparseをインポート
from concurrent.futures import ProcessPoolExecutor
from datasets import load_dataset
import pandas

# ★ 変更点 1: 関数に引数を追加し、取得範囲を指定できるようにする
def fetch_problems(start_index=1, num_to_fetch=3):
    """
    Hugging Face Hubから指定された範囲の数学問題を取得する
    """
    print(f"📚 Fetching {num_to_fetch} problems from Hugging Face Hub, starting from problem #{start_index}...")
    try:
        dataset = load_dataset("Man-snow/evolved-math-problems-OlympiadBench-from-deepseek-r1-0528-free", split='train', streaming=True)
        
        problems = []
        
        # isliceを使ってデータセットの指定された範囲を効率的に処理
        from itertools import islice
        start_position = start_index - 1 # 0-based index
        dataset_slice = islice(dataset, start_position, start_position + num_to_fetch)

        for i, example in enumerate(dataset_slice):
            problems.append({
                "id": start_index + i, # 問題番号を正しく設定
                "problem": example['evolved_problem']
            })
        
        print(f"✅ Successfully fetched {len(problems)} problems.")
        if not problems:
            print("Warning: No problems were fetched. Check start_index and dataset size.")
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
        "agent_modified.py", # ここはご自身のファイル名に合わせてください
        problem_file,
        "--log",
        log_file
    ]
    
    print(f"🚀 Starting agent {agent_id} for problem {problem_id}...")
    
    try:
        subprocess.run(
            cmd,
            timeout=1800,
            check=False
        )
        if os.path.exists("SUCCESS_SIGNAL.txt"):
            os.remove("SUCCESS_SIGNAL.txt")
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
                    executor.shutdown(wait=False, cancel_futures=True)
                    break 
            except Exception as e:
                 print(f"A worker process failed: {e}")

    os.remove(problem_file)
    
    if not solution_found:
        print(f"\n❌ No solution found for Problem {problem_id} after {num_agents} attempts.")
        
    return solution_found

def main():
    parser = argparse.ArgumentParser(description='Solve math problems from Hugging Face Hub.')
    parser.add_argument('--num_agents', '-n', type=int, default=3, 
                        help='Number of parallel agents to run per problem (default: 1)')
    parser.add_argument('--start_problem', type=int, default=4, 
                        help='The starting problem number to fetch (default: 4)')
    parser.add_argument('--num_problems', type=int, default=7, 
                        help='The number of problems to fetch (default: 7, for problems 4 to 10)')
    args = parser.parse_args()
    
    # ★ 変更点 2: コマンドライン引数をfetch_problemsに渡す
    problems = fetch_problems(start_index=args.start_problem, num_to_fetch=args.num_problems)
    
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    start_time = time.time()
    
    solved_count = 0
    for problem in problems:
        if solve_problem_in_parallel(problem['id'], problem['problem'], num_agents=args.num_agents, log_dir=log_dir):
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