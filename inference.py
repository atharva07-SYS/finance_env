import logging
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openenv_wrapper import FinanceOpenEnv, FinanceAction

def main():
    """
    Baseline inference script for OpenEnv evaluation.
    This script runs the environment for a single episode and emits
    the required [START], [STEP], and [END] structured logs to stdout.
    """
    env = FinanceOpenEnv()
    
    # 1. Reset Environment
    obs = env.reset()
    task_name = "finance_env"
    
    # Print START block
    print(f"[START] task={task_name}", flush=True)

    # 2. Run Episode
    total_reward = 0.0
    steps = 0
    done = False
    
    # Using a dummy action for baseline inference
    action = FinanceAction()

    # We will just run for a few steps for validation if not fully evaluating,
    # but let's run until done (or a max of 10 steps to ensure it finishes quickly for validation)
    max_steps = 10
    
    while not done and steps < max_steps:
        steps += 1
        obs = env.step(action)
        reward = obs.reward
        total_reward += reward
        done = obs.done
        
        # Print STEP block
        print(f"[STEP] step={steps} reward={reward:.4f}", flush=True)
        
    score = total_reward  # Modify this if you have a specific score calculation
    
    # 3. Print END block
    print(f"[END] task={task_name} score={score:.4f} steps={steps}", flush=True)
    
    env.close()

if __name__ == "__main__":
    main()