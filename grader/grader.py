import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.finance_env import FinanceEnv

def run_grader(episodes=5):
    env = FinanceEnv()
    results = []

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        state = env.state()
        total_return = (state['portfolio_value'] - 10000) / 10000 * 100
        results.append({
            "episode": ep + 1,
            "final_value": state['portfolio_value'],
            "total_return_pct": round(total_return, 2),
            "total_reward": round(total_reward, 2)
        })

    # Summary
    returns = [r['total_return_pct'] for r in results]
    print("\n========= GRADER RESULTS =========")
    for r in results:
        print(f"Episode {r['episode']}: ₹{r['final_value']} | Return: {r['total_return_pct']}% | Reward: {r['total_reward']}")
    print(f"\nAvg Return : {round(np.mean(returns), 2)}%")
    print(f"Best Return: {round(max(returns), 2)}%")
    print(f"Worst Return: {round(min(returns), 2)}%")
    print("==================================\n")

if __name__ == "__main__":
    run_grader()