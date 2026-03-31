import subprocess
import sys

# Force install required packages
subprocess.check_call([sys.executable, "-m", "pip", "install",
    "gymnasium==0.29.1", "yfinance", "numpy", "pandas", "matplotlib"])

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.finance_env import FinanceEnv

def run_episode():
    env = FinanceEnv()
    obs, _ = env.reset()
    total_reward = 0
    done = False
    steps = 0
    log = []
    portfolio_values = [10000]

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

        state = env.state()
        portfolio_values.append(state['portfolio_value'])

        if steps % 20 == 0:
            log.append(
                f"Step {steps} | "
                f"Portfolio: ₹{state['portfolio_value']} | "
                f"Drawdown: {state['drawdown_pct']}% | "
                f"Reward: {round(reward, 3)}"
            )

    state = env.state()
    total_return = (state['portfolio_value'] - 10000) / 10000 * 100

    summary = f"""
===== EPISODE SUMMARY =====
Final Portfolio Value : ₹{state['portfolio_value']}
Total Return          : {round(total_return, 2)}%
Peak Value            : ₹{state['peak_value']}
Final Drawdown        : {state['drawdown_pct']}%
Total Steps           : {steps}
Total Reward          : {round(total_reward, 2)}
===========================
    """

    # Generate chart
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(portfolio_values, color='cyan', linewidth=2, label='Portfolio Value')
    ax.axhline(y=10000, color='white', linestyle='--', alpha=0.5, label='Starting Capital ₹10,000')
    ax.fill_between(range(len(portfolio_values)), 
                    portfolio_values, 10000,
                    where=[v >= 10000 for v in portfolio_values],
                    color='green', alpha=0.3, label='Profit Zone')
    ax.fill_between(range(len(portfolio_values)),
                    portfolio_values, 10000,
                    where=[v < 10000 for v in portfolio_values],
                    color='red', alpha=0.3, label='Loss Zone')
    ax.set_title('Portfolio Value Over Time', color='white', fontsize=14)
    ax.set_xlabel('Trading Steps', color='white')
    ax.set_ylabel('Portfolio Value (₹)', color='white')
    ax.tick_params(colors='white')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    plt.tight_layout()

    return "\n".join(log) + "\n" + summary, fig

demo = gr.Interface(
    fn=run_episode,
    inputs=[],
    outputs=[
        gr.Textbox(label="Episode Results", lines=20),
        gr.Plot(label="Portfolio Performance Chart")
    ],
    title="🏦 Finance-Env: Multi-Asset Portfolio Trading",
    description="Click Run to simulate one episode of AI portfolio trading on real NSE data!",
)

demo.launch()