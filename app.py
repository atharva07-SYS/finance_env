import subprocess
import sys

# Force install required packages
subprocess.check_call([sys.executable, "-m", "pip", "install", 
    "gymnasium==0.29.1", "yfinance", "numpy", "pandas"])

import gradio as gr
import numpy as np
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

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

        if steps % 20 == 0:
            state = env.state()
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

    return "\n".join(log) + "\n" + summary

demo = gr.Interface(
    fn=run_episode,
    inputs=[],
    outputs=gr.Textbox(label="Episode Results", lines=20),
    title="🏦 Finance-Env: Multi-Asset Portfolio Trading",
    description="Click Run to simulate one episode of AI portfolio trading on real NSE data!",
)

demo.launch()