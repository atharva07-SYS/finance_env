import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install",
    "gymnasium==0.29.1", "yfinance", "numpy", "pandas", "matplotlib",
    "git+https://github.com/meta-pytorch/OpenEnv.git"])

import os
import threading
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openenv_wrapper import FinanceOpenEnv, FinanceAction, FinanceObservation
from openenv_core import create_fastapi_app
from fastapi.responses import JSONResponse
import uvicorn
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from env.finance_env import FinanceEnv
from env.data_loader import get_latest_prices

# ── OpenEnv FastAPI app ──────────────────────────────────────
openenv_app = create_fastapi_app(FinanceOpenEnv, FinanceAction, FinanceObservation)

@openenv_app.get("/")
async def root():
    return JSONResponse({
        "name": "Finance-Env",
        "description": "Multi-Asset Portfolio Trading RL Environment",
        "author": "Atharva - Mumbai, India",
        "stocks": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.NS"],
        "endpoints": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "docs": "GET /docs",
            "health": "GET /health"
        },
        "status": "running"
    })

# ── Gradio demo ──────────────────────────────────────────────
def run_episode(mode):
    live = (mode == "🔴 Live NSE Data")
    env = FinanceEnv(live=live)
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
    prices = get_latest_prices()
    price_str = "\n".join([f"  {t}: ₹{p}" for t, p in prices.items()])

    summary = f"""
📡 Mode: {state['mode']}

📈 LIVE NSE PRICES (Today):
{price_str}

===== EPISODE SUMMARY =====
Final Portfolio Value : ₹{state['portfolio_value']}
Total Return          : {round(total_return, 2)}%
Peak Value            : ₹{state['peak_value']}
Final Drawdown        : {state['drawdown_pct']}%
Total Steps           : {steps}
Total Reward          : {round(total_reward, 2)}
===========================
    """

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
    ax.set_title(f'Portfolio Value Over Time ({state["mode"]})', color='white', fontsize=14)
    ax.set_xlabel('Trading Steps', color='white')
    ax.set_ylabel('Portfolio Value (₹)', color='white')
    ax.tick_params(colors='white')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    plt.tight_layout()

    return "\n".join(log) + "\n" + summary, fig

gradio_app = gr.Interface(
    fn=run_episode,
    inputs=[
        gr.Radio(
            choices=["🔴 Live NSE Data", "📊 Historical Data (2022-2024)"],
            value="📊 Historical Data (2022-2024)",
            label="Select Data Mode"
        )
    ],
    outputs=[
        gr.Textbox(label="Episode Results", lines=20),
        gr.Plot(label="Portfolio Performance Chart")
    ],
    title="🏦 Finance-Env: Multi-Asset Portfolio Trading",
    description="Simulate AI portfolio trading on real NSE data!",
)

# ── Mount Gradio inside FastAPI ──────────────────────────────
gradio_app = gr.mount_gradio_app(openenv_app, gradio_app, path="/demo")

if __name__ == "__main__":
    print("Server running!")
    print("Demo: http://0.0.0.0:7860/demo")
    print("API:  http://0.0.0.0:7860/docs")
    uvicorn.run(openenv_app, host="0.0.0.0", port=7860)