import subprocess
import sys
import os

subprocess.check_call([sys.executable, "-m", "pip", "install",
    "gymnasium==0.29.1", "yfinance==1.2.0", "numpy", "pandas", 
    "matplotlib", "openenv-core"])

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openenv_wrapper import FinanceOpenEnv, FinanceAction, FinanceObservation
from openenv_core import create_fastapi_app
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from env.finance_env import FinanceEnv
from env.data_loader import get_latest_prices

# OpenEnv FastAPI app
app = create_fastapi_app(FinanceOpenEnv, FinanceAction, FinanceObservation)

@app.get("/")
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
            "health": "GET /health",
            "demo": "GET /demo"
        },
        "status": "running"
    })

@app.get("/demo", response_class=HTMLResponse)
async def demo():
    # Run episode
    env = FinanceEnv(live=False)
    obs, _ = env.reset()
    done = False
    steps = 0
    portfolio_values = [10000]

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        done = terminated or truncated
        state = env.state()
        portfolio_values.append(state['portfolio_value'])

    state = env.state()
    total_return = (state['portfolio_value'] - 10000) / 10000 * 100
    prices = get_latest_prices()

    # Generate chart
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(portfolio_values, color='cyan', linewidth=2)
    ax.axhline(y=10000, color='white', linestyle='--', alpha=0.5)
    ax.fill_between(range(len(portfolio_values)),
                    portfolio_values, 10000,
                    where=[v >= 10000 for v in portfolio_values],
                    color='green', alpha=0.3)
    ax.fill_between(range(len(portfolio_values)),
                    portfolio_values, 10000,
                    where=[v < 10000 for v in portfolio_values],
                    color='red', alpha=0.3)
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    chart = base64.b64encode(buf.read()).decode()
    plt.close()

    price_rows = "".join([f"<tr><td>{t}</td><td>₹{p}</td></tr>" for t, p in prices.items()])

    html = f"""
    <html>
    <head><title>Finance-Env Demo</title>
    <style>
        body {{ background: #1a1a2e; color: white; font-family: Arial; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        td, th {{ border: 1px solid #444; padding: 8px; }}
        th {{ background: #333; }}
    </style>
    </head>
    <body>
        <h1>🏦 Finance-Env: Multi-Asset Portfolio Trading</h1>
        <h2>Episode Results</h2>
        <p>Final Portfolio: ₹{state['portfolio_value']} | Return: {round(total_return, 2)}% | Steps: {steps}</p>
        <img src="data:image/png;base64,{chart}" width="100%"/>
        <h2>Live NSE Prices</h2>
        <table><tr><th>Stock</th><th>Price</th></tr>{price_rows}</table>
        <p><a href="/docs" style="color:cyan">📖 API Docs</a> | 
        <a href="/health" style="color:cyan">❤️ Health</a></p>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

def main():
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Finance-Env Server starting on port {port}")
    print(f"📊 Demo: http://0.0.0.0:{port}/demo")
    print(f"📖 API Docs: http://0.0.0.0:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()