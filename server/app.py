"""
FastAPI application for the Finance Environment.

This module creates the OpenEnv HTTP server, adds a rich '/' metadata
endpoint, and mounts the Gradio trading dashboard at '/demo'.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from fastapi.responses import JSONResponse

from openenv.core.env_server.http_server import create_app
from openenv_wrapper import FinanceOpenEnv, FinanceAction, FinanceObservation

# ─── 1. Create the core OpenEnv app (provides /health, /reset, /step, etc.) ──
app = create_app(
    FinanceOpenEnv, FinanceAction, FinanceObservation, env_name="finance_env"
)


# ─── 2. Rich root metadata endpoint ──────────────────────────────────────────
@app.get("/")
async def root():
    """Return environment metadata for the top-level endpoint."""
    return JSONResponse({
        "name": "Finance-Env",
        "version": "1.0.0",
        "description": (
            "Multi-Asset Portfolio Trading RL Environment using real NSE data. "
            "Manage a portfolio of RELIANCE, TCS, INFY, HDFCBANK, and WIPRO "
            "with professional reward signals including Sharpe ratio, drawdown "
            "penalties, and transaction costs."
        ),
        "openenv_spec": "0.2.3",
        "action_space": {
            "type": "FinanceAction",
            "fields": {
                "weights": {
                    "type": "list[float]",
                    "length": 5,
                    "default": [0.2, 0.2, 0.2, 0.2, 0.2],
                    "description": "Portfolio allocation weights for each stock (must sum to 1).",
                }
            },
        },
        "observation_space": {
            "type": "FinanceObservation",
            "fields": {
                "portfolio_value": "float — current portfolio value in INR",
                "daily_return": "float — daily return percentage",
                "sharpe": "float — rolling Sharpe ratio",
                "drawdown": "float — current drawdown percentage",
                "step_num": "int — current step in the episode",
                "done": "bool — whether the episode has ended",
                "reward": "float — reward for the last step",
            },
        },
        "stocks": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.NS"],
        "episode_length": 200,
        "starting_capital": 10000,
        "endpoints": {
            "health": "/health",
            "reset": "/reset",
            "step": "/step",
            "state": "/state",
            "schema": "/schema",
            "demo_ui": "/demo",
        },
    })


# ─── 3. Mount Gradio dashboard at /demo ──────────────────────────────────────
try:
    from app import gradio_app
    app = gr.mount_gradio_app(app, gradio_app, path="/demo")
except Exception:
    # If Gradio import fails (e.g. during validation), the core server still works
    pass


# ─── 4. Entry point ──────────────────────────────────────────────────────────
def main():
    """Entry point for direct execution."""
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()