import subprocess
import sys

# Force install OpenEnv
subprocess.check_call([sys.executable, "-m", "pip", "install",
    "gymnasium==0.29.1", "yfinance", "numpy", "pandas",
    "git+https://github.com/meta-pytorch/OpenEnv.git"])

import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openenv_wrapper import FinanceOpenEnv, FinanceAction, FinanceObservation
from openenv_core import create_fastapi_app
from fastapi.responses import JSONResponse
import uvicorn

# Entry point for OpenEnv validation
app = create_fastapi_app(FinanceOpenEnv, FinanceAction, FinanceObservation)

# Add root route
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
            "health": "GET /health"
        },
        "status": "running"
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)