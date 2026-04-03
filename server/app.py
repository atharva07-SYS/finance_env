import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv_wrapper import FinanceOpenEnv, FinanceAction, FinanceObservation
from openenv_core import create_fastapi_app
from fastapi.responses import JSONResponse
import uvicorn

app = create_fastapi_app(FinanceOpenEnv, FinanceAction, FinanceObservation)

@app.get("/")
async def root():
    return JSONResponse({
        "name": "Finance-Env",
        "description": "Multi-Asset Portfolio Trading RL Environment",
        "author": "Atharva - Mumbai, India",
        "stocks": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.NS"],
        "status": "running"
    })

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()