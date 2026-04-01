import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openenv_wrapper import FinanceOpenEnv, FinanceAction, FinanceObservation
from openenv_core import create_fastapi_app
import uvicorn

# Entry point for OpenEnv validation
app = create_fastapi_app(FinanceOpenEnv, FinanceAction, FinanceObservation)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)