import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openenv_core import create_fastapi_app
from openenv_wrapper import FinanceOpenEnv, FinanceAction, FinanceObservation
import uvicorn

# Pass CLASS not instance
app = create_fastapi_app(FinanceOpenEnv, FinanceAction, FinanceObservation)

if __name__ == "__main__":
    print("Finance OpenEnv Server running on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)