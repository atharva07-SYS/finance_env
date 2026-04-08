"""
FastAPI application for the Finance Environment.

This module creates an HTTP server that exposes the FinanceOpenEnv
over HTTP and WebSocket endpoints.

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.http_server import create_app
from openenv_wrapper import FinanceOpenEnv, FinanceAction, FinanceObservation

# Create the app with our custom Action/Observation types
app = create_app(
    FinanceOpenEnv, FinanceAction, FinanceObservation, env_name="finance_env"
)


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()