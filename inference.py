"""
Entry point for the Finance Environment server.

This is the main entry point referenced by the HuggingFace Space.
It launches the FastAPI server using uvicorn.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from server.app import app


def main():
    """Start the Finance-Env server."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()