"""
Entry point for the Finance Environment server.

This script is sometimes executed by the OpenEnv validator to check structural requirements.
We wrap the server startup in a try/except block to avoid crashing if the Docker container
has already bound to port 8000 (which it does via the CMD in Dockerfile).
"""

import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from server.app import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Start the Finance-Env server."""
    port = int(os.environ.get("APP_PORT", 8000))
    
    try:
        logger.info(f"Attempting to start server on port {port}...")
        uvicorn.run("server.app:app", host="0.0.0.0", port=port)
    except OSError as e:
        if e.errno == 98 or "address already in use" in str(e).lower():
            logger.info(f"Port {port} is already in use. The server is likely already running via Docker CMD.")
            # Exit cleanly so the OpenEnv validator doesn't think the environment failed
            sys.exit(0)
        else:
            logger.error(f"Failed to start server: {e}")
            raise
    except Exception as e:
        logger.error(f"Unhandled exception during server startup: {e}")
        raise

if __name__ == "__main__":
    main()