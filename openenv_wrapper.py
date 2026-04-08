import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State

from pydantic import Field
import numpy as np
from typing import Any, Optional
from uuid import uuid4
from env.finance_env import FinanceEnv
from fastmcp import FastMCP


class FinanceAction(Action):
    """Action for Finance environment - portfolio weights for 5 stocks."""
    weights: list = Field(
        default=[0.2, 0.2, 0.2, 0.2, 0.2],
        description="Portfolio weights for each stock (RELIANCE, TCS, INFY, HDFCBANK, WIPRO)"
    )


class FinanceObservation(Observation):
    """Observation from the Finance environment."""
    portfolio_value: float = Field(default=10000.0, description="Current portfolio value")
    daily_return: float = Field(default=0.0, description="Daily return percentage")
    sharpe: float = Field(default=0.0, description="Sharpe ratio")
    drawdown: float = Field(default=0.0, description="Current drawdown percentage")
    reward: float = Field(default=0.0, description="Step reward")
    step: int = Field(default=0, description="Current step number")


class FinanceOpenEnv(MCPEnvironment):
    """
    Multi-Asset Portfolio Trading RL Environment using real NSE data.

    This environment exposes trading functionality through MCP tools:
    - `trade`: Execute a trade with given portfolio weights
    - `get_portfolio_status`: Get current portfolio status

    Stocks: RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, WIPRO.NS
    """

    def __init__(self):
        # Create MCP server and define tools inline
        mcp = FastMCP("finance_env")

        self._env = FinanceEnv(live=False)
        self._episode_state = State(episode_id=str(uuid4()), step_count=0)

        env_ref = self._env

        @mcp.tool
        def trade(weights: list[float] = [0.2, 0.2, 0.2, 0.2, 0.2]) -> dict:
            """
            Execute a trade with given portfolio weights for 5 NSE stocks.
            Weights are normalized to sum to 1.

            Args:
                weights: List of 5 floats representing portfolio allocation
                         [RELIANCE, TCS, INFY, HDFCBANK, WIPRO]

            Returns:
                Dictionary with portfolio value, reward, and trade info
            """
            w = np.array(weights[:5], dtype=np.float32)
            obs, reward, terminated, truncated, info = env_ref.step(w)
            done = terminated or truncated
            state = env_ref.state()
            return {
                "portfolio_value": state["portfolio_value"],
                "reward": round(float(reward), 4),
                "daily_return": info.get("daily_return", 0.0),
                "sharpe": info.get("sharpe", 0.0),
                "drawdown": info.get("drawdown", 0.0),
                "step": state["step"],
                "done": done,
                "mode": state["mode"],
            }

        @mcp.tool
        def get_portfolio_status() -> dict:
            """
            Get the current portfolio status including value, drawdown, and mode.

            Returns:
                Dictionary with portfolio status information
            """
            state = env_ref.state()
            return {
                "portfolio_value": state["portfolio_value"],
                "peak_value": state["peak_value"],
                "drawdown_pct": state["drawdown_pct"],
                "step": state["step"],
                "mode": state["mode"],
            }

        # Pass the MCP server to the base class
        super().__init__(mcp)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Reset the environment for a new episode."""
        self._env = FinanceEnv(live=False)
        obs, _ = self._env.reset()
        self._episode_state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "status": "ready",
                "message": "Finance environment ready!",
                "portfolio_value": float(self._env.portfolio_value),
                "stocks": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.NS"],
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Handle non-MCP actions."""
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use ListToolsAction or CallToolAction for MCP interactions."
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Execute a step in the environment."""
        self._episode_state.step_count += 1
        return super().step(action, timeout_s=timeout_s, **kwargs)

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Async step used by the WebSocket handler."""
        self._episode_state.step_count += 1
        return await super().step_async(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._episode_state