import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State

from pydantic import Field
import numpy as np
from typing import Any, Optional
from uuid import uuid4
from env.finance_env import FinanceEnv


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
    step_num: int = Field(default=0, description="Current step number")


class FinanceOpenEnv(Environment):
    """
    Multi-Asset Portfolio Trading RL Environment using real NSE data.

    Stocks: RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, WIPRO.NS
    Starting Capital: ₹10,000
    Episode Length: 200 trading days
    """

    def __init__(self):
        super().__init__()
        self._env = FinanceEnv(live=False)
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Reset the environment for a new episode."""
        self._env = FinanceEnv(live=False)
        obs, _ = self._env.reset()
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        return FinanceObservation(
            done=False,
            reward=0.0,
            portfolio_value=float(self._env.portfolio_value),
            daily_return=0.0,
            sharpe=0.0,
            drawdown=0.0,
            step_num=0,
            metadata={
                "status": "ready",
                "message": "Finance environment ready!",
                "stocks": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.NS"],
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Execute a step in the environment.

        Accepts a FinanceAction with portfolio weights, or a base Action
        (in which case default equal weights are used).
        """
        self._state.step_count += 1

        # Extract weights from action
        if isinstance(action, FinanceAction):
            weights = np.array(action.weights[:5], dtype=np.float32)
        else:
            # Default equal weights for base Action (e.g., from validator)
            weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)

        obs, reward, terminated, truncated, info = self._env.step(weights)
        done = terminated or truncated
        env_state = self._env.state()

        return FinanceObservation(
            done=done,
            reward=float(reward),
            portfolio_value=float(env_state["portfolio_value"]),
            daily_return=float(info.get("daily_return", 0.0)),
            sharpe=float(info.get("sharpe", 0.0)),
            drawdown=float(info.get("drawdown", 0.0)),
            step_num=int(env_state["step"]),
            metadata={
                "mode": env_state["mode"],
                "peak_value": float(env_state["peak_value"]),
                "drawdown_pct": float(env_state["drawdown_pct"]),
            },
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state