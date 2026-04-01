import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openenv_core import Environment, Action, Observation, State
from pydantic import BaseModel
from typing import Any, Dict, Optional
import numpy as np
from env.finance_env import FinanceEnv

# --- Action Model ---
class FinanceAction(Action):
    weights: list[float] = [0.2, 0.2, 0.2, 0.2, 0.2]

# --- Observation Model ---
class FinanceObservation(Observation):
    portfolio_value: float
    daily_return: float
    sharpe: float
    drawdown: float
    reward: float
    step: int

# --- State Model ---
class FinanceState(State):
    step: int
    portfolio_value: float
    peak_value: float
    drawdown_pct: float
    mode: str

# --- OpenEnv Compliant Environment ---
class FinanceOpenEnv(Environment):

    def __init__(self):
        self.env = FinanceEnv(live=False)
        self._last_info = {}

    async def reset(self) -> FinanceObservation:
        obs, _ = self.env.reset()
        self._last_info = {}
        return FinanceObservation(
            portfolio_value=float(self.env.portfolio_value),
            daily_return=0.0,
            sharpe=0.0,
            drawdown=0.0,
            reward=0.0,
            step=self.env.current_step
        )

    async def step(self, action: FinanceAction) -> tuple[FinanceObservation, float, bool]:
        weights = np.array(action.weights, dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(weights)
        self._last_info = info
        done = terminated or truncated

        return FinanceObservation(
            portfolio_value=float(self.env.portfolio_value),
            daily_return=float(info.get('daily_return', 0.0)),
            sharpe=float(info.get('sharpe', 0.0)),
            drawdown=float(info.get('drawdown', 0.0)),
            reward=float(reward),
            step=self.env.current_step
        ), float(reward), done

    async def state(self) -> FinanceState:
        s = self.env.state()
        return FinanceState(
            step=s['step'],
            portfolio_value=s['portfolio_value'],
            peak_value=s['peak_value'],
            drawdown_pct=s['drawdown_pct'],
            mode=s['mode']
        )