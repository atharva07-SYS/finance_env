import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openenv_core import Environment, Action, Observation, State
from pydantic import BaseModel
from typing import Optional
import numpy as np
from env.finance_env import FinanceEnv

# --- Action Model ---
class FinanceAction(Action):
    weights: list = [0.2, 0.2, 0.2, 0.2, 0.2]

# --- Observation Model ---
class FinanceObservation(Observation):
    portfolio_value: float = 10000.0
    daily_return: float = 0.0
    sharpe: float = 0.0
    drawdown: float = 0.0
    reward: float = 0.0
    step: int = 0
    done: bool = False

# --- State Model ---
class FinanceState(State):
    step: int = 0
    portfolio_value: float = 10000.0
    peak_value: float = 10000.0
    drawdown_pct: float = 0.0
    mode: str = "HISTORICAL"

# --- OpenEnv Compliant Environment ---
class FinanceOpenEnv(Environment):

    def __init__(self):
        self._env = None

    def _get_env(self):
        if self._env is None:
            self._env = FinanceEnv(live=False)
        return self._env

    async def reset(self) -> FinanceObservation:
        env = self._get_env()
        obs, _ = env.reset()
        return FinanceObservation(
            portfolio_value=float(env.portfolio_value),
            daily_return=0.0,
            sharpe=0.0,
            drawdown=0.0,
            reward=0.0,
            step=int(env.current_step),
            done=False
        )

    async def step(self, action: FinanceAction) -> tuple:
        env = self._get_env()
        weights = np.array(action.weights, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(weights)
        done = terminated or truncated

        return FinanceObservation(
            portfolio_value=float(env.portfolio_value),
            daily_return=float(info.get('daily_return', 0.0)),
            sharpe=float(info.get('sharpe', 0.0)),
            drawdown=float(info.get('drawdown', 0.0)),
            reward=float(reward),
            step=int(env.current_step),
            done=done
        ), float(reward), done

    async def state(self) -> FinanceState:
        env = self._get_env()
        s = env.state()
        return FinanceState(
            step=int(s['step']),
            portfolio_value=float(s['portfolio_value']),
            peak_value=float(s['peak_value']),
            drawdown_pct=float(s['drawdown_pct']),
            mode=str(s['mode'])
        )