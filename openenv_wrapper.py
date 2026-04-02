import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openenv_core import Environment, Action, Observation, State
import numpy as np
from env.finance_env import FinanceEnv

class FinanceAction(Action):
    weights: list = [0.2, 0.2, 0.2, 0.2, 0.2]

class FinanceObservation(Observation):
    portfolio_value: float = 10000.0
    daily_return: float = 0.0
    sharpe: float = 0.0
    drawdown: float = 0.0
    reward: float = 0.0
    step: int = 0
    done: bool = False

class FinanceState(State):
    step: int = 0
    portfolio_value: float = 10000.0
    peak_value: float = 10000.0
    drawdown_pct: float = 0.0
    mode: str = "HISTORICAL"

class FinanceOpenEnv(Environment):

    def __init__(self):
        self._env = FinanceEnv(live=False)

    def reset(self) -> FinanceObservation:
        obs, _ = self._env.reset()
        return FinanceObservation(
            portfolio_value=float(self._env.portfolio_value),
            daily_return=0.0,
            sharpe=0.0,
            drawdown=0.0,
            reward=0.0,
            step=int(self._env.current_step),
            done=False
        )

    def step(self, action: FinanceAction):
        weights = np.array(action.weights, dtype=np.float32)
        obs, reward, terminated, truncated, info = self._env.step(weights)
        done = terminated or truncated

        observation = FinanceObservation(
            portfolio_value=float(self._env.portfolio_value),
            daily_return=float(info.get('daily_return', 0.0)),
            sharpe=float(info.get('sharpe', 0.0)),
            drawdown=float(info.get('drawdown', 0.0)),
            reward=float(reward),
            step=int(self._env.current_step),
            done=done
        )
        return observation, float(reward), done

    def state(self) -> FinanceState:
        s = self._env.state()
        return FinanceState(
            step=int(s['step']),
            portfolio_value=float(s['portfolio_value']),
            peak_value=float(s['peak_value']),
            drawdown_pct=float(s['drawdown_pct']),
            mode=str(s['mode'])
        )