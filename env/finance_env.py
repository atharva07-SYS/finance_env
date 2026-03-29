import gymnasium as gym
import numpy as np
from env.data_loader import load_data, get_observation, get_daily_returns
from env.reward import RewardEngine

class FinanceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.data = load_data()
        self.reward_engine = RewardEngine(initial_value=10000)
        self.current_step = 0
        self.portfolio_value = 10000
        self.prev_value = 10000
        self.max_steps = 200

        # 5 stocks x 4 features
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(5, 4), dtype=np.float32
        )
        # Portfolio weights per stock
        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(5,), dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 20  # Start after MA warmup
        self.portfolio_value = 10000
        self.prev_value = 10000
        self.reward_engine.reset()
        obs = get_observation(self.data, self.current_step)
        return obs, {}

    def step(self, action):
        # Normalize weights
        weights = action / (action.sum() + 1e-9)

        # Get returns for each stock today
        returns = get_daily_returns(self.data, self.current_step)

        # Portfolio return = weighted sum of stock returns
        portfolio_return = float(np.dot(weights, returns))
        self.portfolio_value = self.prev_value * (1 + portfolio_return)

        # Count trades (non-zero allocations)
        num_trades = int(np.sum(weights > 0.05))

        # Compute reward
        reward, info = self.reward_engine.compute(
            self.prev_value, self.portfolio_value, num_trades
        )

        # Stop-loss check
        stop_loss_hit = self.reward_engine.is_stopped(self.portfolio_value)
        if stop_loss_hit:
            reward -= 50

        self.prev_value = self.portfolio_value
        self.current_step += 1

        terminated = (
            self.current_step >= self.max_steps or stop_loss_hit
        )

        obs = get_observation(self.data, self.current_step)
        return obs, reward, terminated, False, info

    def state(self):
        return {
            "step": self.current_step,
            "portfolio_value": round(self.portfolio_value, 2),
            "peak_value": round(self.reward_engine.peak_value, 2),
            "drawdown_pct": round(
                (self.reward_engine.peak_value - self.portfolio_value)
                / (self.reward_engine.peak_value + 1e-9) * 100, 2
            )
        }