import numpy as np

class RewardEngine:
    def __init__(self, initial_value=10000, min_return=0.001, max_drawdown=0.05):
        self.initial_value = initial_value
        self.min_return = min_return        # 0.1% minimum daily return
        self.max_drawdown = max_drawdown    # 5% max drawdown allowed
        self.peak_value = initial_value
        self.daily_returns = []

    def reset(self):
        self.peak_value = self.initial_value
        self.daily_returns = []

    def compute(self, prev_value, curr_value, num_trades):
        # Update peak
        self.peak_value = max(self.peak_value, curr_value)

        # 1. Daily return
        daily_return = (curr_value - prev_value) / (prev_value + 1e-9)
        self.daily_returns.append(daily_return)

        # 2. Sharpe component
        if len(self.daily_returns) > 1:
            sharpe = np.mean(self.daily_returns) / (np.std(self.daily_returns) + 1e-9)
        else:
            sharpe = 0.0

        # 3. Drawdown penalty (Capital Protection)
        drawdown = (self.peak_value - curr_value) / (self.peak_value + 1e-9)
        drawdown_penalty = -10.0 if drawdown > self.max_drawdown else 0.0

        # 4. Minimum return gate
        return_bonus = daily_return * 100 if daily_return > self.min_return else -1.0

        # 5. Transaction cost
        transaction_cost = -0.001 * num_trades

        # Final reward
        reward = return_bonus + 0.1 * sharpe + drawdown_penalty + transaction_cost

        return round(reward, 4), {
            "daily_return": round(daily_return * 100, 3),
            "sharpe": round(sharpe, 3),
            "drawdown": round(drawdown * 100, 3),
            "reward": round(reward, 4)
        }

    def is_stopped(self, curr_value):
        # Stop-loss: kill episode if down 10%
        return curr_value < self.initial_value * 0.90