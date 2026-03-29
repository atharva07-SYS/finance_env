import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.finance_env import FinanceEnv

env = FinanceEnv()

print("⏳ Loading real NSE data... please wait")

obs, info = env.reset()
print("✅ reset() works — obs shape:", obs.shape)

action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print("✅ step() works — reward:", reward, "| info:", info)

state = env.state()
print("✅ state() works —", state)