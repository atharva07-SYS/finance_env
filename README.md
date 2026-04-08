---
title: Finance Env
emoji: 🏦
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
base_path: /web
pinned: false
tags:
  - openenv-0.2.3
  - openenv
---

This Space is built from OpenEnv environment `finance_env`.

- Space URL: `https://huggingface.co/spaces/20atharva26/finance-env`
- OpenEnv pinned ref: `0.2.3`
- Hub tag: `openenv`

```python
from finance_env import FinanceOpenEnv

env = FinanceOpenEnv()
```

# 🏦 Finance-Env: Multi-Asset Portfolio Trading Environment

A professional-grade Reinforcement Learning environment for algorithmic trading using real Indian stock market (NSE) data.

Built for the **Meta x Scaler OpenEnv Hackathon 2025**.

---

## 🎯 Problem Statement

Train an AI agent to manage a portfolio of 5 NSE stocks intelligently — maximizing returns while protecting capital through professional risk management.

---

## 🧠 What Makes This Special

- 📈 **Real NSE Data** — Reliance, TCS, Infosys, HDFC Bank, Wipro
- 🛡️ **Capital Protection** — Stop-loss at 10% drawdown
- 📊 **Professional Rewards** — Sharpe Ratio + Drawdown penalty
- 💸 **Transaction Costs** — Realistic trading simulation
- 🎯 **Multi-Asset** — Agent manages 5 stocks simultaneously

---

## ⚙️ Environment Details

| Property | Value |
|----------|-------|
| Stocks | RELIANCE, TCS, INFY, HDFCBANK, WIPRO |
| Actions | Portfolio weights per stock (0-1) |
| Observation | Price, Volume, MA20, RSI per stock |
| Episode Length | 200 trading days |
| Starting Capital | ₹10,000 |
| Stop-Loss | -10% from peak |

---

## 🚀 Quick Start

### Install
```bash
pip install -e .
```

### Test Environment
```bash
python test_env.py
```

### Run Server
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

---

## 📊 Reward Function
```
reward = return_bonus + 0.1 × sharpe - drawdown_penalty - transaction_cost
```

| Component | Description |
|-----------|-------------|
| return_bonus | Daily portfolio return × 100 |
| sharpe | Risk-adjusted return metric |
| drawdown_penalty | -10 if drawdown > 5% |
| transaction_cost | Cost per trade to discourage overtrading |

---

## 🛡️ Capital Protection Features

| Feature | Description |
|---------|-------------|
| Stop-Loss | Episode ends if portfolio drops 10% |
| Drawdown Shield | -10 penalty if drawdown exceeds 5% |
| Min Return Gate | Penalizes lazy/no-action behavior |
| Transaction Cost | -0.001 per trade to discourage overtrading |

---

## 📁 Project Structure
```
finance-env/
├── env/
│   ├── finance_env.py    # Core Gymnasium environment
│   ├── reward.py         # Reward logic
│   └── data_loader.py    # NSE data pipeline
├── server/
│   └── app.py            # FastAPI application
├── openenv_wrapper.py    # OpenEnv MCPEnvironment wrapper
├── inference.py          # Server entry point
├── openenv.yaml          # Environment manifest
├── pyproject.toml        # Dependencies
└── Dockerfile            # Container image
```

---

## 👨‍💻 Author
Built by Atharva — Mumbai, India 🇮🇳