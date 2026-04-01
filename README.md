---
---
title: Finance Env
emoji: 🏦
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.23.0"
app_file: inference.py
python_version: "3.11"
pinned: false
startup_duration_timeout: 1h
---

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
pip install -r requirements.txt
```

### Test Environment
```bash
python test_env.py
```

### Run Grader
```bash
python grader/grader.py
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
| transaction_cost | -0.001 per trade |

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
│   ├── finance_env.py    # Core environment
│   ├── reward.py         # Reward logic
│   └── data_loader.py    # NSE data pipeline
├── grader/
│   └── grader.py         # Auto scoring
├── app.py                # HuggingFace Spaces demo
├── test_env.py           # Quick test
└── requirements.txt
```

---

## 🏆 Sample Grader Results
```
========= GRADER RESULTS =========
Episode 1: ₹8995 | Return: -10.04% | Reward: -516
Episode 2: ₹8888 | Return: -11.12% | Reward: -1184
Avg Return : -10.37%
==================================
```

> Note: Negative returns are expected with a random agent.
> A trained RL agent will learn to maximize returns over time!

---

## 👨‍💻 Author
Built by Atharva — Mumbai, India 🇮🇳