"""
Finance-Env Gradio Dashboard — Professional Trading Terminal UI.

This module provides a polished gr.Blocks dashboard that is mounted
onto the OpenEnv FastAPI server at /demo.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from env.finance_env import FinanceEnv
from env.data_loader import get_latest_prices, STOCKS

# ─── Theme ────────────────────────────────────────────────────────────────────
THEME = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="teal",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _classify_action(weights):
    """Classify each stock's weight into Buy / Hold / Sell signals."""
    signals = []
    for i, w in enumerate(weights):
        ticker = STOCKS[i].replace(".NS", "")
        if w > 0.25:
            signals.append(f"🟢 {ticker}: BUY  ({w:.0%})")
        elif w < 0.10:
            signals.append(f"🔴 {ticker}: SELL ({w:.0%})")
        else:
            signals.append(f"⚪ {ticker}: HOLD ({w:.0%})")
    return " │ ".join(signals)


def _fmt_inr(x, _):
    """Format y-axis as ₹ with commas."""
    return f"₹{x:,.0f}"


# ─── Core Episode Runner ─────────────────────────────────────────────────────

def run_episode(mode):
    """Run one full episode and return (log_text, chart_figure, summary_text)."""
    live = mode == "🔴 Live NSE Data"
    env = FinanceEnv(live=live)
    obs, _ = env.reset()

    total_reward = 0.0
    done = False
    steps = 0
    log_lines = []
    portfolio_values = [10000.0]

    # Header
    log_lines.append("━" * 60)
    log_lines.append(f"  {'🔴 LIVE TRADING SESSION' if live else '📊 HISTORICAL BACKTEST'}")
    log_lines.append(f"  Stocks: {', '.join(s.replace('.NS','') for s in STOCKS)}")
    log_lines.append(f"  Starting Capital: ₹10,000")
    log_lines.append("━" * 60)

    while not done:
        action = env.action_space.sample()
        weights = action / (action.sum() + 1e-9)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

        state = env.state()
        portfolio_values.append(state["portfolio_value"])

        # Log every 10 steps + first + last
        if steps == 1 or steps % 10 == 0 or done:
            signals = _classify_action(weights)
            pv = state["portfolio_value"]
            dd = state["drawdown_pct"]
            ret_pct = (pv - 10000) / 100
            status = "🏁 DONE" if done else "⏳"

            log_lines.append(
                f"\n  Step {steps:>3}  {status}\n"
                f"  ├─ Signals: {signals}\n"
                f"  ├─ Portfolio: ₹{pv:,.2f}  ({ret_pct:+.2f}%)\n"
                f"  ├─ Drawdown: {dd:.2f}%\n"
                f"  └─ Reward:   {reward:+.4f}"
            )

    # ── Summary ───────────────────────────────────────────────────────────
    state = env.state()
    total_return = (state["portfolio_value"] - 10000) / 100
    peak = state["peak_value"]

    # Live prices
    try:
        prices = get_latest_prices()
        price_lines = "\n".join(f"    {t.replace('.NS',''):>10}: ₹{p}" for t, p in prices.items())
    except Exception:
        price_lines = "    (Could not fetch live prices)"

    summary = (
        f"{'═' * 50}\n"
        f"  📡 Mode        : {state['mode']}\n"
        f"{'─' * 50}\n"
        f"  💰 Final Value  : ₹{state['portfolio_value']:,.2f}\n"
        f"  📈 Total Return : {total_return:+.2f}%\n"
        f"  🏔️  Peak Value   : ₹{peak:,.2f}\n"
        f"  📉 Drawdown     : {state['drawdown_pct']:.2f}%\n"
        f"  🏃 Steps        : {steps}\n"
        f"  🎯 Total Reward : {total_reward:+.2f}\n"
        f"{'─' * 50}\n"
        f"  📡 Live NSE Prices (Today)\n{price_lines}\n"
        f"{'═' * 50}"
    )

    # ── Chart ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))

    x = range(len(portfolio_values))
    ax.plot(portfolio_values, color="#10b981", linewidth=2.2, label="Portfolio Value", zorder=3)
    ax.axhline(y=10000, color="#94a3b8", linestyle="--", alpha=0.6, label="Starting Capital ₹10,000")

    # Profit / loss zones
    ax.fill_between(
        x, portfolio_values, 10000,
        where=[v >= 10000 for v in portfolio_values],
        color="#10b981", alpha=0.15, label="Profit Zone",
    )
    ax.fill_between(
        x, portfolio_values, 10000,
        where=[v < 10000 for v in portfolio_values],
        color="#ef4444", alpha=0.15, label="Loss Zone",
    )

    # Peak marker
    peak_idx = int(np.argmax(portfolio_values))
    ax.annotate(
        f"Peak ₹{max(portfolio_values):,.0f}",
        xy=(peak_idx, max(portfolio_values)),
        xytext=(peak_idx + 8, max(portfolio_values) + 200),
        fontsize=9, color="#10b981", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#10b981", lw=1.2),
    )

    ax.set_title(
        f"Portfolio Performance — {state['mode']}",
        fontsize=15, fontweight="bold", color="#e2e8f0", pad=14,
    )
    ax.set_xlabel("Trading Steps", fontsize=11, color="#94a3b8")
    ax.set_ylabel("Portfolio Value (₹)", fontsize=11, color="#94a3b8")
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_inr))
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.legend(facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=9, loc="upper left")

    for spine in ax.spines.values():
        spine.set_color("#334155")

    ax.grid(axis="y", color="#1e293b", linewidth=0.5)
    plt.tight_layout()

    return "\n".join(log_lines), fig, summary


# ─── Gradio Blocks Dashboard ─────────────────────────────────────────────────

with gr.Blocks(theme=THEME, title="🏦 Finance-Env Trading Terminal") as gradio_app:

    gr.Markdown(
        """
        # 🏦 Finance-Env: Multi-Asset Portfolio Trading Terminal
        > Simulate AI portfolio trading on **real NSE data** — Reliance, TCS, Infosys, HDFC Bank & Wipro.
        > Built for the **Meta × Scaler OpenEnv Hackathon 2025**.
        """,
    )

    with gr.Row(equal_height=False):

        # ── Left Column: Controls ─────────────────────────────────────────
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### ⚙️ Settings")

            mode_radio = gr.Radio(
                choices=["📊 Historical Data (2022-2024)", "🔴 Live NSE Data"],
                value="📊 Historical Data (2022-2024)",
                label="Data Source",
                info="Historical uses 2 years of cached data. Live pulls the last 6 months from Yahoo Finance.",
            )

            start_btn = gr.Button(
                "▶  Start Trading",
                variant="primary",
                size="lg",
            )

            gr.Markdown("### 📋 Trading Log")
            log_box = gr.Textbox(
                label="",
                lines=22,
                max_lines=40,
                show_copy_button=True,
                interactive=False,
            )

        # ── Right Column: Outputs ─────────────────────────────────────────
        with gr.Column(scale=2, min_width=500):
            gr.Markdown("### 📈 Performance Chart")
            chart = gr.Plot(label="")

            gr.Markdown("### 🏁 Episode Summary")
            summary_box = gr.Textbox(
                label="",
                lines=14,
                interactive=False,
                show_copy_button=True,
            )

    # Wire button
    start_btn.click(
        fn=run_episode,
        inputs=[mode_radio],
        outputs=[log_box, chart, summary_box],
    )

    gr.Markdown(
        """
        ---
        <center>
        <small>Built by <b>Atharva</b> — Mumbai, India 🇮🇳 &nbsp;|&nbsp; OpenEnv v0.2.3 &nbsp;|&nbsp;
        <a href="https://huggingface.co/spaces/20atharva26/finance-env" target="_blank">HuggingFace Space</a></small>
        </center>
        """,
    )

# Allow standalone testing: python app.py
if __name__ == "__main__":
    gradio_app.launch()