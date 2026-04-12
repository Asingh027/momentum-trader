"""Equity curve chart with drawdown underlay.

Produces:
- outputs/equity_curve.png  — strategy vs SPY, drawdown underlay
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from trader.backtest.metrics import drawdown_series


def plot_equity_curve(
    strategy_equity: pd.Series,
    spy_equity: pd.Series,
    output_path: Path,
    title: str = "Equity Curve vs SPY (Buy & Hold)",
) -> None:
    """Save equity curve + drawdown chart to output_path."""
    strategy_norm = strategy_equity / strategy_equity.iloc[0]
    spy_norm = spy_equity / spy_equity.iloc[0]

    strategy_dd = drawdown_series(strategy_equity)
    spy_dd = drawdown_series(spy_equity)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ── Top: equity curves ────────────────────────────────────────────────────
    ax1.plot(strategy_norm.index, strategy_norm.values, label="Strategy", color="#2563eb", linewidth=1.8)
    ax1.plot(spy_norm.index, spy_norm.values, label="SPY B&H", color="#9ca3af", linewidth=1.2, linestyle="--")
    ax1.set_ylabel("Portfolio Value (normalised)")
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(1, color="black", linewidth=0.5, alpha=0.4)

    # Annotate final values
    ax1.annotate(
        f"{strategy_norm.iloc[-1]:.1%}",
        xy=(strategy_norm.index[-1], strategy_norm.iloc[-1]),
        xytext=(8, 0), textcoords="offset points",
        fontsize=9, color="#2563eb",
    )
    ax1.annotate(
        f"{spy_norm.iloc[-1]:.1%}",
        xy=(spy_norm.index[-1], spy_norm.iloc[-1]),
        xytext=(8, 0), textcoords="offset points",
        fontsize=9, color="#6b7280",
    )

    # ── Bottom: drawdown ──────────────────────────────────────────────────────
    ax2.fill_between(strategy_dd.index, strategy_dd.values, 0, alpha=0.4, color="#ef4444", label="Strategy DD")
    ax2.fill_between(spy_dd.index, spy_dd.values, 0, alpha=0.2, color="#9ca3af", label="SPY DD")
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Date")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved -> {output_path}")
