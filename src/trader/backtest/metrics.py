"""Extract and format backtest metrics.

Computes all required metrics plus SPY buy-and-hold benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualised_return(equity: pd.Series) -> float:
    """CAGR from equity curve."""
    if len(equity) < 2:
        return float("nan")
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def total_return(equity: pd.Series) -> float:
    if equity.iloc[0] == 0:
        return float("nan")
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised Sharpe (risk-free = 0)."""
    if returns.std() == 0:
        return float("nan")
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised Sortino (risk-free = 0, downside only)."""
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("nan")
    return float(returns.mean() / downside.std() * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction."""
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return float(dd.min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    roll_max = equity.cummax()
    return (equity - roll_max) / roll_max


def compute_trade_stats(trades: pd.DataFrame) -> dict:
    """Extract win rate, avg win/loss, trade count, avg hold from a trades DataFrame.

    Expects columns: pnl (or return), duration (in days or bars).
    vectorbt returns a Trades accessor — caller should pass .records_readable or .stats().
    """
    if trades is None or len(trades) == 0:
        return {
            "trade_count": 0,
            "win_rate": float("nan"),
            "avg_win": float("nan"),
            "avg_loss": float("nan"),
            "avg_hold_days": float("nan"),
        }

    # Handle vectorbt records_readable format
    if "Return" in trades.columns:
        rets = trades["Return"]
    elif "PnL" in trades.columns:
        rets = trades["PnL"]
    else:
        rets = pd.Series(dtype=float)

    wins = rets[rets > 0]
    losses = rets[rets < 0]

    # Compute avg hold: from explicit Duration col, or from timestamp diff
    avg_hold = float("nan")
    hold_col = None
    for c in ["Duration", "duration", "Bars Held", "bars_held"]:
        if c in trades.columns:
            hold_col = c
            break

    if hold_col:
        raw = trades[hold_col]
        # If timedelta, convert to days
        if hasattr(raw.iloc[0], "days"):
            avg_hold = float(raw.apply(lambda x: x.days if hasattr(x, "days") else float("nan")).mean())
        else:
            avg_hold = float(raw.mean())
    elif "Entry Timestamp" in trades.columns and "Exit Timestamp" in trades.columns:
        try:
            entry_ts = pd.to_datetime(trades["Entry Timestamp"])
            exit_ts = pd.to_datetime(trades["Exit Timestamp"])
            hold_td = (exit_ts - entry_ts).dt.days
            avg_hold = float(hold_td.mean())
        except Exception:
            pass

    return {
        "trade_count": len(trades),
        "win_rate": float(len(wins) / len(trades)) if len(trades) > 0 else float("nan"),
        "avg_win": float(wins.mean()) if len(wins) > 0 else float("nan"),
        "avg_loss": float(losses.mean()) if len(losses) > 0 else float("nan"),
        "avg_hold_days": avg_hold,
    }


def spy_benchmark(spy_bars: pd.DataFrame, start: str, end: str, capital: float) -> dict:
    """SPY buy-and-hold metrics over the backtest window."""
    spy = spy_bars["Close"].loc[start:end]
    equity = spy / spy.iloc[0] * capital
    daily_ret = equity.pct_change().dropna()
    return {
        "total_return": total_return(equity),
        "cagr": annualised_return(equity),
        "sharpe": sharpe_ratio(daily_ret),
        "sortino": sortino_ratio(daily_ret),
        "max_drawdown": max_drawdown(equity),
        "equity": equity,
    }


def format_metrics_table(strategy: dict, benchmark: dict) -> str:
    """Return a markdown metrics table."""
    rows = [
        ("Metric", "Strategy", "SPY B&H"),
        ("---", "---", "---"),
        ("Total Return", f"{strategy.get('total_return', float('nan')):.1%}", f"{benchmark.get('total_return', float('nan')):.1%}"),
        ("CAGR", f"{strategy.get('cagr', float('nan')):.1%}", f"{benchmark.get('cagr', float('nan')):.1%}"),
        ("Sharpe Ratio", f"{strategy.get('sharpe', float('nan')):.2f}", f"{benchmark.get('sharpe', float('nan')):.2f}"),
        ("Sortino Ratio", f"{strategy.get('sortino', float('nan')):.2f}", f"{benchmark.get('sortino', float('nan')):.2f}"),
        ("Max Drawdown", f"{strategy.get('max_drawdown', float('nan')):.1%}", f"{benchmark.get('max_drawdown', float('nan')):.1%}"),
        ("Win Rate", f"{strategy.get('win_rate', float('nan')):.1%}", "—"),
        ("Avg Win", f"{strategy.get('avg_win', float('nan')):.1%}", "—"),
        ("Avg Loss", f"{strategy.get('avg_loss', float('nan')):.1%}", "—"),
        ("Trade Count", str(strategy.get('trade_count', 0)), "—"),
        ("Avg Hold (days)", f"{strategy.get('avg_hold_days', float('nan')):.1f}", "—"),
    ]
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)
