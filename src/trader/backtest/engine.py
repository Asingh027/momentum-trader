"""Backtest engine — vectorbt portfolio construction.

Why vectorbt:
- Vectorized operations run a 5-year daily backtest in seconds
- Native sl_stop / tp_stop arguments handle profit target / stop loss exits
  without writing an event loop
- Portfolio.from_signals() maps cleanly to our entry/exit boolean DataFrames
- Built-in Sharpe, Sortino, drawdown, trades accessor

Limitations / gotchas:
- vectorbt processes all tickers with cash_sharing=True, respecting portfolio-level
  cash constraints across all concurrent positions.
- sl_stop and tp_stop are applied at bar close in vectorbt's default mode.
  In reality these would trigger intraday. This slightly overstates stop accuracy.
- Time stop is implemented as explicit exit signals (not a native vectorbt arg)
  because max_open_trade_len is not available in vectorbt 0.28.
- size_type="percent" allocates each position as a fraction of current equity.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import vectorbt as vbt

from trader.config import TradingConfig

logger = logging.getLogger(__name__)


def add_time_stop_exits(
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    time_stop_days: int,
) -> pd.DataFrame:
    """For each entry signal at bar T, inject an exit signal at bar T + time_stop_days.

    This implements the spec's 10-day time stop in signal space.
    Existing exit signals are preserved (OR-merged).
    """
    result = exits.copy()
    all_dates = entries.index

    for col in entries.columns:
        entry_dates = all_dates[entries[col].values]
        for entry_date in entry_dates:
            idx_pos = all_dates.get_loc(entry_date)
            exit_pos = idx_pos + time_stop_days
            if exit_pos < len(all_dates):
                exit_date = all_dates[exit_pos]
                result.at[exit_date, col] = True

    return result


def run_backtest(
    close: pd.DataFrame,
    open_prices: pd.DataFrame,
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    cfg: TradingConfig,
) -> vbt.Portfolio:
    """Run a single backtest window and return a vectorbt Portfolio.

    Parameters
    ----------
    close : pd.DataFrame
        Adjusted close prices (dates × tickers)
    open_prices : pd.DataFrame
        Open prices used for entry fills (next-bar open after signal)
    entries : pd.DataFrame
        Boolean entry signals (dates × tickers)
    exits : pd.DataFrame
        Boolean exit signals — regime stop + time stop (dates × tickers)
    cfg : TradingConfig
    """
    common_idx = close.index
    common_cols = close.columns

    entries = entries.reindex(index=common_idx, columns=common_cols, fill_value=False)
    exits = exits.reindex(index=common_idx, columns=common_cols, fill_value=False)
    open_prices = open_prices.reindex(index=common_idx, columns=common_cols, method="ffill")

    # Inject time-stop exits
    exits = add_time_stop_exits(entries, exits, cfg.time_stop_days)

    slippage = cfg.slippage_rate

    portfolio = vbt.Portfolio.from_signals(
        close=close,
        open=open_prices,
        entries=entries,
        exits=exits,
        price=open_prices,
        size=cfg.target_position_pct,
        size_type="percent",
        sl_stop=cfg.stop_loss_pct,
        tp_stop=cfg.profit_target_pct,
        fees=cfg.commission,
        slippage=slippage,
        init_cash=cfg.paper_capital,
        group_by=True,
        cash_sharing=True,
    )

    return portfolio


def aggregate_portfolio_equity(portfolio: vbt.Portfolio) -> pd.Series:
    """Return portfolio equity curve as a Series."""
    val = portfolio.value()
    if isinstance(val, pd.DataFrame):
        return val.iloc[:, 0]
    return val
