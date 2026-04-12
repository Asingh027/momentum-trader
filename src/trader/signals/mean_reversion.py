"""Mean-reversion dip-buy signal computation.

Entry (ALL must be true at close, order placed at next open):
  1. Price > 200d SMA                (long-term trend filter)
  2. Price down >= 5% from 10d high  (pullback)
  3. RSI(14) < 30                    (oversold)
  4. Volume >= 1.5x 20d avg          (capitulation)
  5. No earnings in next 5 days      (earnings gate — best-effort via yfinance)
  6. SPY > SPY 200d SMA              (market regime gate)

Exit (ANY fires first):
  1. Price >= entry * (1 + profit_target_pct)     (+10% profit target)
  2. Price <= entry * (1 - stop_loss_pct)          (-5% stop loss)
  3. Position held >= time_stop_days               (10-day time stop)
  4. SPY < SPY 200d SMA                            (regime stop)

For vectorbt, we return entry/exit *signal* DataFrames (dates × tickers).
Stop-loss and profit-target exits are handled via vectorbt's sl_stop / tp_stop
portfolio arguments, not in the signal DataFrame.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from trader.config import TradingConfig

logger = logging.getLogger(__name__)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder RSI — matches TradingView / most references."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    # When avg_loss == 0, RSI = 100 (no losses at all)
    rsi = pd.Series(index=close.index, dtype=float)
    has_loss = avg_loss > 0
    rs = avg_gain[has_loss] / avg_loss[has_loss]
    rsi[has_loss] = 100 - (100 / (1 + rs))
    rsi[~has_loss & avg_gain.notna() & (avg_gain >= 0)] = 100.0
    return rsi


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def compute_entry_signals(
    bars: dict[str, pd.DataFrame],
    spy_bars: pd.DataFrame,
    cfg: TradingConfig,
    earnings_dates: dict[str, list[pd.Timestamp]] | None = None,
) -> pd.DataFrame:
    """Return boolean DataFrame (dates × tickers).

    True on dates where ALL entry conditions are met.
    Order fills at next day's open.

    earnings_dates: optional dict of ticker -> list of known earnings dates.
    If None, earnings gate is skipped (with a warning).
    """
    if earnings_dates is None:
        logger.warning("No earnings dates provided — earnings gate DISABLED. Results will be optimistic.")

    # Build SPY regime mask
    spy_close = spy_bars["Close"]
    spy_sma = _sma(spy_close, cfg.spy_trend_sma_period)
    spy_above_sma = (spy_close > spy_sma).rename("spy_regime")

    all_dates = spy_bars.index
    signals: dict[str, pd.Series] = {}

    for ticker, df in bars.items():
        try:
            close = df["Close"].reindex(all_dates).ffill(limit=5)
            volume = df["Volume"].reindex(all_dates).ffill(limit=5)

            # 1. Price > 200d SMA
            sma200 = _sma(close, cfg.trend_sma_period)
            above_trend = close > sma200

            # 2. Down >= 5% from 10d high
            high_10d = close.rolling(cfg.pullback_lookback).max()
            pulled_back = close <= high_10d * (1 - cfg.pullback_pct)

            # 3. RSI(14) < 30
            rsi = _rsi(close, cfg.rsi_period)
            oversold = rsi < cfg.rsi_oversold

            # 4. Volume >= 1.5x 20d avg
            vol_avg = volume.rolling(cfg.volume_avg_period).mean()
            high_volume = volume >= vol_avg * cfg.volume_ratio

            # 5. Earnings gate
            if earnings_dates and ticker in earnings_dates:
                dates_arr = pd.DatetimeIndex(earnings_dates[ticker])

                def _no_earnings_next_n(date: pd.Timestamp) -> bool:
                    horizon = date + pd.Timedelta(days=cfg.earnings_buffer_days * 2)
                    upcoming = dates_arr[(dates_arr > date) & (dates_arr <= horizon)]
                    return len(upcoming) == 0

                no_earnings = pd.Series(
                    [_no_earnings_next_n(d) for d in all_dates], index=all_dates
                )
            else:
                no_earnings = pd.Series(True, index=all_dates)

            # 6. SPY regime
            spy_ok = spy_above_sma.reindex(all_dates)

            entry = (
                above_trend
                & pulled_back
                & oversold
                & high_volume
                & no_earnings
                & spy_ok
            ).fillna(False)

            signals[ticker] = entry

        except Exception as exc:
            logger.warning("Signal computation failed for %s: %s", ticker, exc)
            signals[ticker] = pd.Series(False, index=all_dates)

    return pd.DataFrame(signals)


def compute_exit_signals(
    bars: dict[str, pd.DataFrame],
    spy_bars: pd.DataFrame,
    cfg: TradingConfig,
) -> pd.DataFrame:
    """Return boolean DataFrame for time-stop and regime-stop exits.

    Note: profit target (+10%) and stop-loss (-5%) are handled by vectorbt's
    tp_stop / sl_stop arguments at the portfolio level — they execute intra-bar
    which is more accurate than signal-level. This function handles:
    - SPY regime stop (SPY crosses below 200d SMA)

    Time-stop is handled in engine.py via max_open_trade_len.
    """
    spy_close = spy_bars["Close"]
    spy_sma = _sma(spy_close, cfg.spy_trend_sma_period)
    spy_below_sma = (spy_close < spy_sma).rename("spy_regime_exit")

    all_dates = spy_bars.index
    exits: dict[str, pd.Series] = {}

    for ticker in bars:
        exits[ticker] = spy_below_sma.reindex(all_dates).fillna(False)

    return pd.DataFrame(exits)
