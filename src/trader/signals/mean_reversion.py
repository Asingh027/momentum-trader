"""Mean-reversion dip-buy signal computation.

v0.3 entry (ALL must be true at close, order fills at next open):
  1. Price > 200d SMA                     (long-term trend filter)
  2. Price down >= pullback_pct from 10d high  (pullback)
  3. RSI(14) < rsi_oversold               (oversold — 35 for v0.3)
  4. SPY > SPY 200d SMA                   (market regime gate)
  NOTE: volume filter and earnings gate are toggled off for v0.3 (ETF universe)

v0.3 exit (ANY fires first):
  1. RSI(14) crosses ABOVE rsi_exit_level (50 for v0.3 — mean-reversion confirmed)
  2. Price >= entry * (1 + profit_target_pct)  (+8% backup target)
  3. Price <= entry * (1 - stop_loss_pct)      (-5% stop loss)
  4. Position held >= time_stop_days           (15-day time stop)
  5. SPY closes below its 200d SMA             (regime stop)

For vectorbt: sl_stop / tp_stop handle exits 3 and 4 at the portfolio level.
compute_exit_signals() returns the signal-level exits (RSI crossover + regime stop).
Time stop is injected as explicit exit signals via engine.add_time_stop_exits().
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
    rsi = pd.Series(index=close.index, dtype=float)
    has_loss = avg_loss > 0
    rs = avg_gain[has_loss] / avg_loss[has_loss]
    rsi[has_loss] = 100 - (100 / (1 + rs))
    rsi[~has_loss & avg_gain.notna() & (avg_gain >= 0)] = 100.0
    return rsi


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def compute_rsi_series(
    bars: dict[str, pd.DataFrame],
    all_dates: pd.DatetimeIndex,
    rsi_period: int,
) -> dict[str, pd.Series]:
    """Compute RSI for each ticker. Returns dict ticker -> RSI Series.

    Exposed separately so run_v03.py can reuse for exit reason attribution.
    """
    result = {}
    for ticker, df in bars.items():
        close = df["Close"].reindex(all_dates).ffill(limit=5)
        result[ticker] = _rsi(close, rsi_period)
    return result


def compute_entry_signals(
    bars: dict[str, pd.DataFrame],
    spy_bars: pd.DataFrame,
    cfg: TradingConfig,
    earnings_dates: dict[str, list[pd.Timestamp]] | None = None,
) -> pd.DataFrame:
    """Return boolean DataFrame (dates × tickers). True = entry signal.

    Respects cfg.use_volume_filter and cfg.use_regime_gate toggles.
    """
    if earnings_dates is None and cfg.use_volume_filter:
        # Only warn about earnings if we're in equity mode (volume filter on)
        pass  # earnings gate always-pass for ETF universe; no warning needed

    spy_close = spy_bars["Close"]
    spy_sma = _sma(spy_close, cfg.spy_trend_sma_period)
    spy_above_sma = (spy_close > spy_sma).rename("spy_regime")

    all_dates = spy_bars.index
    signals: dict[str, pd.Series] = {}

    for ticker, df in bars.items():
        try:
            close = df["Close"].reindex(all_dates).ffill(limit=5)
            volume = df["Volume"].reindex(all_dates).ffill(limit=5)

            # 1. Price > trend SMA
            sma_trend = _sma(close, cfg.trend_sma_period)
            above_trend = close > sma_trend

            # 2. Pullback from N-day high
            high_nd = close.rolling(cfg.pullback_lookback).max()
            pulled_back = close <= high_nd * (1 - cfg.pullback_pct)

            # 3. RSI oversold
            rsi = _rsi(close, cfg.rsi_period)
            oversold = rsi < cfg.rsi_oversold

            # 4. Volume filter (optional)
            if cfg.use_volume_filter:
                vol_avg = volume.rolling(cfg.volume_avg_period).mean()
                high_volume = volume >= vol_avg * cfg.volume_ratio
            else:
                high_volume = pd.Series(True, index=all_dates)

            # 5. Earnings gate (always-pass when no data provided — ETFs have no earnings)
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

            # 6. SPY regime gate (optional)
            if cfg.use_regime_gate:
                spy_ok = spy_above_sma.reindex(all_dates)
            else:
                spy_ok = pd.Series(True, index=all_dates)

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
    """Return boolean exit DataFrame (dates × tickers).

    Includes:
    - SPY regime stop (SPY crosses below its 200d SMA)
    - RSI crossover exit: RSI crosses ABOVE cfg.rsi_exit_level (if > 0)
      Crossover = RSI[t-1] < level AND RSI[t] >= level

    sl_stop, tp_stop: handled by vectorbt portfolio-level args.
    Time stop: injected via engine.add_time_stop_exits().
    """
    spy_close = spy_bars["Close"]
    spy_sma = _sma(spy_close, cfg.spy_trend_sma_period)
    spy_below_sma = (spy_close < spy_sma).rename("spy_regime_exit")

    all_dates = spy_bars.index
    exits: dict[str, pd.Series] = {}

    use_rsi_exit = cfg.rsi_exit_level > 0

    for ticker, df in bars.items():
        # Base: regime stop (same for all tickers)
        regime_exit = spy_below_sma.reindex(all_dates).fillna(False)

        if use_rsi_exit and df is not None:
            try:
                close = df["Close"].reindex(all_dates).ffill(limit=5)
                rsi = _rsi(close, cfg.rsi_period)
                # Crossover: was below level, now at or above
                rsi_cross_above = (rsi >= cfg.rsi_exit_level) & (rsi.shift(1) < cfg.rsi_exit_level)
                rsi_cross_above = rsi_cross_above.fillna(False)
                exits[ticker] = (regime_exit | rsi_cross_above)
            except Exception as exc:
                logger.warning("RSI exit computation failed for %s: %s", ticker, exc)
                exits[ticker] = regime_exit
        else:
            exits[ticker] = regime_exit

    return pd.DataFrame(exits)
