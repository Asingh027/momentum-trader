"""Momentum / trend-following signal computation.

v1.0 entry (ALL must be true at close, order fills at next open):
  1. Price > N-day high (breakout_lookback bars, default 126 ~ 6 months)
  2. Price > 50d SMA                     (short-term trend)
  3. 50d SMA > 200d SMA                  (golden cross — confirmed uptrend)
  4. Price > 200d SMA                    (long-term trend)
  5. 63-day stock return > SPY 63-day return + min_outperformance  (RS filter)
  6. SPY > SPY 200d SMA                  (market regime gate)
  7. No earnings within next earnings_buffer_days trading days

v1.0 exit (ANY fires first):
  1. Price closes BELOW trailing_sma_days SMA   (PRIMARY — trailing stop)
  2. Price <= entry * (1 - hard_stop_pct)        (-8% hard floor)
  3. SPY closes below its 200d SMA               (regime stop)

No profit target, no time stop.
sl_stop = hard_stop_pct handled by vectorbt Portfolio.
compute_exit_signals() returns trailing SMA cross + regime stop.
"""

from __future__ import annotations

import logging

import pandas as pd

from trader.config import TradingConfig

logger = logging.getLogger(__name__)


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def compute_entry_signals(
    bars: dict[str, pd.DataFrame],
    spy_bars: pd.DataFrame,
    cfg: TradingConfig,
    earnings_dates: dict[str, list[pd.Timestamp]] | None = None,
) -> pd.DataFrame:
    """Return boolean DataFrame (dates x tickers). True = entry signal.

    All 7 conditions must be True simultaneously.
    """
    spy_close = spy_bars["Close"]
    spy_sma200 = _sma(spy_close, cfg.spy_trend_sma_period)
    spy_above_sma = spy_close > spy_sma200

    # Relative strength baseline: SPY N-day return
    rs_lb = cfg.relative_strength_lookback
    spy_ret_rs = spy_close / spy_close.shift(rs_lb) - 1

    all_dates = spy_bars.index
    signals: dict[str, pd.Series] = {}

    for ticker, df in bars.items():
        try:
            close = df["Close"].reindex(all_dates).ffill(limit=5)

            # 1. Price > N-day high (previous N bars, not including current)
            rolling_high = close.shift(1).rolling(cfg.breakout_lookback).max()
            breakout = close > rolling_high

            # 2. Price > 50d SMA
            sma50 = _sma(close, cfg.trend_sma_short)
            above_sma50 = close > sma50

            # 3. 50d SMA > 200d SMA (golden cross)
            sma200 = _sma(close, cfg.trend_sma_period)
            golden_cross = sma50 > sma200

            # 4. Price > 200d SMA
            above_sma200 = close > sma200

            # 5. Relative strength vs SPY
            stock_ret_rs = close / close.shift(rs_lb) - 1
            rs_ok = (stock_ret_rs - spy_ret_rs.reindex(all_dates)) >= cfg.relative_strength_min_outperformance

            # 6. Regime gate
            if cfg.use_regime_gate:
                spy_ok = spy_above_sma.reindex(all_dates).fillna(False)
            else:
                spy_ok = pd.Series(True, index=all_dates)

            # 7. Earnings gate
            if earnings_dates and ticker in earnings_dates:
                dates_arr = pd.DatetimeIndex(earnings_dates[ticker])
                buffer_days = cfg.earnings_buffer_days

                def _no_earnings_soon(date: pd.Timestamp) -> bool:
                    horizon = date + pd.Timedelta(days=buffer_days * 2)
                    upcoming = dates_arr[(dates_arr > date) & (dates_arr <= horizon)]
                    return len(upcoming) == 0

                no_earnings = pd.Series(
                    [_no_earnings_soon(d) for d in all_dates], index=all_dates
                )
            else:
                no_earnings = pd.Series(True, index=all_dates)

            entry = (
                breakout
                & above_sma50
                & golden_cross
                & above_sma200
                & rs_ok
                & spy_ok
                & no_earnings
            ).fillna(False)

            signals[ticker] = entry

        except Exception as exc:
            logger.warning("Momentum entry signal failed for %s: %s", ticker, exc)
            signals[ticker] = pd.Series(False, index=all_dates)

    return pd.DataFrame(signals)


def compute_exit_signals(
    bars: dict[str, pd.DataFrame],
    spy_bars: pd.DataFrame,
    cfg: TradingConfig,
) -> pd.DataFrame:
    """Return boolean exit DataFrame (dates x tickers).

    Fires when:
    - Price closes BELOW trailing SMA (trailing_sma_days)
    - SPY closes below its 200d SMA (regime stop)

    Hard stop (hard_stop_pct) is handled by vectorbt sl_stop.
    """
    spy_close = spy_bars["Close"]
    spy_sma200 = _sma(spy_close, cfg.spy_trend_sma_period)
    spy_below_sma = (spy_close < spy_sma200).rename("spy_regime_exit")

    all_dates = spy_bars.index
    exits: dict[str, pd.Series] = {}

    for ticker, df in bars.items():
        regime_exit = spy_below_sma.reindex(all_dates).fillna(False)

        if df is not None:
            try:
                close = df["Close"].reindex(all_dates).ffill(limit=5)
                trailing_sma = _sma(close, cfg.trailing_sma_days)
                # Exit when price crosses BELOW trailing SMA
                below_trailing = close < trailing_sma
                exits[ticker] = (below_trailing | regime_exit).fillna(False)
            except Exception as exc:
                logger.warning("Momentum exit signal failed for %s: %s", ticker, exc)
                exits[ticker] = regime_exit
        else:
            exits[ticker] = regime_exit

    return pd.DataFrame(exits)


def compute_trailing_sma_series(
    bars: dict[str, pd.DataFrame],
    all_dates: pd.DatetimeIndex,
    period: int,
) -> dict[str, pd.Series]:
    """Compute trailing SMA for each ticker. Exposed for exit attribution."""
    result = {}
    for ticker, df in bars.items():
        close = df["Close"].reindex(all_dates).ffill(limit=5)
        result[ticker] = _sma(close, period)
    return result
