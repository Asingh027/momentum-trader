"""Pure helper functions for advanced exit logic.

All functions in this module are deterministic and have no side effects —
they accept numeric inputs and return numeric outputs. This keeps them
trivially unit-testable without any broker, DB, or filesystem mocking.

The exit stack uses these helpers in two places:

  monitor.py (intraday, hourly)
    - ATR-based dynamic hard stop (replaces fixed 5%)
    - Trailing-lock check once a position is up 15%+
    - Peak tracking updated each check

  runner.py (EOD, 4:30 PM)
    - Partial profit-taking at +25% and +50%
    - Trailing-lock check (mirrors monitor)
    - Cooldown skip on entries when ticker recently stopped out
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Iterable, Optional

import pandas as pd


# ── ATR ───────────────────────────────────────────────────────────────────────

def compute_atr(
    highs: Iterable[float],
    lows: Iterable[float],
    closes: Iterable[float],
    period: int = 14,
) -> Optional[float]:
    """Return the Wilder-smoothed Average True Range over the last `period` bars.

    True Range = max(high-low, |high - prev_close|, |low - prev_close|).
    ATR is the simple mean of TR over `period` bars (Wilder's classic formulation
    uses an EMA; we use SMA for simplicity — the difference is negligible for
    sizing stops over the timeframe we care about).

    Returns None if fewer than `period + 1` bars are available.
    """
    h = pd.Series(list(highs), dtype="float64")
    l = pd.Series(list(lows), dtype="float64")
    c = pd.Series(list(closes), dtype="float64")

    if len(h) != len(l) or len(h) != len(c):
        raise ValueError("highs, lows, closes must all be the same length")
    if len(c) < period + 1:
        return None

    prev_close = c.shift(1)
    tr = pd.concat(
        [
            (h - l).abs(),
            (h - prev_close).abs(),
            (l - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.iloc[-period:].mean()
    if pd.isna(atr):
        return None
    return float(atr)


def atr_stop_pct(
    atr: float,
    close: float,
    floor: float = 0.05,
    cap: float = 0.12,
    multiplier: float = 2.0,
) -> float:
    """Return an ATR-based stop distance as a fraction of close price.

    Stop distance = (ATR * multiplier) / close, clamped to [floor, cap].
    A 2x ATR multiplier is the textbook default — wide enough to ignore noise
    but tight enough to catch real breakdowns.
    """
    if close <= 0 or atr <= 0:
        return floor
    raw = (atr * multiplier) / close
    if raw < floor:
        return floor
    if raw > cap:
        return cap
    return raw


# ── Trailing lock ─────────────────────────────────────────────────────────────

def trailing_lock_price(
    entry_price: float,
    peak_pct: float,
    lock_ratio: float = 0.5,
    activation_pct: float = 0.15,
) -> Optional[float]:
    """Return the trailing stop price once activation is reached, else None.

    Once a position has reached `activation_pct` gain at its peak, we lock in
    `lock_ratio` of that peak gain. e.g. peak +30%, lock_ratio 0.5 → stop at +15%.

    Parameters
    ----------
    entry_price : float
        Original fill price.
    peak_pct : float
        Highest unrealized gain seen so far, as a fraction (0.30 = +30%).
    lock_ratio : float, default 0.5
        Fraction of peak gain to lock in.
    activation_pct : float, default 0.15
        Trailing lock is dormant below this peak gain.

    Returns None when peak_pct < activation_pct (lock not yet armed).
    """
    if peak_pct < activation_pct:
        return None
    locked_gain = peak_pct * lock_ratio
    return entry_price * (1.0 + locked_gain)


# ── Partial profit-taking ─────────────────────────────────────────────────────

def next_partial_milestone(
    current_gain_pct: float,
    sold_milestones: Optional[Iterable[float]] = None,
    milestones: Optional[list[tuple[float, float]]] = None,
) -> Optional[tuple[float, float]]:
    """Return (threshold, fraction) for the next partial sell that should fire.

    Walks `milestones` in order and returns the first (threshold, fraction)
    pair where current_gain_pct >= threshold and threshold not already in
    `sold_milestones`. Returns None if all milestones already sold or none
    reached.

    Default milestones: sell 1/3 at +25%, sell 1/3 at +50%.
    """
    if milestones is None:
        milestones = [(0.25, 1 / 3), (0.50, 1 / 3)]
    sold = set(sold_milestones or [])

    for threshold, fraction in milestones:
        if threshold in sold:
            continue
        if current_gain_pct >= threshold:
            return (threshold, fraction)
    return None


# ── Cooldown ──────────────────────────────────────────────────────────────────

def is_in_cooldown(
    ticker: str,
    last_stop_date: Optional[date | str],
    cooldown_days: int = 5,
    today: Optional[date] = None,
) -> bool:
    """Return True if `ticker` was stopped out within the last `cooldown_days` trading days.

    `last_stop_date` is the date of the most recent stop-loss exit (or None if never).
    We approximate trading days as weekdays — close enough for a 5-day cooldown that
    doesn't need market-calendar precision.

    Parameters
    ----------
    ticker : str
        Only used for diagnostics; the cooldown logic is purely date-based.
    last_stop_date : date | str | None
        ISO-format string or date object. None means no prior stop.
    cooldown_days : int
        Number of trading days the ticker stays in cooldown.
    today : date, optional
        Override for testing. Defaults to date.today().
    """
    if last_stop_date is None:
        return False

    if isinstance(last_stop_date, str):
        try:
            last = datetime.fromisoformat(last_stop_date).date()
        except ValueError:
            return False
    else:
        last = last_stop_date

    today = today or date.today()
    if today < last:
        return False

    trading_days_elapsed = 0
    cursor = last
    while cursor < today:
        cursor = cursor + timedelta(days=1)
        if cursor.weekday() < 5:
            trading_days_elapsed += 1

    return trading_days_elapsed < cooldown_days
