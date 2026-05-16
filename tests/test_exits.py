"""Tests for pure exit-helper functions in trader.risk.exits.

These helpers must be deterministic and side-effect-free — every test here
runs in microseconds without any DB, broker, or filesystem dependency.
"""

from __future__ import annotations

from datetime import date

import pytest

from trader.risk.exits import (
    atr_stop_pct,
    compute_atr,
    is_in_cooldown,
    next_partial_milestone,
    trailing_lock_price,
)


# ── compute_atr ───────────────────────────────────────────────────────────────

class TestComputeAtr:

    def test_returns_none_when_insufficient_bars(self):
        highs = [10, 11, 12]
        lows = [9, 10, 11]
        closes = [9.5, 10.5, 11.5]
        assert compute_atr(highs, lows, closes, period=14) is None

    def test_returns_none_at_exactly_period_bars(self):
        # Need period + 1 bars (one prev close required)
        highs = [10] * 14
        lows = [9] * 14
        closes = [9.5] * 14
        assert compute_atr(highs, lows, closes, period=14) is None

    def test_basic_atr_constant_range(self):
        # Constant range of 1.0 each day, no gaps → ATR should be 1.0
        highs = [11.0] * 20
        lows = [10.0] * 20
        closes = [10.5] * 20
        atr = compute_atr(highs, lows, closes, period=14)
        assert atr == pytest.approx(1.0, abs=0.01)

    def test_gap_up_increases_tr(self):
        # Gap up from prev_close 10 to next bar high 15 → TR = 5
        highs = [10] * 14 + [15.0]
        lows = [9] * 14 + [14.0]
        closes = [10] * 14 + [14.5]
        atr = compute_atr(highs, lows, closes, period=14)
        # Most days TR = 1 (10-9). Last day TR = max(15-14, |15-10|, |14-10|) = 5.
        # Last-period mean spans days 2..15: 13 days at TR=1 + 1 day at TR=5 → mean = 18/14 ≈ 1.286
        assert atr == pytest.approx(18 / 14, abs=0.01)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            compute_atr([1, 2, 3], [1, 2], [1, 2, 3], period=2)

    def test_smaller_period(self):
        highs = [11, 12, 13, 14, 15]
        lows = [10, 11, 12, 13, 14]
        closes = [10.5, 11.5, 12.5, 13.5, 14.5]
        # Each day: range = 1.0, plus gap from prev close = 1.0 → TR = max(1, 2.5, 1.5) for day 2 onwards
        atr = compute_atr(highs, lows, closes, period=3)
        assert atr is not None
        assert atr > 1.0  # gaps push TR above pure range


# ── atr_stop_pct ──────────────────────────────────────────────────────────────

class TestAtrStopPct:

    def test_normal_range_uses_2x_atr(self):
        # ATR = 2, close = 100, 2x ATR = 4 → 4% → clamped to floor 5%
        assert atr_stop_pct(atr=2.0, close=100.0) == 0.05

    def test_high_atr_clamps_to_cap(self):
        # ATR = 10, close = 100, 2x = 20% → cap at 12%
        assert atr_stop_pct(atr=10.0, close=100.0) == 0.12

    def test_mid_range_atr_passes_through(self):
        # ATR = 4, close = 100, 2x = 8% → between floor and cap
        assert atr_stop_pct(atr=4.0, close=100.0) == pytest.approx(0.08)

    def test_zero_close_returns_floor(self):
        assert atr_stop_pct(atr=1.0, close=0.0) == 0.05

    def test_zero_atr_returns_floor(self):
        assert atr_stop_pct(atr=0.0, close=100.0) == 0.05

    def test_custom_bounds_respected(self):
        # ATR = 3, close = 100, 2x = 6% → cap at 4%
        assert atr_stop_pct(atr=3.0, close=100.0, floor=0.02, cap=0.04) == 0.04
        # Same ATR with wider band → passes through
        assert atr_stop_pct(atr=3.0, close=100.0, floor=0.02, cap=0.10) == pytest.approx(0.06)

    def test_custom_multiplier(self):
        # 3x ATR with ATR=2, close=100 → 6%, between floor and cap
        assert atr_stop_pct(atr=2.0, close=100.0, multiplier=3.0) == pytest.approx(0.06)


# ── trailing_lock_price ───────────────────────────────────────────────────────

class TestTrailingLockPrice:

    def test_below_activation_returns_none(self):
        # Peak +10% < 15% activation → not yet armed
        assert trailing_lock_price(entry_price=100.0, peak_pct=0.10) is None

    def test_exactly_at_activation_arms_lock(self):
        # Peak +15%, lock 50% → stop locks +7.5% gain = $107.5
        assert trailing_lock_price(entry_price=100.0, peak_pct=0.15) == pytest.approx(107.5)

    def test_high_peak_locks_higher(self):
        # Peak +40%, lock 50% → stop locks +20% gain = $120
        assert trailing_lock_price(entry_price=100.0, peak_pct=0.40) == pytest.approx(120.0)

    def test_custom_lock_ratio(self):
        # Peak +30%, lock 75% → stop locks +22.5% = $122.50
        assert trailing_lock_price(
            entry_price=100.0, peak_pct=0.30, lock_ratio=0.75
        ) == pytest.approx(122.5)

    def test_custom_activation(self):
        # Peak +20%, activation set to 25% → not armed
        assert trailing_lock_price(
            entry_price=100.0, peak_pct=0.20, activation_pct=0.25
        ) is None
        # Peak +30%, same → armed at 15% gain locked
        assert trailing_lock_price(
            entry_price=100.0, peak_pct=0.30, activation_pct=0.25
        ) == pytest.approx(115.0)

    def test_scales_with_entry_price(self):
        # Higher-priced stock: peak +20%, lock 50% → stop locks +10%
        assert trailing_lock_price(entry_price=500.0, peak_pct=0.20) == pytest.approx(550.0)


# ── next_partial_milestone ────────────────────────────────────────────────────

class TestNextPartialMilestone:

    def test_no_gain_returns_none(self):
        assert next_partial_milestone(current_gain_pct=0.10) is None

    def test_first_milestone_fires_at_25(self):
        threshold, fraction = next_partial_milestone(current_gain_pct=0.25)
        assert threshold == pytest.approx(0.25)
        assert fraction == pytest.approx(1 / 3)

    def test_above_first_milestone_returns_first_if_unsold(self):
        # At +30%, both milestones not yet sold → return the 25% one first
        threshold, _ = next_partial_milestone(current_gain_pct=0.30)
        assert threshold == pytest.approx(0.25)

    def test_first_milestone_sold_returns_second_when_eligible(self):
        # 25% already sold, gain at 50% → return the 50% milestone
        threshold, fraction = next_partial_milestone(
            current_gain_pct=0.50, sold_milestones=[0.25]
        )
        assert threshold == pytest.approx(0.50)
        assert fraction == pytest.approx(1 / 3)

    def test_first_sold_but_under_second_returns_none(self):
        assert next_partial_milestone(
            current_gain_pct=0.40, sold_milestones=[0.25]
        ) is None

    def test_both_sold_returns_none(self):
        assert next_partial_milestone(
            current_gain_pct=0.80, sold_milestones=[0.25, 0.50]
        ) is None

    def test_custom_milestones(self):
        # Single 100% milestone, sell half
        milestones = [(1.0, 0.5)]
        assert next_partial_milestone(current_gain_pct=0.50, milestones=milestones) is None
        threshold, fraction = next_partial_milestone(
            current_gain_pct=1.0, milestones=milestones
        )
        assert threshold == pytest.approx(1.0)
        assert fraction == pytest.approx(0.5)


# ── is_in_cooldown ────────────────────────────────────────────────────────────

class TestIsInCooldown:

    def test_no_prior_stop_not_in_cooldown(self):
        assert is_in_cooldown("AAPL", last_stop_date=None) is False

    def test_stop_today_is_in_cooldown(self):
        today = date(2026, 5, 14)  # Thursday
        assert is_in_cooldown("AAPL", last_stop_date=today, today=today) is True

    def test_stop_yesterday_still_in_cooldown(self):
        today = date(2026, 5, 14)  # Thursday
        yesterday = date(2026, 5, 13)  # Wednesday (1 trading day elapsed)
        assert is_in_cooldown("AAPL", last_stop_date=yesterday, today=today) is True

    def test_exactly_5_trading_days_clears_cooldown(self):
        # Stop on Mon May 4 → 5 trading days = Tue May 5, Wed 6, Thu 7, Fri 8, Mon 11
        # On Mon May 11, exactly 5 trading days have elapsed → no longer in cooldown
        last = date(2026, 5, 4)   # Monday
        today = date(2026, 5, 11)  # Monday (5 trading days later)
        assert is_in_cooldown("AAPL", last_stop_date=last, today=today) is False

    def test_4_trading_days_still_in_cooldown(self):
        last = date(2026, 5, 4)   # Monday
        today = date(2026, 5, 8)  # Friday — 4 trading days elapsed
        assert is_in_cooldown("AAPL", last_stop_date=last, today=today) is True

    def test_weekend_does_not_count_as_trading_days(self):
        # Stop on Fri May 8 → next trading day is Mon May 11 (1 elapsed)
        last = date(2026, 5, 8)    # Friday
        today = date(2026, 5, 11)  # Monday
        assert is_in_cooldown("AAPL", last_stop_date=last, today=today) is True

    def test_accepts_iso_string(self):
        today = date(2026, 5, 14)
        assert is_in_cooldown("AAPL", last_stop_date="2026-05-13", today=today) is True

    def test_invalid_string_treated_as_no_cooldown(self):
        today = date(2026, 5, 14)
        assert is_in_cooldown("AAPL", last_stop_date="not-a-date", today=today) is False

    def test_future_stop_date_not_in_cooldown(self):
        # Defensive: if for some reason last_stop is in the future, don't block
        today = date(2026, 5, 14)
        future = date(2026, 5, 20)
        assert is_in_cooldown("AAPL", last_stop_date=future, today=today) is False

    def test_custom_cooldown_days(self):
        last = date(2026, 5, 4)    # Monday
        today = date(2026, 5, 6)   # Wednesday — 2 trading days elapsed
        assert is_in_cooldown("AAPL", last_stop_date=last, today=today, cooldown_days=2) is False
        assert is_in_cooldown("AAPL", last_stop_date=last, today=today, cooldown_days=3) is True
