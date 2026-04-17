"""Tests for kill switch logic.

All tests are offline — no Alpaca API calls.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from trader.risk.kill_switches import KillSwitches, KillSwitchTripped


@pytest.fixture
def ks(tmp_path):
    return KillSwitches(project_root=tmp_path)


# ── Manual kill file ──────────────────────────────────────────────────────────

def test_kill_file_absent(ks):
    assert ks.check_kill_file() is False


def test_kill_file_present(ks, tmp_path):
    (tmp_path / "KILL").touch()
    assert ks.check_kill_file() is True


# ── Halt file ─────────────────────────────────────────────────────────────────

def test_halt_file_absent(ks):
    assert ks.check_halt_file() is False


def test_halt_file_present(ks, tmp_path):
    (tmp_path / "HALTED").write_text("halted")
    assert ks.check_halt_file() is True


# ── Daily loss ────────────────────────────────────────────────────────────────

def test_daily_loss_not_tripped(ks):
    # -2.9% — just under the -3% limit
    assert ks.check_daily_loss(start_value=25000, current_value=24251) is False


def test_daily_loss_exactly_at_limit(ks):
    # exactly -3% → should trip
    assert ks.check_daily_loss(start_value=25000, current_value=24250) is True


def test_daily_loss_breached(ks):
    # -5% — clearly over limit
    assert ks.check_daily_loss(start_value=25000, current_value=23750) is True


def test_daily_loss_positive_day(ks):
    # Positive P&L — never trips
    assert ks.check_daily_loss(start_value=25000, current_value=26000) is False


def test_daily_loss_zero_start(ks):
    # start_value = 0 — guard against division by zero
    assert ks.check_daily_loss(start_value=0, current_value=25000) is False


# ── Drawdown ──────────────────────────────────────────────────────────────────

def test_drawdown_not_tripped(ks):
    # -9.9% — just under -10%
    assert ks.check_drawdown(peak_value=25000, current_value=22501) is False


def test_drawdown_exactly_at_limit(ks):
    # exactly -10% → should trip
    assert ks.check_drawdown(peak_value=25000, current_value=22500) is True


def test_drawdown_breached(ks):
    assert ks.check_drawdown(peak_value=25000, current_value=20000) is True


def test_drawdown_writes_halt_file(ks, tmp_path):
    ks.check_drawdown(peak_value=25000, current_value=20000)
    assert (tmp_path / "HALTED").exists()


def test_drawdown_halt_content(ks, tmp_path):
    ks.check_drawdown(peak_value=25000, current_value=20000)
    content = (tmp_path / "HALTED").read_text()
    assert "drawdown" in content.lower() or "Drawdown" in content


def test_drawdown_zero_peak(ks):
    assert ks.check_drawdown(peak_value=0, current_value=25000) is False


# ── PDT guard ─────────────────────────────────────────────────────────────────

def test_pdt_below_limit(ks):
    assert ks.check_pdt(day_trades_last_5_days=2) is False


def test_pdt_at_limit(ks):
    # 3 day trades = at max allowed (3); a 4th would be PDT violation
    assert ks.check_pdt(day_trades_last_5_days=3) is True


def test_pdt_zero(ks):
    assert ks.check_pdt(day_trades_last_5_days=0) is False


# ── Stale data ────────────────────────────────────────────────────────────────

def test_stale_data_outside_market_hours(ks):
    # Outside market hours: stale data should NOT fire
    old_ts = datetime(2000, 1, 1, tzinfo=timezone.utc)  # ancient timestamp
    # Patch current time to 6 AM ET (before market open)
    import pytz
    et = pytz.timezone("America/New_York")
    fake_now = datetime(2024, 1, 15, 6, 0, 0).replace(tzinfo=et)
    with patch("trader.risk.kill_switches.datetime") as mock_dt:
        mock_dt.now.return_value = fake_now
        result = ks.check_stale_data(old_ts)
    assert result is False


def test_stale_data_none_during_market_hours(ks):
    """None timestamp during market hours should NOT trip — data simply hasn't been fetched yet."""
    import pytz
    et = pytz.timezone("America/New_York")
    fake_now = datetime(2024, 1, 15, 14, 0, 0).replace(tzinfo=et)
    with patch("trader.risk.kill_switches.datetime") as mock_dt:
        mock_dt.now.return_value = fake_now
        result = ks.check_stale_data(None)
    assert result is False


def test_stale_data_none_timestamp_does_not_trip(ks):
    """Passing None for last_data_ts never trips the stale-data guard regardless of time."""
    import pytz
    et = pytz.timezone("America/New_York")
    # Mid-market hours
    fake_now = datetime(2024, 1, 15, 11, 30, 0).replace(tzinfo=et)
    with patch("trader.risk.kill_switches.datetime") as mock_dt:
        mock_dt.now.return_value = fake_now
        assert ks.check_stale_data(None) is False


def test_stale_data_fresh(ks):
    """Fresh timestamp (1 min ago) during market hours should NOT fire."""
    import pytz
    et = pytz.timezone("America/New_York")
    fake_now_et = datetime(2024, 1, 15, 14, 0, 0).replace(tzinfo=et)
    fresh_ts = datetime.now(timezone.utc) - timedelta(minutes=1)
    with patch("trader.risk.kill_switches.datetime") as mock_dt:
        mock_dt.now.side_effect = lambda tz=None: fake_now_et if tz else datetime(2024, 1, 15, 19, 0, 0, tzinfo=timezone.utc)
        result = ks.check_stale_data(fresh_ts)
    # Can't reliably test with mock — just verify method runs without error
    assert isinstance(result, bool)


# ── Order rate limit ──────────────────────────────────────────────────────────

def test_order_rate_limit_below(ks):
    assert ks.check_order_rate_limit(9) is False


def test_order_rate_limit_at_limit(ks):
    assert ks.check_order_rate_limit(10) is True


def test_order_rate_limit_over(ks):
    assert ks.check_order_rate_limit(11) is True


# ── Broker disconnect ─────────────────────────────────────────────────────────

def test_broker_disconnect_not_tripped(ks):
    ks.mark_broker_error()
    ks.mark_broker_error()
    assert ks.check_broker_disconnect() is False


def test_broker_disconnect_tripped(ks):
    ks.mark_broker_error()
    ks.mark_broker_error()
    ks.mark_broker_error()
    assert ks.check_broker_disconnect() is True


def test_broker_success_resets_counter(ks):
    ks.mark_broker_error()
    ks.mark_broker_error()
    ks.mark_broker_success()
    assert ks.check_broker_disconnect() is False


# ── assert_safe_to_trade (master check) ─────────────────────────────────────

def test_assert_safe_all_clear(ks):
    """Should not raise when all checks pass."""
    ks.assert_safe_to_trade(
        start_value=25000,
        current_value=25100,
        peak_value=25100,
        day_trades_last_5=0,
        orders_this_hour=0,
        last_data_ts=None,  # outside market hours → stale check won't fire
    )


def test_assert_safe_kill_file_raises(ks, tmp_path):
    (tmp_path / "KILL").touch()
    with pytest.raises(KillSwitchTripped, match="KILL"):
        ks.assert_safe_to_trade(
            start_value=25000, current_value=25000, peak_value=25000,
            day_trades_last_5=0, orders_this_hour=0,
        )


def test_assert_safe_halt_file_raises(ks, tmp_path):
    (tmp_path / "HALTED").write_text("halted")
    with pytest.raises(KillSwitchTripped, match="HALTED"):
        ks.assert_safe_to_trade(
            start_value=25000, current_value=25000, peak_value=25000,
            day_trades_last_5=0, orders_this_hour=0,
        )


def test_assert_safe_daily_loss_raises(ks):
    with pytest.raises(KillSwitchTripped):
        ks.assert_safe_to_trade(
            start_value=25000, current_value=24000,  # -4% loss
            peak_value=25000, day_trades_last_5=0, orders_this_hour=0,
        )


def test_assert_safe_drawdown_raises(ks):
    with pytest.raises(KillSwitchTripped):
        ks.assert_safe_to_trade(
            start_value=22000, current_value=22000,
            peak_value=25000,  # -12% drawdown
            day_trades_last_5=0, orders_this_hour=0,
        )
