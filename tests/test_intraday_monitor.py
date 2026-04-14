"""Tests for the intraday risk monitor.

Focuses on the pure helper functions (no API calls):
  - check_hard_stops: fires at threshold, ignores noise above it
  - check_regime_gate: fires when SPY < SMA, holds when SPY >= SMA
  - SPY SMA cache: round-trips correctly, misses on wrong date
  - run_intraday_monitor: skips entries, fires stops, handles regime exit

Integration tests use a full mock broker — no real Alpaca calls.
"""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trader.execution.broker import AccountInfo, Position
from trader.monitor import (
    INTRADAY_HARD_STOP_PCT,
    check_hard_stops,
    check_regime_gate,
    compute_spy_sma200,
    load_spy_sma_cache,
    save_spy_sma_cache,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _pos(symbol: str, entry: float, current: float = None) -> Position:
    current = current if current is not None else entry
    pl = current - entry
    plpc = (current - entry) / entry
    return Position(
        symbol=symbol,
        qty=10.0,
        avg_entry_price=entry,
        current_price=current,
        market_value=current * 10.0,
        unrealized_pl=pl * 10.0,
        unrealized_plpc=plpc,
        side="long",
    )


# ── check_hard_stops ──────────────────────────────────────────────────────────

class TestCheckHardStops:

    def test_no_positions_returns_empty(self):
        assert check_hard_stops([], {}) == []

    def test_no_current_price_skips_position(self):
        pos = _pos("AAPL", entry=100.0)
        result = check_hard_stops([pos], current_prices={})
        assert result == []

    def test_exactly_at_threshold_fires(self):
        """PL exactly at -5% should fire."""
        entry = 100.0
        current = entry * (1 - INTRADAY_HARD_STOP_PCT)  # exactly -5%
        pos = _pos("AAPL", entry=entry)
        result = check_hard_stops([pos], {"AAPL": current})
        assert len(result) == 1
        assert result[0][0].symbol == "AAPL"
        assert result[0][1] == "intraday_hard_stop"

    def test_below_threshold_fires(self):
        """PL at -7% should fire (below -5% threshold)."""
        pos = _pos("AAPL", entry=100.0)
        result = check_hard_stops([pos], {"AAPL": 93.0})  # -7%
        assert len(result) == 1

    def test_above_threshold_does_not_fire(self):
        """PL at -3% should NOT fire (above -5% threshold)."""
        pos = _pos("AAPL", entry=100.0)
        result = check_hard_stops([pos], {"AAPL": 97.0})  # -3%
        assert result == []

    def test_just_below_threshold_fires(self):
        """PL at -5.01% should fire."""
        pos = _pos("AAPL", entry=100.0)
        result = check_hard_stops([pos], {"AAPL": 94.99})  # -5.01%
        assert len(result) == 1

    def test_positive_return_does_not_fire(self):
        """Profitable position should never fire the stop."""
        pos = _pos("AAPL", entry=100.0)
        result = check_hard_stops([pos], {"AAPL": 115.0})  # +15%
        assert result == []

    def test_five_simultaneous_positions_only_breached_fire(self):
        """Only positions that breached the threshold are returned."""
        positions = [
            _pos("AAPL", entry=100.0),   # current set below
            _pos("MSFT", entry=200.0),
            _pos("NVDA", entry=500.0),
            _pos("META", entry=300.0),
            _pos("AMZN", entry=150.0),
        ]
        prices = {
            "AAPL": 94.0,   # -6% → FIRES
            "MSFT": 198.0,  # -1% → safe
            "NVDA": 470.0,  # -6% → FIRES
            "META": 285.0,  # -5% → FIRES (exactly at threshold)
            "AMZN": 148.0,  # -1.3% → safe
        }
        result = check_hard_stops(positions, prices)
        fired = {r[0].symbol for r in result}
        assert fired == {"AAPL", "NVDA", "META"}

    def test_custom_threshold(self):
        """Custom threshold respected."""
        pos = _pos("AAPL", entry=100.0)
        # At -3%, fires with 0.02 threshold but not with 0.05 (default)
        assert check_hard_stops([pos], {"AAPL": 97.0}, hard_stop_pct=0.02) != []
        assert check_hard_stops([pos], {"AAPL": 97.0}, hard_stop_pct=0.05) == []


# ── check_regime_gate ─────────────────────────────────────────────────────────

class TestCheckRegimeGate:

    def test_spy_above_sma_no_regime_exit(self):
        assert check_regime_gate(spy_price=520.0, spy_sma200=500.0) is False

    def test_spy_equal_to_sma_no_regime_exit(self):
        assert check_regime_gate(spy_price=500.0, spy_sma200=500.0) is False

    def test_spy_below_sma_triggers_regime_exit(self):
        assert check_regime_gate(spy_price=490.0, spy_sma200=500.0) is True

    def test_large_gap_below_sma(self):
        assert check_regime_gate(spy_price=400.0, spy_sma200=500.0) is True

    def test_just_below_sma(self):
        assert check_regime_gate(spy_price=499.99, spy_sma200=500.0) is True


# ── SPY SMA cache ─────────────────────────────────────────────────────────────

class TestSpySmaCache:

    def test_cache_miss_on_empty_dir(self, tmp_path):
        result = load_spy_sma_cache(tmp_path, "2026-04-13")
        assert result is None

    def test_cache_round_trip(self, tmp_path):
        save_spy_sma_cache(tmp_path, "2026-04-13", 512.34)
        result = load_spy_sma_cache(tmp_path, "2026-04-13")
        assert result == pytest.approx(512.34)

    def test_cache_miss_on_wrong_date(self, tmp_path):
        save_spy_sma_cache(tmp_path, "2026-04-12", 512.34)  # yesterday
        result = load_spy_sma_cache(tmp_path, "2026-04-13")  # today
        assert result is None

    def test_cache_overwrites_stale(self, tmp_path):
        save_spy_sma_cache(tmp_path, "2026-04-12", 510.0)
        save_spy_sma_cache(tmp_path, "2026-04-13", 515.0)
        assert load_spy_sma_cache(tmp_path, "2026-04-13") == pytest.approx(515.0)
        assert load_spy_sma_cache(tmp_path, "2026-04-12") is None


# ── run_intraday_monitor integration (fully mocked) ──────────────────────────

def _make_mock_broker(positions=None, quotes=None, spy_bars=None):
    """Build a mock AlpacaBroker."""
    from trader.execution.broker import AccountInfo
    import pandas as pd

    broker = MagicMock()
    broker.get_account.return_value = AccountInfo(
        account_id="TEST123",
        status="ACTIVE",
        cash=20_000.0,
        portfolio_value=25_000.0,
        buying_power=40_000.0,
        equity=25_000.0,
        last_equity=24_800.0,
    )
    broker.get_positions.return_value = positions or []
    broker.get_latest_quotes.return_value = quotes or {}
    broker.get_open_orders.return_value = []

    # Fake SPY bars for SMA computation (300 bars around 500)
    idx = pd.date_range("2024-06-01", periods=300, freq="B")
    spy_df = pd.DataFrame({"Close": [500.0] * 300, "Open": [500.0] * 300}, index=idx)
    broker.get_bars.return_value = {"SPY": spy_df}

    return broker


class TestRunIntradayMonitor:

    def _run_monitor(self, broker, dry_run=True, tmp_path=None):
        """Run the monitor with a mocked broker and temp DB/cache."""
        import sqlite3
        import os
        from trader.monitor import run_intraday_monitor

        db_path = tmp_path / "trader.db" if tmp_path else Path(tempfile.mktemp(suffix=".db"))
        cache_dir = tmp_path if tmp_path else Path(tempfile.mkdtemp())

        mock_db = MagicMock()
        mock_db.get_start_of_day_value.return_value = 25_000.0
        mock_db.get_peak_value.return_value = 25_000.0
        mock_db.count_day_trades_last_5_days.return_value = 0

        mock_om = MagicMock()
        mock_om.orders_this_hour.return_value = 0

        with (
            patch("trader.monitor.AlpacaBroker", return_value=broker),
            patch("trader.monitor.load_credentials", return_value={
                "ALPACA_KEY_ID": "test", "ALPACA_SECRET_KEY": "test",
            }),
            patch("trader.monitor._PROJECT_ROOT", cache_dir),
            patch("trader.monitor.TradingDB", return_value=mock_db),
            patch("trader.monitor.OrderManager", return_value=mock_om),
        ):
            run_intraday_monitor(dry_run=dry_run, env_path=Path("/fake/alpaca.env"))

        return mock_db, mock_om

    def test_monitor_skips_entries_no_positions(self, tmp_path):
        """With no open positions, monitor exits without calling place_entry_order."""
        broker = _make_mock_broker(positions=[])
        mock_db, mock_om = self._run_monitor(broker, tmp_path=tmp_path)
        mock_om.place_entry_order.assert_not_called()
        mock_om.place_exit_order.assert_not_called()

    def test_monitor_skips_entries_with_safe_position(self, tmp_path):
        """With a safe position (no stop breach), no exits and no entries."""
        pos = _pos("AAPL", entry=100.0)
        broker = _make_mock_broker(
            positions=[pos],
            quotes={"AAPL": 102.0, "SPY": 510.0},  # AAPL up, SPY above SMA
        )
        # Cache SPY SMA so bars aren't refetched
        save_spy_sma_cache(tmp_path, date.today().isoformat(), 500.0)

        mock_db, mock_om = self._run_monitor(broker, tmp_path=tmp_path)
        mock_om.place_entry_order.assert_not_called()
        mock_om.place_exit_order.assert_not_called()

    def test_hard_stop_fires_exit_order(self, tmp_path):
        """Position -6% below entry should trigger an exit order."""
        pos = _pos("NVDA", entry=500.0)
        broker = _make_mock_broker(
            positions=[pos],
            quotes={"NVDA": 470.0, "SPY": 510.0},  # NVDA -6%, SPY safe
        )
        save_spy_sma_cache(tmp_path, date.today().isoformat(), 500.0)

        mock_db, mock_om = self._run_monitor(broker, dry_run=True, tmp_path=tmp_path)

        mock_om.place_exit_order.assert_called_once()
        call_kwargs = mock_om.place_exit_order.call_args
        assert call_kwargs.kwargs["symbol"] == "NVDA"
        assert call_kwargs.kwargs["reason"] == "intraday_hard_stop"
        assert call_kwargs.kwargs["dry_run"] is True

    def test_noise_above_threshold_does_not_fire(self, tmp_path):
        """Position only -2% should not trigger a stop."""
        pos = _pos("AAPL", entry=100.0)
        broker = _make_mock_broker(
            positions=[pos],
            quotes={"AAPL": 98.0, "SPY": 510.0},   # -2%, well above 5% stop
        )
        save_spy_sma_cache(tmp_path, date.today().isoformat(), 500.0)

        mock_db, mock_om = self._run_monitor(broker, tmp_path=tmp_path)
        mock_om.place_exit_order.assert_not_called()

    def test_regime_exit_fires_for_all_positions(self, tmp_path):
        """When SPY breaks below 200d SMA, all positions should exit."""
        positions = [_pos("AAPL", entry=100.0), _pos("MSFT", entry=200.0)]
        broker = _make_mock_broker(
            positions=positions,
            quotes={"AAPL": 99.0, "MSFT": 199.0, "SPY": 480.0},  # SPY < 500 SMA
        )
        save_spy_sma_cache(tmp_path, date.today().isoformat(), 500.0)

        mock_db, mock_om = self._run_monitor(broker, dry_run=True, tmp_path=tmp_path)

        assert mock_om.place_exit_order.call_count == 2
        reasons = {
            c.kwargs["reason"] for c in mock_om.place_exit_order.call_args_list
        }
        assert reasons == {"intraday_regime_stop"}

    def test_regime_exit_reason_logged_to_db(self, tmp_path):
        """DB log_decision should be called with intraday_regime_stop reason."""
        pos = _pos("AAPL", entry=100.0)
        broker = _make_mock_broker(
            positions=[pos],
            quotes={"AAPL": 99.0, "SPY": 480.0},
        )
        save_spy_sma_cache(tmp_path, date.today().isoformat(), 500.0)

        mock_db, mock_om = self._run_monitor(broker, dry_run=True, tmp_path=tmp_path)

        mock_db.log_decision.assert_called_once()
        logged_reason = mock_db.log_decision.call_args.kwargs["reason"]
        assert logged_reason == "intraday_regime_stop"
