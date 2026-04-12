"""Tests for OrderManager idempotency and rate limiting.

All tests use mock brokers — no Alpaca API calls.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from trader.execution.broker import Order, Position
from trader.execution.order_manager import OrderManager, _MAX_ORDERS_PER_HOUR


def _make_order(order_id="ord1", symbol="AAPL", side="buy", status="new") -> Order:
    return Order(
        order_id=order_id, symbol=symbol, qty=10.0, filled_qty=0.0,
        side=side, order_type="market", status=status,
        submitted_at=None, filled_at=None, filled_avg_price=None,
        time_in_force="day",
    )


def _make_position(symbol="AAPL", qty=5.0) -> Position:
    return Position(
        symbol=symbol, qty=qty, avg_entry_price=150.0,
        current_price=160.0, market_value=800.0,
        unrealized_pl=50.0, unrealized_plpc=0.067,
        side="long",
    )


def _make_broker(open_orders=None, positions=None) -> MagicMock:
    broker = MagicMock()
    broker.get_open_orders.return_value = open_orders or []
    broker.get_positions.return_value = positions or []
    broker.submit_order.return_value = _make_order()
    return broker


# ── Entry order idempotency ───────────────────────────────────────────────────

def test_entry_skips_if_open_buy_order_exists():
    broker = _make_broker(open_orders=[_make_order(side="buy")])
    om = OrderManager(broker)
    result = om.place_entry_order("AAPL", notional=5000)
    assert result is None
    broker.submit_order.assert_not_called()


def test_entry_skips_if_position_held():
    broker = _make_broker(positions=[_make_position("AAPL")])
    om = OrderManager(broker)
    result = om.place_entry_order("AAPL", notional=5000)
    assert result is None
    broker.submit_order.assert_not_called()


def test_entry_proceeds_when_no_open_order_no_position():
    broker = _make_broker()
    om = OrderManager(broker)
    result = om.place_entry_order("AAPL", notional=5000)
    assert result is not None
    broker.submit_order.assert_called_once()


def test_entry_dry_run_does_not_call_submit():
    broker = _make_broker()
    om = OrderManager(broker)
    result = om.place_entry_order("AAPL", notional=5000, dry_run=True)
    assert result is None
    broker.submit_order.assert_not_called()


def test_entry_open_sell_order_does_not_block_buy():
    """An existing open sell order for a ticker should NOT block a new buy."""
    broker = _make_broker(open_orders=[_make_order(side="sell")])
    om = OrderManager(broker)
    # No position held — should proceed to submit
    result = om.place_entry_order("AAPL", notional=5000)
    assert result is not None
    broker.submit_order.assert_called_once()


def test_entry_uses_notional_not_qty():
    """Entry orders should use notional (dollar amount) for fractional shares."""
    broker = _make_broker()
    om = OrderManager(broker)
    om.place_entry_order("AAPL", notional=5000)
    call_kwargs = broker.submit_order.call_args
    assert call_kwargs.kwargs.get("notional") == 5000 or (
        len(call_kwargs.args) > 3 and call_kwargs.args[3] == 5000
    )


# ── Exit order idempotency ────────────────────────────────────────────────────

def test_exit_skips_if_open_sell_order_exists():
    broker = _make_broker(open_orders=[_make_order(side="sell")])
    om = OrderManager(broker)
    result = om.place_exit_order("AAPL", qty=5.0, reason="trailing_sma")
    assert result is None
    broker.submit_order.assert_not_called()


def test_exit_proceeds_when_no_open_sell():
    broker = _make_broker()
    om = OrderManager(broker)
    result = om.place_exit_order("AAPL", qty=5.0)
    assert result is not None
    broker.submit_order.assert_called_once()


def test_exit_dry_run():
    broker = _make_broker()
    om = OrderManager(broker)
    result = om.place_exit_order("AAPL", qty=5.0, dry_run=True)
    assert result is None
    broker.submit_order.assert_not_called()


def test_exit_open_buy_order_does_not_block_sell():
    """An existing buy order should NOT block a sell."""
    broker = _make_broker(open_orders=[_make_order(side="buy")])
    om = OrderManager(broker)
    result = om.place_exit_order("AAPL", qty=5.0)
    assert result is not None
    broker.submit_order.assert_called_once()


# ── Rate limiting ─────────────────────────────────────────────────────────────

def test_rate_limit_blocks_at_limit():
    broker = _make_broker()
    om = OrderManager(broker)

    # Fill up to the limit
    for i in range(_MAX_ORDERS_PER_HOUR):
        om._record_order_time()

    # Now entry should be blocked
    result = om.place_entry_order("AAPL", notional=5000)
    assert result is None
    broker.submit_order.assert_not_called()


def test_rate_limit_resets_after_one_hour():
    broker = _make_broker()
    om = OrderManager(broker)

    # Add order times older than 1 hour
    old_time = datetime.now(timezone.utc) - timedelta(hours=1, minutes=1)
    from collections import deque
    om._order_times = deque([old_time] * _MAX_ORDERS_PER_HOUR)

    # Should be allowed now (old orders pruned)
    result = om.place_entry_order("AAPL", notional=5000)
    assert result is not None
    broker.submit_order.assert_called_once()


def test_orders_this_hour_count():
    broker = _make_broker()
    om = OrderManager(broker)
    assert om.orders_this_hour() == 0
    om._record_order_time()
    om._record_order_time()
    assert om.orders_this_hour() == 2


# ── Cancel all open orders ────────────────────────────────────────────────────

def test_cancel_all_cancels_open_orders():
    broker = _make_broker(open_orders=[
        _make_order("ord1", "AAPL"),
        _make_order("ord2", "MSFT"),
    ])
    broker.cancel_order.return_value = True
    om = OrderManager(broker)
    count = om.cancel_all_open_orders()
    assert count == 2
    assert broker.cancel_order.call_count == 2


def test_cancel_all_dry_run_does_not_cancel():
    broker = _make_broker(open_orders=[_make_order("ord1", "AAPL")])
    om = OrderManager(broker)
    count = om.cancel_all_open_orders(dry_run=True)
    broker.cancel_order.assert_not_called()


# ── Broker error handling ─────────────────────────────────────────────────────

def test_entry_proceeds_on_open_orders_fetch_error():
    """If get_open_orders raises, proceed cautiously (don't silently skip)."""
    broker = MagicMock()
    broker.get_open_orders.side_effect = Exception("connection error")
    broker.get_positions.return_value = []
    broker.submit_order.return_value = _make_order()
    om = OrderManager(broker)
    # Should still attempt submission despite error checking open orders
    result = om.place_entry_order("AAPL", notional=5000)
    assert result is not None
