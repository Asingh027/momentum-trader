"""OrderManager — idempotent order placement with rate limiting and logging.

Responsibilities:
- Idempotency: check open orders and existing positions before submitting
- Rate limiting: max 10 orders per hour (hard-coded safety limit)
- Fractional shares: use notional dollar amounts for consistent sizing
- Lot tracking: log cost basis per position for P&L calculation
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Optional

from trader.execution.broker import BrokerProtocol, Order

logger = logging.getLogger(__name__)

_MAX_ORDERS_PER_HOUR = 10


class OrderManager:
    """Wraps BrokerProtocol with idempotency, rate limiting, and lot tracking."""

    def __init__(self, broker: BrokerProtocol):
        self._broker = broker
        # Track order submission timestamps (rolling window for rate limit)
        self._order_times: deque[datetime] = deque()
        # In-memory cost basis log: symbol -> (qty, avg_entry_price)
        self._lots: dict[str, tuple[float, float]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def place_entry_order(
        self,
        symbol: str,
        notional: float,        # dollar amount to invest (fractional-share safe)
        dry_run: bool = False,
    ) -> Optional[Order]:
        """Submit a market buy order. Returns None if skipped (idempotency/rate limit).

        Uses notional (dollar amount) instead of shares to support fractional shares.
        This ensures consistent 20%-of-portfolio sizing regardless of share price.
        """
        # Rate limit check
        if not self._check_rate_limit():
            logger.warning("[%s] Order rate limit reached (%d/hr) — skipping entry", symbol, _MAX_ORDERS_PER_HOUR)
            return None

        # Idempotency: no existing open buy order
        try:
            open_orders = self._broker.get_open_orders(symbol=symbol)
            buy_orders = [o for o in open_orders if "buy" in o.side.lower()]
            if buy_orders:
                logger.info("[%s] Idempotency: open buy order %s already exists, skipping", symbol, buy_orders[0].order_id)
                return None
        except Exception as exc:
            logger.warning("[%s] Could not check open orders: %s — proceeding cautiously", symbol, exc)

        # Idempotency: no existing position
        try:
            positions = self._broker.get_positions()
            held = {p.symbol for p in positions}
            if symbol in held:
                logger.info("[%s] Idempotency: position already held, skipping", symbol)
                return None
        except Exception as exc:
            logger.warning("[%s] Could not check positions: %s — proceeding cautiously", symbol, exc)

        if dry_run:
            logger.info("[DRY RUN] Would BUY %s — notional=$%.2f", symbol, notional)
            return None

        logger.info("Submitting BUY %s — notional=$%.2f", symbol, notional)
        try:
            order = self._broker.submit_order(
                symbol=symbol,
                qty=0,          # ignored when notional is set
                side="buy",
                notional=notional,
            )
            self._record_order_time()
            logger.info("Order submitted: %s %s qty=%.4f id=%s", order.side, order.symbol, order.qty, order.order_id)
            return order
        except Exception as exc:
            logger.error("Failed to submit BUY %s: %s", symbol, exc)
            raise

    def place_exit_order(
        self,
        symbol: str,
        qty: float,
        reason: str = "",
        dry_run: bool = False,
    ) -> Optional[Order]:
        """Submit a market sell order for the full position quantity.

        qty should be the full held quantity (sell entire position per v1 exit rules).
        """
        # Rate limit check
        if not self._check_rate_limit():
            logger.warning("[%s] Order rate limit reached (%d/hr) — skipping exit", symbol, _MAX_ORDERS_PER_HOUR)
            return None

        # Idempotency: no existing open sell order
        try:
            open_orders = self._broker.get_open_orders(symbol=symbol)
            sell_orders = [o for o in open_orders if "sell" in o.side.lower()]
            if sell_orders:
                logger.info("[%s] Idempotency: open sell order %s already exists, skipping", symbol, sell_orders[0].order_id)
                return None
        except Exception as exc:
            logger.warning("[%s] Could not check open orders for exit: %s", symbol, exc)

        if dry_run:
            logger.info("[DRY RUN] Would SELL %s qty=%.4f reason=%s", symbol, qty, reason)
            return None

        logger.info("Submitting SELL %s qty=%.4f reason=%s", symbol, qty, reason)
        try:
            order = self._broker.submit_order(
                symbol=symbol,
                qty=qty,
                side="sell",
            )
            self._record_order_time()
            logger.info("Exit order submitted: %s %s qty=%.4f id=%s", order.side, order.symbol, order.qty, order.order_id)
            return order
        except Exception as exc:
            logger.error("Failed to submit SELL %s: %s", symbol, exc)
            raise

    def cancel_all_open_orders(self, dry_run: bool = False) -> int:
        """Cancel all open orders. Returns count of cancelled orders."""
        try:
            orders = self._broker.get_open_orders()
        except Exception as exc:
            logger.error("Could not fetch open orders for cancellation: %s", exc)
            return 0

        cancelled = 0
        for order in orders:
            if dry_run:
                logger.info("[DRY RUN] Would cancel order %s (%s %s)", order.order_id, order.side, order.symbol)
            else:
                if self._broker.cancel_order(order.order_id):
                    cancelled += 1
        return cancelled

    def orders_this_hour(self) -> int:
        """Return number of orders submitted in the last 60 minutes."""
        self._prune_order_times()
        return len(self._order_times)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _check_rate_limit(self) -> bool:
        self._prune_order_times()
        return len(self._order_times) < _MAX_ORDERS_PER_HOUR

    def _record_order_time(self) -> None:
        self._order_times.append(datetime.now(timezone.utc))

    def _prune_order_times(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        while self._order_times and self._order_times[0] < cutoff:
            self._order_times.popleft()
