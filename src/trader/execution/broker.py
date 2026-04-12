"""BrokerProtocol — abstract interface for all broker integrations.

Any live broker (Alpaca, IBKR, etc.) implements this protocol so the rest
of the system stays broker-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd


@dataclass
class AccountInfo:
    account_id: str
    status: str
    cash: float
    portfolio_value: float
    buying_power: float
    equity: float
    last_equity: float  # previous day close equity


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float  # fraction, e.g. 0.05 = +5%
    side: str  # "long" or "short"


@dataclass
class Order:
    order_id: str
    symbol: str
    qty: float
    filled_qty: float
    side: str          # "buy" or "sell"
    order_type: str    # "market", "limit", etc.
    status: str        # "new", "filled", "canceled", etc.
    submitted_at: Optional[datetime]
    filled_at: Optional[datetime]
    filled_avg_price: Optional[float]
    time_in_force: str  # "day", "gtc", etc.


class BrokerProtocol(ABC):
    """Abstract broker interface. All live broker implementations must subclass this."""

    @abstractmethod
    def get_account(self) -> AccountInfo:
        """Return current account status and balances."""

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Return all currently held positions."""

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,               # "buy" or "sell"
        order_type: str = "market",
        time_in_force: str = "day",
        notional: Optional[float] = None,  # dollar amount instead of shares
    ) -> Order:
        """Submit an order. Returns the submitted order object.

        For fractional shares, pass notional (dollar amount) instead of qty.
        Both qty and notional cannot be set simultaneously.
        """

    @abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        """Return all open (pending) orders. Optionally filter by symbol."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID. Returns True if successful."""

    @abstractmethod
    def get_bars(
        self,
        symbols: list[str],
        start: str,      # "YYYY-MM-DD"
        end: str,        # "YYYY-MM-DD"
        timeframe: str = "1Day",
    ) -> dict[str, pd.DataFrame]:
        """Return OHLCV daily bars for each symbol.

        Returns dict: symbol -> DataFrame with columns [Open, High, Low, Close, Volume]
        Index is DatetimeIndex (date-only, tz-naive).
        """

    @abstractmethod
    def get_latest_quote(self, symbol: str) -> Optional[float]:
        """Return the latest trade price for a symbol."""
