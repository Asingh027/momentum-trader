"""AlpacaBroker — alpaca-py SDK implementation of BrokerProtocol.

Uses Alpaca paper trading endpoint. All API calls are wrapped with retry
logic (3 retries, exponential backoff, 10s timeout per call).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

from trader.execution.broker import AccountInfo, BrokerProtocol, Order, Position

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 1.5  # seconds; retry waits 1.5s, 3s, 6s


def _retry(func):
    """Decorator: retry up to _MAX_RETRIES times with exponential backoff."""
    def wrapper(*args, **kwargs):
        last_exc = None
        for attempt in range(_MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                wait = _BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "Alpaca API call %s failed (attempt %d/%d): %s — retrying in %.1fs",
                    func.__name__, attempt + 1, _MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)
        logger.error("Alpaca API call %s failed after %d retries: %s", func.__name__, _MAX_RETRIES, last_exc)
        raise last_exc
    return wrapper


def _timeframe_from_str(tf: str) -> TimeFrame:
    mapping = {
        "1Day": TimeFrame.Day,
        "1Hour": TimeFrame.Hour,
        "1Min": TimeFrame.Minute,
    }
    return mapping.get(tf, TimeFrame.Day)


class AlpacaBroker(BrokerProtocol):
    """Paper trading broker implementation using alpaca-py SDK."""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self._trading = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )
        self._data = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key,
        )
        logger.info("AlpacaBroker initialised (paper=%s)", paper)

    # ── Account ──────────────────────────────────────────────────────────────

    @_retry
    def get_account(self) -> AccountInfo:
        acct = self._trading.get_account()
        return AccountInfo(
            account_id=str(acct.account_number),
            status=str(acct.status),
            cash=float(acct.cash),
            portfolio_value=float(acct.portfolio_value),
            buying_power=float(acct.buying_power),
            equity=float(acct.equity),
            last_equity=float(acct.last_equity),
        )

    # ── Positions ─────────────────────────────────────────────────────────────

    @_retry
    def get_positions(self) -> list[Position]:
        positions = self._trading.get_all_positions()
        result = []
        for p in positions:
            result.append(Position(
                symbol=str(p.symbol),
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price) if p.current_price else 0.0,
                market_value=float(p.market_value) if p.market_value else 0.0,
                unrealized_pl=float(p.unrealized_pl) if p.unrealized_pl else 0.0,
                unrealized_plpc=float(p.unrealized_plpc) if p.unrealized_plpc else 0.0,
                side=str(p.side),
            ))
        return result

    # ── Orders ────────────────────────────────────────────────────────────────

    @_retry
    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        notional: Optional[float] = None,
    ) -> Order:
        alpaca_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        alpaca_tif = TimeInForce.DAY if time_in_force == "day" else TimeInForce.GTC

        if notional is not None:
            req = MarketOrderRequest(
                symbol=symbol,
                notional=round(notional, 2),
                side=alpaca_side,
                time_in_force=alpaca_tif,
            )
        else:
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=alpaca_side,
                time_in_force=alpaca_tif,
            )

        order = self._trading.submit_order(req)
        return self._to_order(order)

    @_retry
    def get_order_by_id(self, order_id: str) -> Order:
        order = self._trading.get_order_by_id(order_id)
        return self._to_order(order)

    @_retry
    def get_open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol] if symbol else None)
        orders = self._trading.get_orders(req)
        return [self._to_order(o) for o in orders]

    @_retry
    def cancel_order(self, order_id: str) -> bool:
        try:
            self._trading.cancel_order_by_id(order_id)
            return True
        except Exception as exc:
            logger.warning("Failed to cancel order %s: %s", order_id, exc)
            return False

    # ── Market Data ───────────────────────────────────────────────────────────

    @_retry
    def get_bars(
        self,
        symbols: list[str],
        start: str,
        end: str,
        timeframe: str = "1Day",
    ) -> dict[str, pd.DataFrame]:
        """Fetch historical daily bars from Alpaca IEX feed (free tier)."""
        tf = _timeframe_from_str(timeframe)

        # Alpaca requires timezone-aware timestamps
        start_dt = pd.Timestamp(start, tz="America/New_York")
        end_dt = pd.Timestamp(end, tz="America/New_York")

        # Batch in chunks of 100 to avoid request size limits
        result: dict[str, pd.DataFrame] = {}
        chunk_size = 100
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i : i + chunk_size]
            req = StockBarsRequest(
                symbol_or_symbols=chunk,
                start=start_dt,
                end=end_dt,
                timeframe=tf,
                feed=DataFeed.IEX,
            )
            bars_response = self._data.get_stock_bars(req)
            bars_df = bars_response.df

            if bars_df.empty:
                continue

            # alpaca-py returns MultiIndex (symbol, timestamp) — reshape to dict
            if isinstance(bars_df.index, pd.MultiIndex):
                for sym in bars_df.index.get_level_values(0).unique():
                    sym_df = bars_df.xs(sym, level=0).copy()
                    sym_df.index = pd.DatetimeIndex(sym_df.index).tz_localize(None).normalize()
                    sym_df.columns = [c.capitalize() for c in sym_df.columns]
                    # Ensure standard columns
                    for col in ["Open", "High", "Low", "Close", "Volume"]:
                        if col not in sym_df.columns:
                            sym_df[col] = float("nan")
                    result[sym] = sym_df[["Open", "High", "Low", "Close", "Volume"]]
            else:
                # Single ticker case
                for sym in chunk:
                    if sym in bars_df.columns:
                        result[sym] = bars_df[[sym]].rename(columns={sym: "Close"})

        return result

    @_retry
    def get_latest_quote(self, symbol: str) -> Optional[float]:
        req = StockLatestTradeRequest(symbol_or_symbols=[symbol], feed=DataFeed.IEX)
        trades = self._data.get_stock_latest_trade(req)
        if symbol in trades:
            return float(trades[symbol].price)
        return None

    @_retry
    def get_latest_quotes(self, symbols: list[str]) -> dict[str, float]:
        """Batch latest-trade prices — one API call for all symbols."""
        if not symbols:
            return {}
        req = StockLatestTradeRequest(symbol_or_symbols=symbols, feed=DataFeed.IEX)
        trades = self._data.get_stock_latest_trade(req)
        return {sym: float(t.price) for sym, t in trades.items()}

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_order(o) -> Order:
        return Order(
            order_id=str(o.id),
            symbol=str(o.symbol),
            qty=float(o.qty) if o.qty else 0.0,
            filled_qty=float(o.filled_qty) if o.filled_qty else 0.0,
            side=str(o.side).lower().replace("orderside.", "").replace("ordersideenum.", ""),
            order_type=str(o.type).lower().replace("ordertype.", ""),
            status=str(o.status).lower().replace("orderstatus.", ""),
            submitted_at=o.submitted_at,
            filled_at=o.filled_at,
            filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price else None,
            time_in_force=str(o.time_in_force).lower().replace("timeinforce.", ""),
        )
