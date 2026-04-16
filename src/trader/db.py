"""SQLite decision log for live trading.

Tables:
  decisions      — every action the bot considers or takes
  daily_summary  — one row per trading day with full portfolio snapshot
  trades         — completed round-trip trade records

Default path: C:\\Users\\Avneet\\Documents\\Trading Helper\\trader.db
Override via TRADER_DB_PATH environment variable (DB_PATH accepted as fallback).
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path(r"C:\Users\Avneet\Documents\Trading Helper\trader.db")


def get_db_path() -> Path:
    env = os.environ.get("TRADER_DB_PATH", "") or os.environ.get("DB_PATH", "")
    return Path(env) if env.strip() else _DEFAULT_DB_PATH


class TradingDB:
    """SQLite wrapper for the trading decision log."""

    def __init__(self, db_path: Optional[Path] = None):
        self._path = db_path or get_db_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        logger.info("TradingDB at %s", self._path)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS decisions (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT NOT NULL,
                    ticker          TEXT NOT NULL,
                    action          TEXT NOT NULL,  -- entry / exit / skip / error
                    reason          TEXT,
                    price           REAL,
                    shares          REAL,
                    notional        REAL,
                    portfolio_value REAL,
                    order_id        TEXT,
                    dry_run         INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS daily_summary (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    date                TEXT NOT NULL UNIQUE,
                    portfolio_value     REAL,
                    cash                REAL,
                    equity              REAL,
                    open_positions      INTEGER,
                    realized_pnl        REAL,
                    unrealized_pnl      REAL,
                    start_of_day_value  REAL,
                    peak_value          REAL,
                    daily_pnl_pct       REAL,
                    drawdown_from_peak  REAL,
                    kill_switch_status  TEXT,
                    orders_placed       INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker          TEXT NOT NULL,
                    entry_date      TEXT,
                    exit_date       TEXT,
                    entry_price     REAL,
                    exit_price      REAL,
                    shares          REAL,
                    pnl             REAL,
                    pnl_pct         REAL,
                    exit_reason     TEXT,
                    holding_days    INTEGER
                );

                CREATE INDEX IF NOT EXISTS idx_decisions_ticker ON decisions(ticker);
                CREATE INDEX IF NOT EXISTS idx_decisions_ts     ON decisions(timestamp);
                CREATE INDEX IF NOT EXISTS idx_trades_ticker    ON trades(ticker);
            """)

    # ── Decisions ─────────────────────────────────────────────────────────────

    def log_decision(
        self,
        ticker: str,
        action: str,
        reason: str = "",
        price: Optional[float] = None,
        shares: Optional[float] = None,
        notional: Optional[float] = None,
        portfolio_value: Optional[float] = None,
        order_id: Optional[str] = None,
        dry_run: bool = False,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO decisions
                   (timestamp, ticker, action, reason, price, shares, notional, portfolio_value, order_id, dry_run)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.utcnow().isoformat(),
                    ticker, action, reason,
                    price, shares, notional,
                    portfolio_value, order_id,
                    1 if dry_run else 0,
                ),
            )

    # ── Daily Summary ─────────────────────────────────────────────────────────

    def upsert_daily_summary(
        self,
        date_str: str,          # "YYYY-MM-DD"
        portfolio_value: float,
        cash: float,
        equity: float,
        open_positions: int,
        realized_pnl: float,
        unrealized_pnl: float,
        start_of_day_value: float,
        peak_value: float,
        kill_switch_status: str = "OK",
        orders_placed: int = 0,
    ) -> None:
        daily_pnl_pct = (portfolio_value - start_of_day_value) / start_of_day_value if start_of_day_value > 0 else 0.0
        drawdown = (portfolio_value - peak_value) / peak_value if peak_value > 0 else 0.0

        with self._conn() as conn:
            conn.execute(
                """INSERT INTO daily_summary
                   (date, portfolio_value, cash, equity, open_positions,
                    realized_pnl, unrealized_pnl, start_of_day_value,
                    peak_value, daily_pnl_pct, drawdown_from_peak,
                    kill_switch_status, orders_placed)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(date) DO UPDATE SET
                     portfolio_value    = excluded.portfolio_value,
                     cash               = excluded.cash,
                     equity             = excluded.equity,
                     open_positions     = excluded.open_positions,
                     realized_pnl       = excluded.realized_pnl,
                     unrealized_pnl     = excluded.unrealized_pnl,
                     start_of_day_value = excluded.start_of_day_value,
                     peak_value         = excluded.peak_value,
                     daily_pnl_pct      = excluded.daily_pnl_pct,
                     drawdown_from_peak = excluded.drawdown_from_peak,
                     kill_switch_status = excluded.kill_switch_status,
                     orders_placed      = excluded.orders_placed
                """,
                (
                    date_str, portfolio_value, cash, equity,
                    open_positions, realized_pnl, unrealized_pnl,
                    start_of_day_value, peak_value,
                    daily_pnl_pct, drawdown,
                    kill_switch_status, orders_placed,
                ),
            )

    def get_start_of_day_value(self, date_str: str) -> Optional[float]:
        """Return the start-of-day portfolio value stored yesterday, or None."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT portfolio_value FROM daily_summary WHERE date < ? ORDER BY date DESC LIMIT 1",
                (date_str,),
            ).fetchone()
        return row["portfolio_value"] if row else None

    def get_peak_value(self) -> Optional[float]:
        """Return the all-time highest portfolio value recorded."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(portfolio_value) as pk FROM daily_summary"
            ).fetchone()
        return row["pk"] if row and row["pk"] is not None else None

    # ── Trades ────────────────────────────────────────────────────────────────

    def log_trade(
        self,
        ticker: str,
        entry_date: str,
        exit_date: str,
        entry_price: float,
        exit_price: float,
        shares: float,
        pnl: float,
        exit_reason: str,
        holding_days: int,
    ) -> None:
        pnl_pct = pnl / (entry_price * shares) if entry_price * shares > 0 else 0.0
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO trades
                   (ticker, entry_date, exit_date, entry_price, exit_price,
                    shares, pnl, pnl_pct, exit_reason, holding_days)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (ticker, entry_date, exit_date, entry_price, exit_price,
                 shares, pnl, pnl_pct, exit_reason, holding_days),
            )

    # ── PDT tracking ─────────────────────────────────────────────────────────

    def count_day_trades_last_5_days(self) -> int:
        """Count same-day round-trip trades in the last 5 business days.

        A day trade = entry and exit logged on the same date.
        Approximation: if holding_days == 0, it was a day trade.
        """
        with self._conn() as conn:
            row = conn.execute(
                """SELECT COUNT(*) as n FROM trades
                   WHERE holding_days = 0
                     AND exit_date >= date('now', '-7 days')"""
            ).fetchone()
        return row["n"] if row else 0

    # ── Recent decisions ──────────────────────────────────────────────────────

    def get_today_decisions(self, date_str: Optional[str] = None) -> list[dict]:
        if date_str is None:
            date_str = date.today().isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM decisions WHERE timestamp LIKE ? ORDER BY timestamp",
                (f"{date_str}%",),
            ).fetchall()
        return [dict(r) for r in rows]
