"""Kill switches — hard safety limits for the live execution engine.

ALL thresholds are hard-coded. These are safety rails, NOT tunable parameters.
Do not move these to TradingConfig or environment variables.

Kill switches:
  1. Daily loss halt    — daily P&L <= -3% of start-of-day portfolio value
  2. Drawdown halt      — portfolio <= -10% from all-time high watermark
  3. Manual kill switch — KILL file exists in project root
  4. Halt file          — HALTED file exists (written by drawdown halt; must be
                          manually deleted by Avneet to resume)
  5. PDT guard          — would be the 4th day trade in rolling 5 business days
  6. Stale data guard   — market data timestamp >10 min stale during market hours
  7. Order rate limit   — 10 orders already submitted in the last 60 minutes
  8. Broker disconnect  — 3 consecutive API failures (set externally via .mark_broker_error())
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Optional

import pytz

logger = logging.getLogger(__name__)

# ── Hard-coded thresholds (do NOT move to config) ──────────────────────────
_DAILY_LOSS_LIMIT = -0.03        # -3% of start-of-day portfolio value
_DRAWDOWN_LIMIT = -0.10          # -10% from all-time high watermark
_PDT_MAX_DAY_TRADES = 3          # block if rolling-5-day count would reach 4
_STALE_DATA_MINUTES = 10         # >10 min without fresh data during mkt hours
_MAX_ORDERS_PER_HOUR = 10        # mirrors OrderManager limit
_BROKER_ERROR_LIMIT = 3          # consecutive failures before halting
_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)
_EASTERN = pytz.timezone("America/New_York")

_HALT_FILENAME = "HALTED"
_KILL_FILENAME = "KILL"


class KillSwitchTripped(Exception):
    """Raised when a kill switch fires. Message describes which one."""


class KillSwitches:
    """Evaluates all kill switches and raises KillSwitchTripped on violation.

    Parameters
    ----------
    project_root : Path
        Project root directory. KILL and HALTED files are checked here.
    """

    def __init__(self, project_root: Path):
        self._root = project_root
        self._consecutive_broker_errors: int = 0

    # ── Individual checks (return True = tripped) ─────────────────────────

    def check_kill_file(self) -> bool:
        """True if KILL file exists in project root."""
        return (self._root / _KILL_FILENAME).exists()

    def check_halt_file(self) -> bool:
        """True if HALTED file exists. Must be manually deleted to resume."""
        return (self._root / _HALT_FILENAME).exists()

    def check_daily_loss(self, start_value: float, current_value: float) -> bool:
        """True if today's P&L has breached -3% of start-of-day value.

        Parameters
        ----------
        start_value : float
            Portfolio value at start of trading day.
        current_value : float
            Current portfolio value (equity + cash).
        """
        if start_value <= 0:
            return False
        daily_pct = (current_value - start_value) / start_value
        tripped = daily_pct <= _DAILY_LOSS_LIMIT
        if tripped:
            logger.warning(
                "KILL SWITCH: daily loss %.2f%% breached %.2f%% limit",
                daily_pct * 100, _DAILY_LOSS_LIMIT * 100,
            )
        return tripped

    def check_drawdown(self, peak_value: float, current_value: float) -> bool:
        """True if portfolio has dropped >10% from all-time high watermark.

        When this trips, writes a HALTED file. Bot will not trade until
        the file is manually deleted.
        """
        if peak_value <= 0:
            return False
        drawdown = (current_value - peak_value) / peak_value
        tripped = drawdown <= _DRAWDOWN_LIMIT
        if tripped:
            self._write_halt_file(
                f"Drawdown halt: portfolio {drawdown:.2%} from peak ${peak_value:,.2f}. "
                f"Current: ${current_value:,.2f}. Delete HALTED to resume."
            )
            logger.critical(
                "KILL SWITCH: drawdown %.2f%% from peak $%.2f — HALTED file written",
                drawdown * 100, peak_value,
            )
        return tripped

    def check_pdt(self, day_trades_last_5_days: int) -> bool:
        """True if submitting another order would be the 4th day trade in 5 days.

        day_trades_last_5_days: count of same-day round-trips in last 5 business days.
        Returns True (blocked) if count is already >= _PDT_MAX_DAY_TRADES.
        """
        tripped = day_trades_last_5_days >= _PDT_MAX_DAY_TRADES
        if tripped:
            logger.warning(
                "KILL SWITCH: PDT guard — %d day trades in rolling 5 days (max %d)",
                day_trades_last_5_days, _PDT_MAX_DAY_TRADES,
            )
        return tripped

    def check_stale_data(self, last_data_ts: Optional[datetime]) -> bool:
        """True if market data is stale (>10 min) during market hours.

        Only fires during market hours (9:30 AM – 4:00 PM ET).
        last_data_ts must be timezone-aware (UTC or ET).
        """
        now_et = datetime.now(_EASTERN)
        if not (_MARKET_OPEN <= now_et.time() <= _MARKET_CLOSE):
            return False  # Outside market hours — don't fire

        if last_data_ts is None:
            return False  # No timestamp yet means data hasn't been fetched — not the same as stale

        # Normalise to UTC for comparison
        if last_data_ts.tzinfo is None:
            last_data_ts = last_data_ts.replace(tzinfo=timezone.utc)
        now_utc = datetime.now(timezone.utc)
        age_minutes = (now_utc - last_data_ts).total_seconds() / 60
        tripped = age_minutes > _STALE_DATA_MINUTES
        if tripped:
            logger.warning(
                "KILL SWITCH: stale data — last update %.1f minutes ago (limit %d)",
                age_minutes, _STALE_DATA_MINUTES,
            )
        return tripped

    def check_order_rate_limit(self, orders_this_hour: int) -> bool:
        """True if the hourly order limit has been reached."""
        tripped = orders_this_hour >= _MAX_ORDERS_PER_HOUR
        if tripped:
            logger.warning(
                "KILL SWITCH: order rate limit — %d orders this hour (max %d)",
                orders_this_hour, _MAX_ORDERS_PER_HOUR,
            )
        return tripped

    def check_broker_disconnect(self) -> bool:
        """True if 3+ consecutive broker API failures have been recorded."""
        tripped = self._consecutive_broker_errors >= _BROKER_ERROR_LIMIT
        if tripped:
            logger.critical(
                "KILL SWITCH: broker disconnect — %d consecutive API failures",
                self._consecutive_broker_errors,
            )
        return tripped

    # ── State updates ─────────────────────────────────────────────────────

    def mark_broker_success(self) -> None:
        """Reset consecutive error counter after a successful API call."""
        self._consecutive_broker_errors = 0

    def mark_broker_error(self) -> None:
        """Increment consecutive error counter."""
        self._consecutive_broker_errors += 1
        logger.warning("Broker error count: %d / %d", self._consecutive_broker_errors, _BROKER_ERROR_LIMIT)

    # ── Master check ──────────────────────────────────────────────────────

    def assert_safe_to_trade(
        self,
        start_value: float,
        current_value: float,
        peak_value: float,
        day_trades_last_5: int,
        orders_this_hour: int,
        last_data_ts: Optional[datetime] = None,
    ) -> None:
        """Run all kill switches. Raises KillSwitchTripped if any fires.

        This is the single call the runner makes before placing any order.
        """
        if self.check_kill_file():
            raise KillSwitchTripped("Manual KILL file present — halted")
        if self.check_halt_file():
            raise KillSwitchTripped("HALTED file present — drawdown halt active. Delete to resume.")
        if self.check_broker_disconnect():
            raise KillSwitchTripped(f"Broker disconnect — {self._consecutive_broker_errors} consecutive failures")
        if self.check_daily_loss(start_value, current_value):
            raise KillSwitchTripped(f"Daily loss limit breached — portfolio: ${current_value:,.2f}, started: ${start_value:,.2f}")
        if self.check_drawdown(peak_value, current_value):
            raise KillSwitchTripped(f"Drawdown limit breached — portfolio: ${current_value:,.2f}, peak: ${peak_value:,.2f}")
        if self.check_pdt(day_trades_last_5):
            raise KillSwitchTripped(f"PDT guard — {day_trades_last_5} day trades in last 5 days")
        if self.check_order_rate_limit(orders_this_hour):
            raise KillSwitchTripped(f"Order rate limit — {orders_this_hour} orders this hour")
        if self.check_stale_data(last_data_ts):
            raise KillSwitchTripped("Stale market data during market hours")

    # ── File I/O ──────────────────────────────────────────────────────────

    def _write_halt_file(self, message: str) -> None:
        halt_path = self._root / _HALT_FILENAME
        try:
            halt_path.write_text(
                f"{datetime.now().isoformat()}\n{message}\n",
                encoding="utf-8",
            )
            logger.critical("HALTED file written to %s", halt_path)
        except Exception as exc:
            logger.error("Failed to write HALTED file: %s", exc)
