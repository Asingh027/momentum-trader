"""Intraday risk monitor — runs hourly during market hours.

Risk-only mode: never originates entries. Checks:
  1. Hard stop-loss breach per position (intraday price vs entry price)
  2. Regime gate: SPY current price < SPY 200d MA → exit all positions
  3. All kill switches (drawdown, daily loss, KILL file, broker disconnect)

SPY 200d SMA is cached per calendar day so hourly runs don't refetch bars.

Designed to be called from:
    python scripts/run_trading.py --monitor-only [--dry-run]

Approximate API calls per run (market open, 3 positions):
  get_account()         1
  get_positions()       1
  get_latest_quotes()   1  (batch: held tickers + SPY in one call)
  SPY bars fetch        0 (cached after first run of day) or 1
  exit orders           0-N (only if stops fire)
  ─────────────────────────
  Total                 3-4 typical (well under 100 limit)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from trader.config import TradingConfig
from trader.db import TradingDB
from trader.execution.alpaca_broker import AlpacaBroker
from trader.execution.order_manager import OrderManager
from trader.risk.exits import atr_stop_pct, compute_atr, trailing_lock_price
from trader.risk.kill_switches import KillSwitches, KillSwitchTripped
from trader.runner import load_credentials, _normalize_bars

logger = logging.getLogger(__name__)

# ── Intraday hard-stop threshold ──────────────────────────────────────────────
# Legacy fixed-stop fraction, kept for backward compatibility in tests and as
# the fallback when ATR can't be computed (insufficient bars for a new symbol).
INTRADAY_HARD_STOP_PCT = 0.05   # -5% from avg entry price

# ── ATR-based stop bounds ────────────────────────────────────────────────────
ATR_PERIOD = 14
ATR_STOP_FLOOR = 0.05
ATR_STOP_CAP = 0.12

# ── Trailing-lock parameters ─────────────────────────────────────────────────
TRAILING_LOCK_RATIO = 0.5
TRAILING_ACTIVATION_PCT = 0.15

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SPY_SMA_CACHE_FILENAME = "spy_sma200_cache.json"


# ── Pure helper functions (unit-testable without API) ─────────────────────────

def check_hard_stops(
    positions: list,
    current_prices: dict[str, float],
    hard_stop_pct: float = INTRADAY_HARD_STOP_PCT,
    per_ticker_stop_pct: Optional[dict[str, float]] = None,
) -> list[tuple]:
    """Return (position, reason) for each position that breached the intraday hard stop.

    Parameters
    ----------
    positions : list of Position dataclass instances
    current_prices : dict of symbol -> current price
    hard_stop_pct : float
        Default fraction below avg_entry_price that triggers the stop (e.g. 0.05 = -5%).
        Used when `per_ticker_stop_pct` lacks an entry for a symbol.
    per_ticker_stop_pct : dict, optional
        Override stop fraction per ticker (typically ATR-derived). Each value is
        the fraction below entry that fires the stop (e.g. {"NVDA": 0.09}).
    """
    per_ticker_stop_pct = per_ticker_stop_pct or {}
    exits = []
    for pos in positions:
        price = current_prices.get(pos.symbol)
        if price is None:
            logger.debug("No current price for %s — skipping hard stop check", pos.symbol)
            continue
        stop = per_ticker_stop_pct.get(pos.symbol, hard_stop_pct)
        pl_pct = (price - pos.avg_entry_price) / pos.avg_entry_price
        if pl_pct <= -stop:
            logger.info(
                "Hard stop fired: %s — current $%.2f vs entry $%.2f (%.1f%%, stop=%.1f%%)",
                pos.symbol, price, pos.avg_entry_price, pl_pct * 100, stop * 100,
            )
            exits.append((pos, "intraday_hard_stop"))
    return exits


def check_trailing_locks(
    positions: list,
    current_prices: dict[str, float],
    peaks: dict[str, dict],
    lock_ratio: float = TRAILING_LOCK_RATIO,
    activation_pct: float = TRAILING_ACTIVATION_PCT,
) -> list[tuple]:
    """Return (position, reason) for each position that breached its trailing lock.

    A position is in a trailing lock once its peak unrealized gain has exceeded
    `activation_pct`. After that, the stop is `lock_ratio * peak` above entry —
    e.g. peak +30%, ratio 0.5 → stop locks in +15%.

    Parameters
    ----------
    peaks : dict
        ticker -> dict with at least {'peak_pct': float, 'entry_price': float}.
        Pulled from the position_peaks table.
    """
    exits = []
    for pos in positions:
        price = current_prices.get(pos.symbol)
        if price is None:
            continue
        peak = peaks.get(pos.symbol)
        if peak is None:
            continue
        entry_price = peak.get("entry_price")
        peak_pct = peak.get("peak_pct")
        if not isinstance(entry_price, (int, float)) or not isinstance(peak_pct, (int, float)):
            continue
        lock = trailing_lock_price(
            entry_price=entry_price,
            peak_pct=peak_pct,
            lock_ratio=lock_ratio,
            activation_pct=activation_pct,
        )
        if lock is None:
            continue
        if price <= lock:
            logger.info(
                "Trailing lock fired: %s — current $%.2f <= lock $%.2f (peak +%.1f%%)",
                pos.symbol, price, lock, peak_pct * 100,
            )
            exits.append((pos, "trailing_lock"))
    return exits


def compute_atr_stops(
    bars: dict[str, pd.DataFrame],
    current_prices: dict[str, float],
    period: int = ATR_PERIOD,
    floor: float = ATR_STOP_FLOOR,
    cap: float = ATR_STOP_CAP,
) -> dict[str, float]:
    """Compute the ATR-based stop fraction for each ticker with sufficient bars.

    Returns a dict ticker -> stop_pct (fraction). Tickers without enough history
    are omitted; callers should fall back to `INTRADAY_HARD_STOP_PCT` for them.
    """
    out: dict[str, float] = {}
    for ticker, df in bars.items():
        if df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
            continue
        close_now = current_prices.get(ticker)
        if close_now is None:
            close_now = float(df["Close"].iloc[-1])
        atr = compute_atr(
            highs=df["High"].tolist(),
            lows=df["Low"].tolist(),
            closes=df["Close"].tolist(),
            period=period,
        )
        if atr is None:
            continue
        out[ticker] = atr_stop_pct(atr=atr, close=close_now, floor=floor, cap=cap)
    return out


def check_regime_gate(spy_price: float, spy_sma200: float) -> bool:
    """Return True if SPY has broken below its 200-day MA (regime exit signal)."""
    return spy_price < spy_sma200


def load_spy_sma_cache(cache_dir: Path, today_str: str) -> Optional[float]:
    """Return cached SPY 200d SMA for today, or None if stale/missing."""
    path = cache_dir / _SPY_SMA_CACHE_FILENAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("date") == today_str:
            return float(data["sma200"])
    except Exception as exc:
        logger.debug("SPY SMA cache read failed: %s", exc)
    return None


def save_spy_sma_cache(cache_dir: Path, today_str: str, sma200: float) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / _SPY_SMA_CACHE_FILENAME
    path.write_text(
        json.dumps({"date": today_str, "sma200": round(sma200, 4)}),
        encoding="utf-8",
    )


def compute_spy_sma200(spy_bars: pd.DataFrame) -> Optional[float]:
    """Compute current SPY 200d SMA from a bars DataFrame."""
    if spy_bars.empty or "Close" not in spy_bars.columns:
        return None
    close = spy_bars["Close"].dropna()
    if len(close) < 200:
        logger.warning("Only %d SPY bars — 200d SMA may be inaccurate", len(close))
    return float(close.rolling(200, min_periods=50).mean().iloc[-1])


# ── Main monitor entry point ───────────────────────────────────────────────────

def run_intraday_monitor(
    dry_run: bool = False,
    env_path: Optional[Path] = None,
    live: bool = False,
) -> None:
    """Intraday risk-only monitor. Checks stops and regime; never places entries.

    Exits silently (no log output) if all positions are safe and no kill switches
    are tripped — keeps cron output clean on uneventful hours.
    """
    import dataclasses

    now_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    today_str = date.today().isoformat()

    # ── 1. Credentials + connect ───────────────────────────────────────────
    try:
        creds = load_credentials(env_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.critical("Monitor: failed to load credentials: %s", exc)
        sys.exit(1)

    logger.info("Monitor endpoint: %s", "api.alpaca.markets" if live else "paper-api.alpaca.markets")
    broker = AlpacaBroker(
        api_key=creds["ALPACA_KEY_ID"],
        secret_key=creds["ALPACA_SECRET_KEY"],
        paper=not live,
    )

    # ── 2. Auth ────────────────────────────────────────────────────────────
    try:
        account = broker.get_account()
    except Exception as exc:
        logger.critical("Monitor: Alpaca auth failed: %s — halting", exc)
        sys.exit(1)

    # ── 3. Kill switches ───────────────────────────────────────────────────
    ks = KillSwitches(project_root=_PROJECT_ROOT)
    from trader.db import get_db_path
    live_db_path = get_db_path().parent / "trader_live.db" if live else None
    db = TradingDB(db_path=live_db_path)
    order_mgr = OrderManager(broker)

    start_value = db.get_start_of_day_value(today_str) or account.portfolio_value
    peak_value = db.get_peak_value() or account.portfolio_value
    pdt_count = db.count_day_trades_last_5_days()

    kill_tripped = False
    kill_reason = "OK"
    try:
        ks.assert_safe_to_trade(
            start_value=start_value,
            current_value=account.portfolio_value,
            peak_value=peak_value,
            day_trades_last_5=pdt_count,
            orders_this_hour=order_mgr.orders_this_hour(),
            last_data_ts=None,
        )
    except KillSwitchTripped as exc:
        kill_tripped = True
        kill_reason = str(exc)
        logger.warning("Monitor: kill switch tripped — %s", kill_reason)
        # Still run position checks: we may need to exit positions despite the kill switch

    # ── 4. Positions ───────────────────────────────────────────────────────
    try:
        positions = broker.get_positions()
        ks.mark_broker_success()
    except Exception as exc:
        ks.mark_broker_error()
        logger.error("Monitor: failed to fetch positions: %s", exc)
        return

    if not positions:
        # No open positions — nothing to monitor. Exit silently.
        logger.debug("Monitor: no open positions — exiting silently")
        return

    held_symbols = [p.symbol for p in positions]

    # ── 5. Build config ────────────────────────────────────────────────────
    V1_OVERRIDES = dict(
        breakout_lookback=126, trend_sma_short=50, relative_strength_lookback=63,
        relative_strength_min_outperformance=0.05, trailing_sma_days=50,
        hard_stop_pct=0.08, profit_target_pct=0.0, time_stop_days=9999,
        stop_loss_pct=0.0, use_volume_filter=False, use_regime_gate=True,
        earnings_buffer_days=3, min_price=20.0, max_price=500.0,
    )
    cfg = dataclasses.replace(TradingConfig.from_env(), **V1_OVERRIDES)

    # ── 6. SPY 200d SMA (cached per day) ──────────────────────────────────
    cache_dir = _PROJECT_ROOT / "data"
    spy_sma200 = load_spy_sma_cache(cache_dir, today_str)

    if spy_sma200 is None:
        logger.info("Monitor: SPY SMA cache miss — fetching bars …")
        bar_start = (pd.Timestamp(today_str) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
        try:
            spy_raw = broker.get_bars(symbols=["SPY"], start=bar_start, end=today_str)
            ks.mark_broker_success()
        except Exception as exc:
            ks.mark_broker_error()
            logger.error("Monitor: failed to fetch SPY bars: %s", exc)
            spy_raw = {}

        spy_map = _normalize_bars(spy_raw)
        spy_bars = spy_map.get("SPY", pd.DataFrame())
        spy_sma200 = compute_spy_sma200(spy_bars)

        if spy_sma200 is not None:
            save_spy_sma_cache(cache_dir, today_str, spy_sma200)
            logger.info("Monitor: SPY 200d SMA = %.2f (cached)", spy_sma200)
        else:
            logger.warning("Monitor: could not compute SPY 200d SMA — regime check skipped")

    # ── 7. Current prices (one batch call) ────────────────────────────────
    symbols_to_quote = held_symbols + (["SPY"] if "SPY" not in held_symbols else [])
    try:
        current_prices = broker.get_latest_quotes(symbols_to_quote)
        ks.mark_broker_success()
        logger.debug("Monitor: fetched %d quotes: %s", len(current_prices),
                     {s: f"${p:.2f}" for s, p in current_prices.items()})
    except Exception as exc:
        ks.mark_broker_error()
        logger.error("Monitor: failed to fetch quotes: %s", exc)
        return

    # ── 7b. Per-ticker ATR stops (fetch ~30 daily bars for held symbols) ──
    atr_stops: dict[str, float] = {}
    try:
        bar_start = (pd.Timestamp(today_str) - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
        held_bars_raw = broker.get_bars(symbols=held_symbols, start=bar_start, end=today_str)
        ks.mark_broker_success()
        held_bars = _normalize_bars(held_bars_raw)
        atr_stops = compute_atr_stops(held_bars, current_prices)
        if atr_stops:
            logger.info(
                "Monitor: ATR stops — %s",
                {s: f"{p*100:.1f}%" for s, p in atr_stops.items()},
            )
    except Exception as exc:
        ks.mark_broker_error()
        logger.warning("Monitor: ATR stop fetch failed (%s) — falling back to fixed %.1f%% stop",
                       exc, INTRADAY_HARD_STOP_PCT * 100)

    # ── 7c. Update peaks from current prices ───────────────────────────────
    peaks: dict[str, dict] = {}
    for pos in positions:
        price = current_prices.get(pos.symbol)
        if price is None or pos.avg_entry_price <= 0:
            continue
        current_gain = (price - pos.avg_entry_price) / pos.avg_entry_price
        db.upsert_peak(
            ticker=pos.symbol,
            peak_pct=max(current_gain, 0.0),
            entry_price=pos.avg_entry_price,
        )
        peak_row = db.get_peak(pos.symbol)
        if peak_row is not None:
            peaks[pos.symbol] = peak_row

    # ── 8. Regime gate check ───────────────────────────────────────────────
    exits_to_make: list[tuple] = []  # (position, reason)

    spy_price = current_prices.get("SPY")
    regime_fired = False
    if spy_sma200 is not None and spy_price is not None:
        regime_fired = check_regime_gate(spy_price, spy_sma200)
        if regime_fired:
            logger.warning(
                "Monitor: REGIME GATE fired — SPY $%.2f < 200d SMA $%.2f — exiting ALL positions",
                spy_price, spy_sma200,
            )
            for pos in positions:
                exits_to_make.append((pos, "intraday_regime_stop"))

    # ── 9. Hard stop + trailing lock checks (skip if regime exiting all) ──
    if not regime_fired:
        hard_stop_exits = check_hard_stops(
            positions, current_prices,
            hard_stop_pct=INTRADAY_HARD_STOP_PCT,
            per_ticker_stop_pct=atr_stops,
        )
        exits_to_make.extend(hard_stop_exits)

        # Trailing-lock only fires for positions not already flagged by hard stop
        already_exiting = {p.symbol for p, _ in exits_to_make}
        survivors = [p for p in positions if p.symbol not in already_exiting]
        trailing_exits = check_trailing_locks(survivors, current_prices, peaks)
        exits_to_make.extend(trailing_exits)

    # ── 10. Nothing fired? Exit silently ──────────────────────────────────
    if not exits_to_make and not kill_tripped:
        logger.debug(
            "Monitor: all clear — %d position(s) safe at %s",
            len(positions), now_str,
        )
        return

    # ── 11. Something fired — log prominently and act ─────────────────────
    if kill_tripped:
        logger.warning("Monitor [%s]: kill switch active — %s | positions=%d%s",
                       now_str, kill_reason, len(positions),
                       " [DRY RUN]" if dry_run else "")

    orders_placed = 0
    for pos, reason in exits_to_make:
        current_price = current_prices.get(pos.symbol, pos.current_price)
        db.log_decision(
            ticker=pos.symbol,
            action="exit",
            reason=reason,
            price=current_price,
            shares=pos.qty,
            portfolio_value=account.portfolio_value,
            dry_run=dry_run,
        )

        logger.warning(
            "Monitor: EXIT %s — %s | price=$%.2f entry=$%.2f pl=%.1f%%%s",
            pos.symbol, reason, current_price, pos.avg_entry_price,
            (current_price - pos.avg_entry_price) / pos.avg_entry_price * 100,
            " [DRY RUN]" if dry_run else "",
        )

        try:
            order = order_mgr.place_exit_order(
                symbol=pos.symbol,
                qty=pos.qty,
                reason=reason,
                dry_run=dry_run,
            )
            if not dry_run:
                orders_placed += 1
                db.clear_peak(pos.symbol)
        except Exception as exc:
            ks.mark_broker_error()
            logger.error("Monitor: failed to submit exit for %s: %s", pos.symbol, exc)

    # ── 12. Update DB summary ──────────────────────────────────────────────
    if exits_to_make:
        unrealized_pnl = sum(p.unrealized_pl for p in positions)
        db.upsert_daily_summary(
            date_str=today_str,
            portfolio_value=account.portfolio_value,
            cash=account.cash,
            equity=account.equity,
            open_positions=len(positions),
            realized_pnl=0.0,
            unrealized_pnl=unrealized_pnl,
            start_of_day_value=start_value,
            peak_value=max(peak_value, account.portfolio_value),
            kill_switch_status=kill_reason,
            orders_placed=orders_placed,
        )

    logger.info("Monitor: run complete — %d exit(s) fired, %d order(s) placed%s",
                len(exits_to_make), orders_placed,
                " [DRY RUN]" if dry_run else "")
