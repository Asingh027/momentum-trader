"""Daily execution runner — core of the live trading loop.

Designed to run once per trading day (scheduled via cron or Task Scheduler).
Typical invocation: 3:50 PM ET (10 min before market close).

Flow:
  1. Load config + credentials, connect to Alpaca
  2. Verify auth — if it fails, halt immediately
  3. Check all kill switches
  4. Fetch current positions + account state
  5. Fetch recent bars for universe + held tickers (from Alpaca IEX feed)
  6. Run exit signals on open positions → submit sell orders
  7. Run entry signals on universe → rank by RS → submit buy orders
  8. Log all decisions to SQLite
  9. Write daily report markdown
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from trader.risk.exits import (
    is_in_cooldown,
    next_partial_milestone,
    trailing_lock_price,
)

TRAILING_LOCK_RATIO = 0.5
TRAILING_ACTIVATION_PCT = 0.15
PARTIAL_MILESTONES = [(0.25, 1 / 3), (0.50, 1 / 3)]
COOLDOWN_TRADING_DAYS = 5

try:
    from notify import send as _notify
except ImportError:
    _notify = lambda *a, **kw: None  # noqa: E731

logger = logging.getLogger(__name__)

# Project root is two levels up from this file (src/trader/runner.py -> src/ -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Universe (same as backtest) ───────────────────────────────────────────────
BACKTEST_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "AMD", "INTC", "QCOM", "TXN",
    "CRM", "ADBE", "NOW", "ORCL", "IBM", "HPQ", "AMAT", "KLAC", "LRCX",
    "JNJ", "UNH", "LLY", "ABBV", "MRK", "PFE", "ABT", "TMO", "DHR", "MDT",
    "BMY", "AMGN", "GILD",
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "USB", "PNC",
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "TGT", "SBUX", "GM", "F",
    "PG", "KO", "PEP", "WMT", "COST", "CL", "MO",
    "CAT", "DE", "HON", "UPS", "BA", "GE", "LMT", "RTX",
    "XOM", "CVX", "COP", "SLB",
    "LIN", "APD", "ECL",
    "NEE", "DUK", "SO",
    "AMT", "PLD", "EQIX",
    "DIS", "NFLX", "CMCSA", "T", "VZ",
]


def load_credentials(env_path: Optional[Path] = None) -> dict[str, str]:
    """Load Alpaca credentials from the env file. Never commit this file."""
    _env_var = os.environ.get("ALPACA_ENV_PATH", "")
    default = Path(_env_var) if _env_var.strip() else Path(r"C:\Users\Avneet\Documents\Trading Helper\alpaca.env")
    path = env_path or default
    if not path.exists():
        raise FileNotFoundError(
            f"Alpaca credentials not found at {path}. "
            "Create the file with ALPACA_ENDPOINT, ALPACA_KEY_ID, ALPACA_SECRET_KEY."
        )
    creds: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            creds[key.strip()] = val.strip()
    required = ["ALPACA_KEY_ID", "ALPACA_SECRET_KEY"]
    for k in required:
        if k not in creds:
            raise ValueError(f"Missing {k} in {path}")
    return creds


def _normalize_bars(bars: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Ensure bars have tz-naive date-only index."""
    out = {}
    for sym, df in bars.items():
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = df.index.normalize()
        out[sym] = df
    return out


def _compute_entry_signals_live(
    bars: dict[str, pd.DataFrame],
    spy_bars: pd.DataFrame,
    cfg,
) -> dict[str, bool]:
    """Compute today's entry signals. Returns dict of ticker -> True/False."""
    from trader.signals.momentum import compute_entry_signals

    if spy_bars.empty:
        return {}

    entries_df = compute_entry_signals(bars, spy_bars, cfg)
    if entries_df.empty:
        return {}

    # Get the signal on the last available date
    last_date = entries_df.index[-1]
    last_row = entries_df.loc[last_date]
    return {ticker: bool(val) for ticker, val in last_row.items()}


def _compute_exit_signals_live(
    bars: dict[str, pd.DataFrame],
    spy_bars: pd.DataFrame,
    cfg,
    held_tickers: list[str],
) -> dict[str, tuple[bool, str]]:
    """Compute today's exit signals for held positions.

    Returns dict of ticker -> (should_exit: bool, reason: str).
    """
    from trader.signals.momentum import _sma

    if not held_tickers or spy_bars.empty:
        return {}

    spy_close = spy_bars["Close"]
    spy_sma200 = _sma(spy_close, cfg.spy_trend_sma_period)

    last_date = spy_bars.index[-1]
    spy_val = spy_close.get(last_date, None)
    spy_sma_val = spy_sma200.get(last_date, None)
    regime_exit = spy_val is not None and spy_sma_val is not None and spy_val < spy_sma_val

    result: dict[str, tuple[bool, str]] = {}
    for ticker in held_tickers:
        if ticker not in bars or bars[ticker].empty:
            result[ticker] = (False, "no_data")
            continue

        close = bars[ticker]["Close"].reindex(spy_bars.index).ffill(limit=5)
        trailing_sma = _sma(close, cfg.trailing_sma_days)

        if last_date not in close.index:
            result[ticker] = (False, "no_data")
            continue

        price = close.get(last_date, None)
        sma_val = trailing_sma.get(last_date, None)

        trailing_exit = price is not None and sma_val is not None and price < sma_val

        if regime_exit:
            result[ticker] = (True, "regime_stop")
        elif trailing_exit:
            result[ticker] = (True, "trailing_sma")
        else:
            result[ticker] = (False, "hold")

    return result


def _rank_by_relative_strength(
    signals: dict[str, bool],
    bars: dict[str, pd.DataFrame],
    spy_bars: pd.DataFrame,
    rs_lookback: int,
) -> list[str]:
    """Sort signaling tickers by 63-day RS vs SPY, strongest first."""
    if spy_bars.empty:
        return [t for t, v in signals.items() if v]

    last_date = spy_bars.index[-1]
    spy_close = spy_bars["Close"]
    rs_scores: dict[str, float] = {}

    for ticker, fired in signals.items():
        if not fired:
            continue
        if ticker not in bars:
            continue
        close = bars[ticker]["Close"].reindex(spy_bars.index).ffill(limit=5)
        if last_date not in close.index or last_date not in spy_close.index:
            continue
        lookback_date_idx = close.index.searchsorted(last_date) - rs_lookback
        if lookback_date_idx < 0:
            continue
        lookback_date = close.index[max(0, lookback_date_idx)]
        stock_ret = close.get(last_date, None)
        stock_old = close.get(lookback_date, None)
        spy_ret = spy_close.get(last_date, None)
        spy_old = spy_close.get(lookback_date, None)
        if any(v is None or v == 0 for v in [stock_ret, stock_old, spy_ret, spy_old]):
            continue
        rs = (stock_ret / stock_old) - (spy_ret / spy_old)
        rs_scores[ticker] = rs

    return sorted(rs_scores.keys(), key=lambda t: rs_scores[t], reverse=True)


def _update_fill(broker, db, row_id: int, order_id: str, ticker: str) -> None:
    """Sleep 2 s, fetch actual fill from Alpaca, write price + shares back to DB."""
    time.sleep(2)
    try:
        o = broker.get_order_by_id(order_id)
        if o.filled_avg_price is not None and o.filled_qty > 0:
            db.update_decision_fill(row_id, order_id, o.filled_avg_price, o.filled_qty)
            logger.info("Fill confirmed: %s %.4f sh @ $%.2f", ticker, o.filled_qty, o.filled_avg_price)
        else:
            db.update_decision_fill(row_id, order_id, None, None)
            logger.info("Order %s not yet filled for %s — order_id stored, will reconcile next run", order_id, ticker)
    except Exception as exc:
        logger.warning("Could not fetch fill for %s order %s: %s", ticker, order_id, exc)


def _reconcile_pending_fills(broker, db) -> None:
    """At session start: update any decisions from prior runs that have an order_id but no fill price."""
    since = (date.today() - timedelta(days=3)).isoformat()
    pending = db.get_pending_fills(since_date=since)
    if not pending:
        return
    logger.info("Reconciling %d unfilled decision(s) from prior sessions", len(pending))
    for row in pending:
        try:
            o = broker.get_order_by_id(row["order_id"])
            if o.filled_avg_price is not None and o.filled_qty > 0:
                db.update_decision_fill(row["id"], row["order_id"], o.filled_avg_price, o.filled_qty)
                logger.info("Reconciled %s: %.4f sh @ $%.2f", row["ticker"], o.filled_qty, o.filled_avg_price)
            else:
                logger.info("Order %s (%s) still pending — status=%s", row["order_id"], row["ticker"], o.status)
        except Exception as exc:
            logger.warning("Reconcile failed for order %s (%s): %s", row["order_id"], row["ticker"], exc)


def run_daily(
    dry_run: bool = False,
    env_path: Optional[Path] = None,
    status_only: bool = False,
    live: bool = False,
) -> None:
    """Main daily execution function.

    Parameters
    ----------
    dry_run : bool
        If True, compute signals but do not place any orders.
    env_path : Path, optional
        Path to alpaca.env credentials file.
    status_only : bool
        If True, print account status and positions, then exit.
    """
    from trader.config import TradingConfig
    from trader.db import TradingDB
    from trader.execution.alpaca_broker import AlpacaBroker
    from trader.execution.order_manager import OrderManager
    from trader.notifications import write_daily_report
    from trader.risk.kill_switches import KillSwitches, KillSwitchTripped
    from trader.risk.sizing import compute_available_capital, compute_position_notional
    import dataclasses

    today = date.today()
    today_str = today.isoformat()

    # ── 1. Credentials + connection ────────────────────────────────────────
    mode_tag = ("LIVE" if live else "PAPER") + (" [DRY RUN]" if dry_run else "")
    tag = "[LIVE]" if live else "[PAPER]"
    logger.info("=== Trading Runner — %s | %s ===", today_str, mode_tag)
    logger.info("Broker endpoint: %s", "api.alpaca.markets" if live else "paper-api.alpaca.markets")
    try:
        creds = load_credentials(env_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.critical("Failed to load credentials: %s", exc)
        sys.exit(1)

    broker = AlpacaBroker(
        api_key=creds["ALPACA_KEY_ID"],
        secret_key=creds["ALPACA_SECRET_KEY"],
        paper=not live,
    )

    # ── 2. Auth verification ───────────────────────────────────────────────
    try:
        account = broker.get_account()
        logger.info(
            "AUTH OK — account %s | status=%s | portfolio=$%.2f | cash=$%.2f | buying_power=$%.2f",
            account.account_id, account.status,
            account.portfolio_value, account.cash, account.buying_power,
        )
    except Exception as exc:
        logger.critical("Alpaca auth FAILED: %s — halting", exc)
        sys.exit(1)

    if status_only:
        positions = broker.get_positions()
        print(f"\nAccount: {account.account_id} | Status: {account.status}")
        print(f"Portfolio: ${account.portfolio_value:,.2f} | Cash: ${account.cash:,.2f} | Buying power: ${account.buying_power:,.2f}")
        print(f"\nOpen positions ({len(positions)}):")
        for p in positions:
            print(f"  {p.symbol:6s} {p.qty:.4f} shares @ ${p.avg_entry_price:.2f} | "
                  f"Current ${p.current_price:.2f} | PnL ${p.unrealized_pl:+.2f} ({p.unrealized_plpc*100:+.1f}%)")
        return

    # ── 3. Kill switches ───────────────────────────────────────────────────
    ks = KillSwitches(project_root=_PROJECT_ROOT)
    # Live mode uses a separate DB so paper history doesn't corrupt kill-switch math
    from trader.db import get_db_path
    live_db_path = get_db_path().parent / "trader_live.db" if live else None
    db = TradingDB(db_path=live_db_path)
    order_mgr = OrderManager(broker)

    _reconcile_pending_fills(broker, db)

    start_value = db.get_start_of_day_value(today_str) or account.portfolio_value
    peak_value = db.get_peak_value() or account.portfolio_value
    pdt_count = db.count_day_trades_last_5_days()
    kill_status = "OK"

    try:
        ks.assert_safe_to_trade(
            start_value=start_value,
            current_value=account.portfolio_value,
            peak_value=peak_value,
            day_trades_last_5=pdt_count,
            orders_this_hour=order_mgr.orders_this_hour(),
            last_data_ts=None,  # will be updated after fetching bars
        )
    except KillSwitchTripped as exc:
        logger.warning("Kill switch tripped: %s — no new entries, exits still run", exc)
        kill_status = str(exc)
        # Continue to process exits even when kill switch is tripped
        # but block new entries (handled below by setting entries_blocked=True)
        entries_blocked = True
        _notify(f"🚨 {tag} <b>KILL SWITCH</b>\n{kill_status}")
    else:
        entries_blocked = False

    # ── 4. Current positions ───────────────────────────────────────────────
    try:
        positions = broker.get_positions()
        ks.mark_broker_success()
    except Exception as exc:
        ks.mark_broker_error()
        logger.error("Failed to fetch positions: %s", exc)
        positions = []

    held_tickers = [p.symbol for p in positions]
    logger.info("Held positions: %s", held_tickers or "none")

    # ── 5. Fetch bars ──────────────────────────────────────────────────────
    bar_end = today_str
    # Need 500 days of history for 200d SMA + 126d breakout lookback
    bar_start = (pd.Timestamp(today) - pd.Timedelta(days=520)).strftime("%Y-%m-%d")

    universe_to_fetch = list(set(BACKTEST_UNIVERSE + held_tickers))
    logger.info("Fetching bars for %d symbols from %s to %s …", len(universe_to_fetch), bar_start, bar_end)

    try:
        bars_raw = broker.get_bars(symbols=universe_to_fetch, start=bar_start, end=bar_end)
        spy_raw = broker.get_bars(symbols=["SPY"], start=bar_start, end=bar_end)
        ks.mark_broker_success()
    except Exception as exc:
        ks.mark_broker_error()
        logger.error("Failed to fetch bars: %s", exc)
        bars_raw, spy_raw = {}, {}

    bars = _normalize_bars(bars_raw)
    spy_map = _normalize_bars(spy_raw)
    spy_bars = spy_map.get("SPY", pd.DataFrame())

    if spy_bars.empty:
        logger.error("No SPY bars available — cannot compute signals. Halting.")
        sys.exit(1)

    last_data_ts = datetime.utcnow()  # we just fetched — data is fresh

    # Re-check stale data now that we have a timestamp
    if ks.check_stale_data(last_data_ts):
        logger.warning("Data stale check fired after fetch — proceeding with caution")

    # ── Build config ───────────────────────────────────────────────────────
    import dataclasses
    from trader.config import TradingConfig
    V1_OVERRIDES = dict(
        breakout_lookback=126, trend_sma_short=50, relative_strength_lookback=63,
        relative_strength_min_outperformance=0.05, trailing_sma_days=50,
        hard_stop_pct=0.08, profit_target_pct=0.0, time_stop_days=9999,
        stop_loss_pct=0.0, use_volume_filter=False, use_regime_gate=True,
        earnings_buffer_days=3, min_price=20.0, max_price=500.0,
    )
    cfg = dataclasses.replace(TradingConfig.from_env(), **V1_OVERRIDES)

    # ── 6. Exit signals ────────────────────────────────────────────────────
    actions: list[dict] = []
    orders_placed = 0

    if held_tickers:
        exit_signals = _compute_exit_signals_live(bars, spy_bars, cfg, held_tickers)
        for ticker, (should_exit, reason) in exit_signals.items():
            pos = next((p for p in positions if p.symbol == ticker), None)
            if pos is None:
                continue

            # Refresh peak from today's close before evaluating trailing lock /
            # milestones — ensures we don't act on stale peaks.
            current_gain = pos.unrealized_plpc
            db.upsert_peak(
                ticker=ticker,
                peak_pct=max(current_gain, 0.0),
                entry_price=pos.avg_entry_price,
            )
            peak_row = db.get_peak(ticker) or {}
            peak_pct = peak_row.get("peak_pct", max(current_gain, 0.0))
            sold_milestones = peak_row.get("partial_sold_at", [])

            # Trailing-lock fires as a full exit (overrides SMA exit).
            lock_price = trailing_lock_price(
                entry_price=pos.avg_entry_price,
                peak_pct=peak_pct,
                lock_ratio=TRAILING_LOCK_RATIO,
                activation_pct=TRAILING_ACTIVATION_PCT,
            )
            trailing_lock_breached = (
                lock_price is not None and pos.current_price <= lock_price
            )

            full_exit = should_exit or trailing_lock_breached
            final_reason = (
                "trailing_lock" if trailing_lock_breached and not should_exit else reason
            )

            row_id = db.log_decision(
                ticker=ticker,
                action="exit" if full_exit else "hold",
                reason=final_reason,
                price=pos.current_price,
                shares=pos.qty,
                portfolio_value=account.portfolio_value,
                dry_run=dry_run,
            )

            if full_exit:
                logger.info("Exit signal: %s — %s", ticker, final_reason)
                try:
                    order = order_mgr.place_exit_order(
                        symbol=ticker,
                        qty=pos.qty,
                        reason=final_reason,
                        dry_run=dry_run,
                    )
                    if order and not dry_run:
                        orders_placed += 1
                        db.clear_peak(ticker)
                        _notify(
                            f"📉 {tag} <b>EXIT</b> {ticker}\n"
                            f"Reason: {final_reason}\n"
                            f"P&amp;L: ${pos.unrealized_pl:+.2f} ({pos.unrealized_plpc*100:+.1f}%)"
                        )
                        _update_fill(broker, db, row_id, order.order_id, ticker)
                    actions.append({"action": "exit", "ticker": ticker, "reason": final_reason, "dry_run": dry_run})
                except Exception as exc:
                    ks.mark_broker_error()
                    logger.error("Failed to submit exit for %s: %s", ticker, exc)
                continue

            # Not exiting fully — check partial profit milestones.
            milestone = next_partial_milestone(
                current_gain_pct=current_gain,
                sold_milestones=sold_milestones,
                milestones=PARTIAL_MILESTONES,
            )
            if milestone is not None:
                threshold, fraction = milestone
                sell_qty = pos.qty * fraction
                partial_reason = f"partial_profit_{int(threshold*100)}pct"
                logger.info(
                    "Partial profit: %s — selling %.4f sh (%.0f%%) at +%.1f%% gain",
                    ticker, sell_qty, fraction * 100, current_gain * 100,
                )
                partial_row = db.log_decision(
                    ticker=ticker,
                    action="exit",
                    reason=partial_reason,
                    price=pos.current_price,
                    shares=sell_qty,
                    portfolio_value=account.portfolio_value,
                    dry_run=dry_run,
                )
                try:
                    order = order_mgr.place_exit_order(
                        symbol=ticker,
                        qty=sell_qty,
                        reason=partial_reason,
                        dry_run=dry_run,
                    )
                    if order and not dry_run:
                        orders_placed += 1
                        db.upsert_peak(
                            ticker=ticker,
                            peak_pct=peak_pct,
                            entry_price=pos.avg_entry_price,
                            partial_sold_at=sorted(set(sold_milestones) | {threshold}),
                        )
                        _notify(
                            f"💰 {tag} <b>PARTIAL</b> {ticker}\n"
                            f"Sold {fraction*100:.0f}% at +{current_gain*100:.1f}% (milestone +{threshold*100:.0f}%)"
                        )
                        _update_fill(broker, db, partial_row, order.order_id, ticker)
                    actions.append({
                        "action": "exit",
                        "ticker": ticker,
                        "reason": partial_reason,
                        "dry_run": dry_run,
                    })
                except Exception as exc:
                    ks.mark_broker_error()
                    logger.error("Failed to submit partial exit for %s: %s", ticker, exc)
            else:
                logger.info("Hold: %s — %s", ticker, final_reason)
                actions.append({"action": "hold", "ticker": ticker, "reason": "continuing trend"})

    # ── 7. Entry signals ───────────────────────────────────────────────────
    if not entries_blocked:
        # Exclude already-held tickers
        universe_bars = {t: bars[t] for t in BACKTEST_UNIVERSE if t in bars}
        entry_signals = _compute_entry_signals_live(universe_bars, spy_bars, cfg)

        n_signals = sum(1 for v in entry_signals.values() if v)
        logger.info("Entry signals today: %d / %d", n_signals, len(entry_signals))

        # Rank by relative strength
        ranked = _rank_by_relative_strength(entry_signals, bars, spy_bars, cfg.relative_strength_lookback)

        # Filter out tickers in post-stop cooldown
        cooled = []
        for ticker in ranked:
            last_stop = db.get_last_stop_exit_date(ticker)
            if is_in_cooldown(ticker, last_stop, cooldown_days=COOLDOWN_TRADING_DAYS, today=today):
                logger.info("Cooldown: skipping %s — last stop on %s", ticker, last_stop)
                db.log_decision(
                    ticker=ticker,
                    action="skip",
                    reason="cooldown_post_stop",
                    portfolio_value=account.portfolio_value,
                    dry_run=dry_run,
                )
                continue
            cooled.append(ticker)
        ranked = cooled

        # Determine how many new positions we can open
        current_positions = len(held_tickers)
        max_new = cfg.max_positions - current_positions
        cash_available = compute_available_capital(account.portfolio_value, account.cash, cfg)

        logger.info(
            "Cash available (after floor): $%.2f | Can open %d new positions",
            cash_available, max_new,
        )

        for ticker in ranked[:max_new]:
            notional = compute_position_notional(account.portfolio_value, cash_available, cfg)
            if notional <= 0:
                logger.info("Insufficient cash for more entries — stopping at %d", orders_placed)
                break

            row_id = db.log_decision(
                ticker=ticker,
                action="entry",
                reason="momentum_breakout",
                portfolio_value=account.portfolio_value,
                notional=notional,
                dry_run=dry_run,
            )

            try:
                order = order_mgr.place_entry_order(
                    symbol=ticker,
                    notional=notional,
                    dry_run=dry_run,
                )
                # Always deduct from cash_available (including dry-run) so each
                # iteration sees the correct remaining capital.
                orders_placed += 1
                cash_available -= notional
                if order and not dry_run:
                    _notify(f"📈 {tag} <b>ENTRY</b> {ticker} | ${notional:,.0f} | Momentum breakout")
                    _update_fill(broker, db, row_id, order.order_id, ticker)
                    # Seed a fresh peak record at gain=0 so partials/trailing
                    # logic has an entry_price to anchor against on the next run.
                    db.clear_peak(ticker)
                    db.upsert_peak(
                        ticker=ticker,
                        peak_pct=0.0,
                        entry_price=None,  # entry price fills in on next monitor/runner pass
                    )
                actions.append({
                    "action": "entry",
                    "ticker": ticker,
                    "reason": f"momentum breakout — notional=${notional:,.2f}",
                    "dry_run": dry_run,
                })
            except Exception as exc:
                ks.mark_broker_error()
                logger.error("Failed to submit entry for %s: %s", ticker, exc)

        # Log skipped signals
        for ticker in [t for t, v in entry_signals.items() if v and t not in ranked[:max_new]]:
            db.log_decision(
                ticker=ticker,
                action="skip",
                reason="position_limit_or_rank",
                portfolio_value=account.portfolio_value,
                dry_run=dry_run,
            )
    else:
        logger.info("Entries blocked by kill switch: %s", kill_status)

    # ── 8. Update DB summary ──────────────────────────────────────────────
    unrealized_pnl = sum(p.unrealized_pl for p in positions)
    db.upsert_daily_summary(
        date_str=today_str,
        portfolio_value=account.portfolio_value,
        cash=account.cash,
        equity=account.equity,
        open_positions=len(positions),
        realized_pnl=0.0,   # Would need trades API to compute accurately
        unrealized_pnl=unrealized_pnl,
        start_of_day_value=start_value,
        peak_value=max(peak_value, account.portfolio_value),
        kill_switch_status=kill_status,
        orders_placed=orders_placed,
    )

    # ── 9. Daily report ───────────────────────────────────────────────────
    daily_pnl_pct = (account.portfolio_value - start_value) / start_value * 100 if start_value > 0 else 0.0
    drawdown_pct = (account.portfolio_value - peak_value) / peak_value * 100 if peak_value > 0 else 0.0

    pos_dicts = []
    for p in positions:
        # Approximate days held from DB
        pos_dicts.append({
            "symbol": p.symbol,
            "qty": p.qty,
            "avg_entry_price": p.avg_entry_price,
            "current_price": p.current_price,
            "unrealized_pl": p.unrealized_pl,
            "unrealized_plpc": p.unrealized_plpc,
            "days_held": 0,  # Could query DB for entry date
        })

    report_path = write_daily_report(
        report_date=today,
        portfolio_value=account.portfolio_value,
        portfolio_pct_change=daily_pnl_pct,
        cash=account.cash,
        positions=pos_dicts,
        actions=actions,
        daily_pnl_pct=daily_pnl_pct,
        drawdown_pct=abs(drawdown_pct),
        pdt_count=pdt_count,
        kill_switch_status=kill_status,
        dry_run=dry_run,
    )

    _notify(
        f"📊 {tag} <b>EOD {today_str}</b>\n"
        f"Portfolio: ${account.portfolio_value:,.0f} ({daily_pnl_pct:+.1f}%)\n"
        f"{len(positions)} positions | Cash: ${account.cash:,.0f}\n"
        f"Orders: {orders_placed} | KS: {kill_status}"
    )
    logger.info("=== Daily run complete — %d order(s) placed | report: %s ===",
                orders_placed, report_path)
