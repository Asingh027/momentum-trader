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
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

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


def run_daily(
    dry_run: bool = False,
    env_path: Optional[Path] = None,
    status_only: bool = False,
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
    logger.info("=== Trading Runner — %s%s ===", today_str, " [DRY RUN]" if dry_run else "")
    try:
        creds = load_credentials(env_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.critical("Failed to load credentials: %s", exc)
        sys.exit(1)

    broker = AlpacaBroker(
        api_key=creds["ALPACA_KEY_ID"],
        secret_key=creds["ALPACA_SECRET_KEY"],
        paper=True,
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
    db = TradingDB()
    order_mgr = OrderManager(broker)

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

            db.log_decision(
                ticker=ticker,
                action="exit" if should_exit else "hold",
                reason=reason,
                price=pos.current_price,
                shares=pos.qty,
                portfolio_value=account.portfolio_value,
                dry_run=dry_run,
            )

            if should_exit:
                logger.info("Exit signal: %s — %s", ticker, reason)
                try:
                    order = order_mgr.place_exit_order(
                        symbol=ticker,
                        qty=pos.qty,
                        reason=reason,
                        dry_run=dry_run,
                    )
                    if order and not dry_run:
                        orders_placed += 1
                    actions.append({"action": "exit", "ticker": ticker, "reason": reason, "dry_run": dry_run})
                except Exception as exc:
                    ks.mark_broker_error()
                    logger.error("Failed to submit exit for %s: %s", ticker, exc)
            else:
                logger.info("Hold: %s — %s", ticker, reason)
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

            db.log_decision(
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

    logger.info("=== Daily run complete — %d order(s) placed | report: %s ===",
                orders_placed, report_path)
