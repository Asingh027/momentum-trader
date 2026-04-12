#!/usr/bin/env python
"""Strategy v1.0 — momentum/trend-following backtest entry point.

Usage:
    uv run python scripts/run_v1.py
    uv run python scripts/run_v1.py --refresh   # force re-download

Outputs (all in outputs/):
    v1_report.md
    v1_equity_curve.png
    v1_trade_log.csv
    v1_top_trades.md
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import vectorbt as vbt

from trader.backtest.metrics import (
    annualised_return,
    compute_trade_stats,
    format_metrics_table,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    spy_benchmark,
    total_return,
)
from trader.backtest.walk_forward import (
    format_wf_table,
    generate_windows,
    slice_bars,
)
from trader.config import TradingConfig
from trader.data.bars import download_bars, load_spy
from trader.reports.equity_curve import plot_equity_curve
from trader.risk.filters import apply_position_cap, apply_price_filter
from trader.signals.momentum import compute_entry_signals, compute_exit_signals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Same 84-stock universe as prior phases — data already cached
BACKTEST_UNIVERSE = [
    # Technology
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "AMD", "INTC", "QCOM", "TXN",
    "CRM", "ADBE", "NOW", "ORCL", "IBM", "HPQ", "AMAT", "KLAC", "LRCX",
    # Healthcare
    "JNJ", "UNH", "LLY", "ABBV", "MRK", "PFE", "ABT", "TMO", "DHR", "MDT",
    "BMY", "AMGN", "GILD",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "USB", "PNC",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "TGT", "SBUX", "GM", "F",
    # Consumer Staples
    "PG", "KO", "PEP", "WMT", "COST", "CL", "MO",
    # Industrials
    "CAT", "DE", "HON", "UPS", "BA", "GE", "LMT", "RTX",
    # Energy
    "XOM", "CVX", "COP", "SLB",
    # Materials
    "LIN", "APD", "ECL",
    # Utilities
    "NEE", "DUK", "SO",
    # Real Estate
    "AMT", "PLD", "EQIX",
    # Communication Services
    "DIS", "NFLX", "CMCSA", "T", "VZ",
]

# v1.0 config overrides applied on top of TradingConfig defaults
V1_OVERRIDES = dict(
    # Momentum entry params
    breakout_lookback=126,
    trend_sma_short=50,
    relative_strength_lookback=63,
    relative_strength_min_outperformance=0.05,
    # Trailing SMA exit (no profit target, no time stop)
    trailing_sma_days=50,
    hard_stop_pct=0.08,
    profit_target_pct=0.0,   # disabled — set to 0 means no tp_stop
    time_stop_days=9999,      # effectively disabled (no time stop in v1)
    stop_loss_pct=0.0,        # hard stop handled separately via sl_stop below
    # Use regime gate; no volume filter (large caps, liquid)
    use_volume_filter=False,
    use_regime_gate=True,
    # Earnings buffer — 3 trading days
    earnings_buffer_days=3,
    # Universe filters — standard large-cap
    min_price=20.0,
    max_price=500.0,
)


def make_v1_config() -> TradingConfig:
    cfg = TradingConfig()
    return dataclasses.replace(cfg, **V1_OVERRIDES)


def _normalize_idx(df: pd.DataFrame) -> pd.DataFrame:
    """Strip timezone from DatetimeIndex and keep date-only."""
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = df.index.normalize()
    return df


def build_price_matrices(
    bars: dict[str, pd.DataFrame],
    tickers: list[str],
    common_idx: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    closes, opens = {}, {}
    for t in tickers:
        if t in bars:
            closes[t] = bars[t]["Close"].reindex(common_idx)
            opens[t] = bars[t]["Open"].reindex(common_idx)
    return pd.DataFrame(closes, index=common_idx), pd.DataFrame(opens, index=common_idx)


def attribute_exit_reason(
    row: pd.Series,
    bars: dict[str, pd.DataFrame],
    spy_bars: pd.DataFrame,
    cfg: TradingConfig,
    trailing_sma_cache: dict[str, pd.Series],
) -> str:
    """Infer exit reason for a single trade."""
    ticker = row.get("Column", "")
    ret = row.get("Return", 0.0)

    # Hard stop band
    if ret <= -(cfg.hard_stop_pct * 0.9):
        return "hard_stop"

    # Regime stop: check SPY on exit date
    exit_ts = row.get("Exit Timestamp")
    if exit_ts is not None:
        try:
            exit_date = pd.Timestamp(exit_ts).normalize()
            spy_close = spy_bars["Close"]
            from trader.signals.momentum import _sma as _m_sma
            spy_sma200 = _m_sma(spy_close, cfg.spy_trend_sma_period)
            spy_val = spy_close.get(exit_date, None)
            sma_val = spy_sma200.get(exit_date, None)
            if spy_val is not None and sma_val is not None and spy_val < sma_val:
                return "regime_stop"
        except Exception:
            pass

    # Trailing SMA: price < trailing SMA on exit
    if ticker in trailing_sma_cache and exit_ts is not None:
        try:
            exit_date = pd.Timestamp(exit_ts).normalize()
            tsma = trailing_sma_cache[ticker]
            if ticker in bars:
                close_val = bars[ticker]["Close"].get(exit_date, None)
                sma_val = tsma.get(exit_date, None)
                if close_val is not None and sma_val is not None and close_val < sma_val:
                    return "trailing_sma"
        except Exception:
            pass

    return "signal_exit_other"


def build_trade_log(
    pf: vbt.Portfolio,
    bars: dict[str, pd.DataFrame],
    spy_bars: pd.DataFrame,
    cfg: TradingConfig,
    trailing_sma_cache: dict[str, pd.Series],
) -> pd.DataFrame:
    """Build flat trade log with exit reason attribution."""
    try:
        trades = pf.trades.records_readable.copy()
    except Exception as exc:
        logger.warning("Could not extract trades: %s", exc)
        return pd.DataFrame()

    if trades.empty:
        return trades

    # Normalise column names
    rename_map = {
        "Entry Timestamp": "entry_date",
        "Exit Timestamp": "exit_date",
        "Avg Entry Price": "entry_price",
        "Avg Exit Price": "exit_price",
        "PnL": "pnl",
        "Return": "return_pct",
        "Column": "ticker",
        "Size": "size",
    }
    for old, new in rename_map.items():
        if old in trades.columns:
            trades.rename(columns={old: new}, inplace=True)

    # Keep the original for attribution
    trades_orig = pf.trades.records_readable.copy()

    exit_reasons = []
    for _, row in trades_orig.iterrows():
        reason = attribute_exit_reason(row, bars, spy_bars, cfg, trailing_sma_cache)
        exit_reasons.append(reason)
    trades["exit_reason"] = exit_reasons

    # Sort by entry_date
    entry_col = "entry_date" if "entry_date" in trades.columns else trades.columns[0]
    trades = trades.sort_values(entry_col)

    return trades


def exit_reason_breakdown(trades: pd.DataFrame) -> str:
    """Markdown table of exit reason distribution."""
    if trades.empty or "exit_reason" not in trades.columns:
        return "No trades."
    counts = trades["exit_reason"].value_counts()
    total = len(trades)
    rows = [("Exit Reason", "% of Trades", "Count"), ("---", "---", "---")]
    for reason, cnt in counts.items():
        rows.append((str(reason), f"{cnt/total:.1%}", str(cnt)))
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def per_ticker_summary(trades: pd.DataFrame) -> str:
    """Markdown table of per-ticker P&L."""
    if trades.empty:
        return "No trades."
    pnl_col = "pnl" if "pnl" in trades.columns else None
    ret_col = "return_pct" if "return_pct" in trades.columns else None
    ticker_col = "ticker" if "ticker" in trades.columns else "Column"
    if pnl_col is None or ticker_col not in trades.columns:
        return "Trade log missing pnl/ticker columns."

    grp = trades.groupby(ticker_col)
    rows = [("Ticker", "Trades", "Total P&L", "Avg Return", "Win%"), ("---", "---", "---", "---", "---")]
    summary = []
    for ticker, g in grp:
        n = len(g)
        total_pnl = g[pnl_col].sum()
        avg_ret = g[ret_col].mean() if ret_col else float("nan")
        win_rate = (g[ret_col] > 0).mean() if ret_col else float("nan")
        summary.append((ticker, n, total_pnl, avg_ret, win_rate))
    summary.sort(key=lambda x: x[2])  # sort by total P&L ascending
    for ticker, n, pnl, avg_ret, win_rate in summary:
        rows.append((str(ticker), str(n), f"${pnl:,.2f}", f"{avg_ret:.1%}" if not np.isnan(avg_ret) else "nan", f"{win_rate:.1%}" if not np.isnan(win_rate) else "nan"))
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def top_trades_report(trades: pd.DataFrame, n: int = 20) -> str:
    """Return top N trades by return as markdown."""
    if trades.empty:
        return "No trades."
    ret_col = "return_pct" if "return_pct" in trades.columns else None
    pnl_col = "pnl" if "pnl" in trades.columns else None
    ticker_col = "ticker" if "ticker" in trades.columns else "Column"

    if ret_col is None:
        return "No return column."

    top = trades.nlargest(n, ret_col)
    rows = [("Rank", "Ticker", "Entry", "Exit", "Return", "P&L", "Exit Reason"),
            ("---", "---", "---", "---", "---", "---", "---")]
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        ticker = str(row.get(ticker_col, ""))
        entry = str(row.get("entry_date", ""))[:10]
        exit_d = str(row.get("exit_date", ""))[:10]
        ret = row.get(ret_col, float("nan"))
        pnl = row.get(pnl_col, float("nan")) if pnl_col else float("nan")
        reason = str(row.get("exit_reason", ""))
        rows.append((
            str(rank), ticker, entry, exit_d,
            f"{ret:.1%}" if not np.isnan(ret) else "nan",
            f"${pnl:,.2f}" if not np.isnan(pnl) else "nan",
            reason,
        ))
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def run_v1_window(
    bars: dict[str, pd.DataFrame],
    spy_bars: pd.DataFrame,
    cfg: TradingConfig,
    tickers: list[str],
    start: str,
    end: str,
) -> dict:
    """Run one walk-forward window (IS or OOS) with 300-day warmup."""
    from dateutil.relativedelta import relativedelta

    # Warmup prefix: 300 calendar days before window start for indicator warmup
    warmup_start = pd.Timestamp(start) - relativedelta(days=300)
    warmup_start_str = warmup_start.strftime("%Y-%m-%d")

    w_bars = slice_bars(bars, warmup_start_str, end)
    w_spy = spy_bars.loc[spy_bars.index >= pd.Timestamp(warmup_start_str).normalize()]

    if w_spy.empty:
        return _empty_metrics()

    for t in list(w_bars.keys()):
        w_bars[t] = _normalize_idx(w_bars[t])
    w_spy = _normalize_idx(w_spy)
    common_idx = w_spy.index

    close_df, open_df = build_price_matrices(w_bars, tickers, common_idx)
    if close_df.empty:
        return _empty_metrics()

    # Compute signals on extended (warmup) range
    entries = compute_entry_signals(w_bars, w_spy, cfg)
    exits_df = compute_exit_signals(w_bars, w_spy, cfg)

    entries = apply_price_filter(entries, close_df, cfg)
    entries = apply_position_cap(entries, cfg)

    # Trim to actual window (post-warmup)
    window_start_ts = pd.Timestamp(start).normalize()
    window_end_ts = pd.Timestamp(end).normalize()
    entries = entries.loc[(entries.index >= window_start_ts) & (entries.index <= window_end_ts)]
    exits_df = exits_df.reindex_like(entries).fillna(False)
    close_df = close_df.loc[(close_df.index >= window_start_ts) & (close_df.index <= window_end_ts)]
    open_df = open_df.loc[(open_df.index >= window_start_ts) & (open_df.index <= window_end_ts)]

    n_signals = int(entries.sum().sum())
    if n_signals == 0:
        return _empty_metrics()

    pf = vbt.Portfolio.from_signals(
        close=close_df,
        open=open_df,
        entries=entries,
        exits=exits_df,
        price=open_df,
        size=cfg.target_position_pct,
        size_type="percent",
        sl_stop=cfg.hard_stop_pct,
        fees=cfg.commission,
        slippage=cfg.slippage_rate,
        init_cash=cfg.paper_capital,
        group_by=True,
        cash_sharing=True,
    )

    equity = pf.value()
    if isinstance(equity, pd.DataFrame):
        equity = equity.iloc[:, 0]
    daily_ret = equity.pct_change().dropna()

    try:
        trades_df = pf.trades.records_readable
        trade_stats = compute_trade_stats(trades_df)
    except Exception:
        trade_stats = {"trade_count": 0, "win_rate": float("nan"),
                       "avg_win": float("nan"), "avg_loss": float("nan"),
                       "avg_hold_days": float("nan")}

    return {
        "total_return": total_return(equity),
        "cagr": annualised_return(equity),
        "sharpe": sharpe_ratio(daily_ret),
        "sortino": sortino_ratio(daily_ret),
        "max_drawdown": max_drawdown(equity),
        "equity": equity,
        **trade_stats,
    }


def _empty_metrics() -> dict:
    return {
        "total_return": 0.0, "cagr": 0.0, "sharpe": float("nan"),
        "sortino": float("nan"), "max_drawdown": 0.0,
        "trade_count": 0, "win_rate": float("nan"),
        "avg_win": float("nan"), "avg_loss": float("nan"),
        "avg_hold_days": float("nan"), "equity": pd.Series(dtype=float),
    }


def write_report(
    cfg: TradingConfig,
    strategy_metrics: dict,
    spy_metrics: dict,
    wf_results: list[dict],
    trades: pd.DataFrame,
    output_path: Path,
) -> None:
    header = f"""\
# Strategy v1.0 Backtest Report
**Universe:** 84 S&P 500 large-caps (same as prior phases)
**Window:** {cfg.backtest_start} -> {cfg.backtest_end} | **Capital:** ${cfg.paper_capital:,.0f}
**Entry:** Price > {cfg.breakout_lookback}d high, Price > {cfg.trend_sma_short}d SMA, Golden Cross (50>200d), RS vs SPY >= +{cfg.relative_strength_min_outperformance:.0%} over {cfg.relative_strength_lookback}d, SPY > 200d SMA
**Exit:** Price < {cfg.trailing_sma_days}d SMA (PRIMARY trailing stop), -{cfg.hard_stop_pct:.0%} hard stop, regime stop
**Costs:** $0 commission + 5 bps slippage/side | Volume filter: OFF | Regime gate: ON

---

## Full-Window Metrics

{format_metrics_table(strategy_metrics, spy_metrics)}

---

## Exit Rule Breakdown

{exit_reason_breakdown(trades)}

---

## Per-Ticker Summary

{per_ticker_summary(trades)}

---

## Walk-Forward Results (IS = 2yr / OOS = 6mo / Step = 3mo)

{format_wf_table(wf_results)}

---

## Honest Assessment

"""
    trade_count = strategy_metrics.get("trade_count", 0)
    sharpe = strategy_metrics.get("sharpe", float("nan"))
    total_ret = strategy_metrics.get("total_return", float("nan"))
    mdd = strategy_metrics.get("max_drawdown", float("nan"))
    spy_ret = spy_metrics.get("total_return", float("nan"))
    win_rate = strategy_metrics.get("win_rate", float("nan"))
    avg_win = strategy_metrics.get("avg_win", float("nan"))
    avg_loss = strategy_metrics.get("avg_loss", float("nan"))
    oos_windows = [r["OOS"] for r in wf_results if "OOS" in r and not np.isnan(r["OOS"].get("sharpe", float("nan")))]
    oos_positive = sum(1 for r in oos_windows if r.get("sharpe", float("nan")) > 0)
    oos_total = len(oos_windows)

    assessment_lines = []
    if trade_count < 100:
        assessment_lines.append(f"- **TRADE COUNT {trade_count} < 100 minimum.** Statistics are unreliable with fewer than 100 trades.")
    else:
        assessment_lines.append(f"- Trade count {trade_count} >= 100 minimum. Statistics are meaningful.")
    if np.isnan(sharpe):
        assessment_lines.append("- **Sharpe ratio is NaN** — no trades or flat equity.")
    elif sharpe < 0:
        assessment_lines.append(f"- **Sharpe {sharpe:.2f} — negative.** Risk-adjusted return is worse than cash.")
    elif sharpe < 0.5:
        assessment_lines.append(f"- Sharpe {sharpe:.2f} — below 0.50 viable threshold.")
    else:
        assessment_lines.append(f"- Sharpe {sharpe:.2f} — above 0.50 viable threshold.")
    if not np.isnan(mdd):
        if mdd < -0.20:
            assessment_lines.append(f"- **Max drawdown {mdd:.1%} exceeds -20% spec limit.**")
        else:
            assessment_lines.append(f"- Max drawdown {mdd:.1%} is within the -20% spec limit.")
    if not np.isnan(total_ret) and not np.isnan(spy_ret):
        diff = total_ret - spy_ret
        assessment_lines.append(f"- Absolute return {total_ret:.1%} vs SPY {spy_ret:.1%} ({diff:+.1%}).")
    if not np.isnan(win_rate):
        assessment_lines.append(f"- Win rate {win_rate:.1%}, avg win {avg_win:.1%}, avg loss {avg_loss:.1%}." if not np.isnan(avg_win) else f"- Win rate {win_rate:.1%}.")
    if oos_total > 0:
        pct = oos_positive / oos_total
        if pct >= 0.60:
            assessment_lines.append(f"- Walk-forward OOS: {oos_positive}/{oos_total} windows positive Sharpe ({pct:.0%}). Edge appears consistent OOS.")
        else:
            assessment_lines.append(f"- Walk-forward OOS: {oos_positive}/{oos_total} windows positive Sharpe ({pct:.0%}). **Edge is inconsistent OOS.**")

    assessment = "\n".join(assessment_lines)

    paper_ready = (
        trade_count >= 100
        and not np.isnan(sharpe) and sharpe >= 0.50
        and not np.isnan(mdd) and mdd > -0.20
        and oos_total > 0 and (oos_positive / oos_total) >= 0.60
    )
    recommendation = (
        "**Ready for paper trading consideration.** OOS walk-forward shows consistent positive Sharpe. "
        "Run minimum 3 months paper before any live capital."
        if paper_ready else
        "**Not yet ready for paper trading.** See assessment above for blockers."
    )

    footer = f"""
---

## Paper-Trading Recommendation

{recommendation}

---

## v1.0 vs v0.3 Design Changes (Summary)

| Parameter | v0.3 | v1.0 | Rationale |
|-----------|------|------|-----------|
| Strategy family | Mean reversion dip-buy | Momentum / trend-following | Mean reversion had negative expectancy in 2021-2026 bull run |
| Universe | 11 sector ETFs | 84 S&P 500 large-caps | More signals; stock-level momentum cleaner than sector ETFs |
| Entry trigger | RSI < 35 oversold | Price > 126d high (breakout) | Buy strength, not weakness |
| Trend filter | Price > 200d SMA | Golden cross (50d > 200d) + Price > 200d + Price > 50d | Multi-confirm uptrend |
| RS filter | None | 63d return outperforms SPY by >= 5pp | Only enter strongest relative performers |
| Primary exit | RSI > 50 crossover | Price < 50d SMA (trailing) | Ride the trend; exit only when trend breaks |
| Profit target | +8% | None | Momentum runs can exceed 8%; cutting winners was a bug |
| Stop loss | -5% | -8% hard stop | Trend strategies need wider stops |
| Time stop | 15 days | None | Good trends last months, not weeks |

---

## Known Limitations

- **Survivorship bias:** Universe uses current S&P 500 constituents. Stocks that were removed (bankruptcy, acquisition, de-listing) between 2021-2026 are excluded. Momentum strategies are particularly vulnerable — failed breakouts in delisted stocks are invisible.
- **Lookahead in universe:** 2021 entry uses stocks that are in the S&P 500 in 2026.
- **Stop execution:** sl_stop applied at bar close by vectorbt. Real gaps can blow through hard stops.
- **2021-2026 window:** Predominantly bullish with two sharp corrections (2022 bear, 2025 correction). Momentum strategies should outperform in this regime.
- **Golden cross lag:** 50d/200d golden cross is a lagging signal. Many large moves have already started when it fires.

---
*Generated by v1.0 backtest harness*
"""

    output_path.write_text(header + assessment + footer, encoding="utf-8")
    logger.info("Report written to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strategy v1.0 momentum backtest")
    parser.add_argument("--refresh", action="store_true", help="Force re-download data")
    parser.add_argument("--tickers", nargs="+", default=None, help="Override ticker list")
    args = parser.parse_args()

    cfg = make_v1_config()
    tickers = args.tickers if args.tickers else BACKTEST_UNIVERSE

    logger.info("=== Strategy v1.0 Momentum Backtest ===")
    logger.info("Universe: %d tickers", len(tickers))
    logger.info("Window: %s -> %s", cfg.backtest_start, cfg.backtest_end)

    # Download data — use extended start for warmup (200d SMA needs ~10 months)
    warmup_start = pd.Timestamp(cfg.backtest_start) - pd.Timedelta(days=400)
    dl_start = warmup_start.strftime("%Y-%m-%d")
    logger.info("Downloading data from %s (includes warmup) …", dl_start)

    bars = download_bars(
        tickers=tickers,
        start=dl_start,
        end=cfg.backtest_end,
        cache_dir=cfg.data_dir / "bars",
        force_refresh=args.refresh,
    )
    spy_bars = load_spy(
        start=dl_start,
        end=cfg.backtest_end,
        cache_dir=cfg.data_dir / "bars",
        force_refresh=args.refresh,
    )

    # Normalize indexes
    for t in list(bars.keys()):
        bars[t] = _normalize_idx(bars[t])
    spy_bars = _normalize_idx(spy_bars)

    available_tickers = [t for t in tickers if t in bars]
    logger.info("Available tickers: %d / %d", len(available_tickers), len(tickers))

    common_idx = spy_bars.index
    close_df, open_df = build_price_matrices(bars, available_tickers, common_idx)

    # Align to backtest window
    start_ts = pd.Timestamp(cfg.backtest_start).normalize()
    end_ts = pd.Timestamp(cfg.backtest_end).normalize()

    # Compute signals on full range (includes warmup prefix for indicator warmup)
    logger.info("Computing entry signals …")
    entries_full = compute_entry_signals(bars, spy_bars, cfg)
    logger.info("Computing exit signals …")
    exits_full = compute_exit_signals(bars, spy_bars, cfg)

    entries_full = apply_price_filter(entries_full, close_df, cfg)
    entries_full = apply_position_cap(entries_full, cfg)

    # Trim to backtest window
    entries = entries_full.loc[
        (entries_full.index >= start_ts) & (entries_full.index <= end_ts)
    ]
    exits_df = exits_full.reindex_like(entries).fillna(False)
    close_win = close_df.loc[(close_df.index >= start_ts) & (close_df.index <= end_ts)]
    open_win = open_df.loc[(open_df.index >= start_ts) & (open_df.index <= end_ts)]

    n_signals = int(entries.sum().sum())
    logger.info("Total entry signals in backtest window: %d", n_signals)

    if n_signals == 0:
        logger.error("Zero entry signals — check data and parameters.")
        sys.exit(1)

    # Run vectorbt backtest — use hard_stop_pct as sl_stop; no tp_stop
    logger.info("Running vectorbt portfolio …")
    pf = vbt.Portfolio.from_signals(
        close=close_win,
        open=open_win,
        entries=entries,
        exits=exits_df,
        price=open_win,
        size=cfg.target_position_pct,
        size_type="percent",
        sl_stop=cfg.hard_stop_pct,
        fees=cfg.commission,
        slippage=cfg.slippage_rate,
        init_cash=cfg.paper_capital,
        group_by=True,
        cash_sharing=True,
    )

    equity = pf.value()
    if isinstance(equity, pd.DataFrame):
        equity = equity.iloc[:, 0]
    daily_ret = equity.pct_change().dropna()

    logger.info("Extracting trade stats …")
    try:
        trades_readable = pf.trades.records_readable
        trade_stats = compute_trade_stats(trades_readable)
    except Exception as exc:
        logger.warning("Trade stats failed: %s", exc)
        trade_stats = {"trade_count": 0, "win_rate": float("nan"),
                       "avg_win": float("nan"), "avg_loss": float("nan"),
                       "avg_hold_days": float("nan")}

    strategy_metrics = {
        "total_return": total_return(equity),
        "cagr": annualised_return(equity),
        "sharpe": sharpe_ratio(daily_ret),
        "sortino": sortino_ratio(daily_ret),
        "max_drawdown": max_drawdown(equity),
        **trade_stats,
    }
    logger.info("Strategy: total_return=%.1f%%, sharpe=%.2f, trades=%d",
                strategy_metrics["total_return"] * 100,
                strategy_metrics["sharpe"] if not np.isnan(strategy_metrics["sharpe"]) else -999,
                strategy_metrics["trade_count"])

    spy_metrics = spy_benchmark(spy_bars, cfg.backtest_start, cfg.backtest_end, cfg.paper_capital)
    logger.info("SPY B&H: total_return=%.1f%%, sharpe=%.2f",
                spy_metrics["total_return"] * 100, spy_metrics["sharpe"])

    # Build trade log
    logger.info("Building trade log …")
    from trader.signals.momentum import compute_trailing_sma_series
    trailing_sma_cache = compute_trailing_sma_series(bars, common_idx, cfg.trailing_sma_days)
    trade_log = build_trade_log(pf, bars, spy_bars, cfg, trailing_sma_cache)

    # Walk-forward
    logger.info("Running walk-forward analysis …")
    wf_windows = generate_windows(cfg)
    wf_results = []
    for window in wf_windows:
        is_result = run_v1_window(bars, spy_bars, cfg, available_tickers, window.is_start, window.is_end)
        oos_result = run_v1_window(bars, spy_bars, cfg, available_tickers, window.oos_start, window.oos_end)
        wf_results.append({
            "window_idx": window.window_idx,
            "IS_start": window.is_start,
            "IS_end": window.is_end,
            "OOS_start": window.oos_start,
            "OOS_end": window.oos_end,
            "IS": is_result,
            "OOS": oos_result,
        })
        logger.info(
            "[WF %d] IS sharpe=%.2f trades=%d | OOS sharpe=%.2f trades=%d",
            window.window_idx,
            is_result["sharpe"] if not np.isnan(is_result["sharpe"]) else float("nan"),
            is_result["trade_count"],
            oos_result["sharpe"] if not np.isnan(oos_result["sharpe"]) else float("nan"),
            oos_result["trade_count"],
        )

    # Outputs
    out_dir = cfg.outputs_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Equity curve
    logger.info("Saving equity curve …")
    plot_equity_curve(
        strategy_equity=equity,
        spy_equity=spy_metrics["equity"],
        output_path=out_dir / "v1_equity_curve.png",
        title="Strategy v1.0 (Momentum) vs SPY Buy & Hold",
    )

    # Trade log CSV
    if not trade_log.empty:
        trade_log.to_csv(out_dir / "v1_trade_log.csv", index=False)
        logger.info("Trade log: %d trades -> %s", len(trade_log), out_dir / "v1_trade_log.csv")
    else:
        logger.warning("Trade log is empty — no CSV written.")

    # Top trades report
    top_md = top_trades_report(trade_log, n=20)
    (out_dir / "v1_top_trades.md").write_text(
        "# Strategy v1.0 — Top 20 Trades by Return\n\n" + top_md + "\n",
        encoding="utf-8",
    )
    logger.info("Top trades written.")

    # Full report
    write_report(
        cfg=cfg,
        strategy_metrics=strategy_metrics,
        spy_metrics=spy_metrics,
        wf_results=wf_results,
        trades=trade_log,
        output_path=out_dir / "v1_report.md",
    )

    logger.info("=== v1.0 complete ===")
    logger.info("Outputs: v1_report.md, v1_equity_curve.png, v1_trade_log.csv, v1_top_trades.md")


if __name__ == "__main__":
    main()
