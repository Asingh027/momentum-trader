#!/usr/bin/env python
"""Trailing SMA period sweep — fast+tight baseline (breakout=63d).

Sweeps TRAILING_SMA_DAYS in [5, 10, 15, 20, 30, 50] while holding all other
parameters at the fast+tight config (breakout_lookback=63, Aggressive sizing:
20%/pos, 4 max, 20% floor, $25K, 5 bps slippage).

Output: outputs/trailing_sma_sweep.md

Usage:
    uv run python scripts/run_trailing_sma_sweep.py
    uv run python scripts/run_trailing_sma_sweep.py --refresh
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
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    spy_benchmark,
    total_return,
)
from trader.config import TradingConfig
from trader.data.bars import download_bars, load_spy
from trader.risk.filters import apply_position_cap, apply_price_filter
from trader.signals.momentum import compute_entry_signals, compute_exit_signals

BASELINE_UNIVERSE = [
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

V1_OVERRIDES = dict(
    breakout_lookback=126,
    trend_sma_short=50,
    relative_strength_lookback=63,
    relative_strength_min_outperformance=0.05,
    trailing_sma_days=50,
    hard_stop_pct=0.08,
    profit_target_pct=0.0,
    time_stop_days=9999,
    stop_loss_pct=0.0,
    use_volume_filter=False,
    use_regime_gate=True,
    earnings_buffer_days=3,
    min_price=20.0,
    max_price=500.0,
)

AGGRESSIVE_OVERRIDES = dict(
    target_position_pct=0.20,
    max_positions=4,
    cash_floor_pct=0.20,
)


def _normalize_idx(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = df.index.normalize()
    return df


def _empty() -> dict:
    return {
        "total_return": float("nan"), "cagr": float("nan"),
        "sharpe": float("nan"), "sortino": float("nan"),
        "max_drawdown": float("nan"), "trade_count": 0,
        "win_rate": float("nan"), "avg_win": float("nan"),
        "avg_loss": float("nan"), "avg_hold_days": float("nan"),
        "equity": pd.Series(dtype=float),
    }


def run_variant(
    bars: dict[str, pd.DataFrame],
    spy_bars: pd.DataFrame,
    cfg: TradingConfig,
    tickers: list[str],
    label: str = "",
) -> dict:
    available = [t for t in tickers if t in bars]
    if not available:
        return _empty()

    common_idx = spy_bars.index
    closes = {t: bars[t]["Close"].reindex(common_idx) for t in available}
    opens  = {t: bars[t]["Open"].reindex(common_idx) for t in available}
    close_df = pd.DataFrame(closes, index=common_idx)
    open_df  = pd.DataFrame(opens,  index=common_idx)

    start_ts = pd.Timestamp(cfg.backtest_start).normalize()
    end_ts   = pd.Timestamp(cfg.backtest_end).normalize()

    logger.info("%s: computing signals (%d tickers) …", label, len(available))
    entries_full = compute_entry_signals(bars, spy_bars, cfg)
    exits_full   = compute_exit_signals(bars, spy_bars, cfg)
    entries_full = apply_price_filter(entries_full, close_df, cfg)
    entries_full = apply_position_cap(entries_full, cfg)

    entries  = entries_full.loc[(entries_full.index >= start_ts) & (entries_full.index <= end_ts)]
    exits_df = exits_full.reindex_like(entries).fillna(False)
    close_win = close_df.loc[(close_df.index >= start_ts) & (close_df.index <= end_ts)]
    open_win  = open_df.loc[(open_df.index >= start_ts) & (open_df.index <= end_ts)]

    n_signals = int(entries.sum().sum())
    logger.info("%s: %d entry signals in window", label, n_signals)
    if n_signals == 0:
        logger.warning("%s: zero signals — skipping", label)
        return _empty()

    pf = vbt.Portfolio.from_signals(
        close=close_win, open=open_win,
        entries=entries, exits=exits_df,
        price=open_win,
        size=cfg.target_position_pct, size_type="percent",
        sl_stop=cfg.hard_stop_pct,
        fees=cfg.commission, slippage=cfg.slippage_rate,
        init_cash=cfg.paper_capital,
        group_by=True, cash_sharing=True,
    )

    equity = pf.value()
    if isinstance(equity, pd.DataFrame):
        equity = equity.iloc[:, 0]
    daily_ret = equity.pct_change().dropna()

    try:
        trade_stats = compute_trade_stats(pf.trades.records_readable)
    except Exception:
        trade_stats = {"trade_count": 0, "win_rate": float("nan"),
                       "avg_win": float("nan"), "avg_loss": float("nan"),
                       "avg_hold_days": float("nan")}

    m = {
        "total_return": total_return(equity),
        "cagr": annualised_return(equity),
        "sharpe": sharpe_ratio(daily_ret),
        "sortino": sortino_ratio(daily_ret),
        "max_drawdown": max_drawdown(equity),
        **trade_stats,
        "equity": equity,
    }
    logger.info(
        "%s: return=%.1f%% CAGR=%.1f%% sharpe=%.2f mdd=%.1f%% trades=%d avghold=%.1fd",
        label,
        m["total_return"] * 100, m["cagr"] * 100,
        m["sharpe"] if not np.isnan(m["sharpe"]) else -99,
        m["max_drawdown"] * 100, m["trade_count"],
        m.get("avg_hold_days") or 0,
    )
    return m

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SWEEP_VALUES = [5, 10, 15, 20, 30, 50]


def make_fast_tight(trailing_sma_days: int) -> TradingConfig:
    """Fast+tight baseline with a specific trailing SMA period."""
    cfg = TradingConfig()
    cfg = dataclasses.replace(cfg, **V1_OVERRIDES)
    cfg = dataclasses.replace(cfg, **AGGRESSIVE_OVERRIDES)
    cfg = dataclasses.replace(cfg, breakout_lookback=63, trailing_sma_days=trailing_sma_days)
    return cfg


def _fmt(val, pct=False, dec=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "n/a"
    if pct:
        return f"{val:.1%}"
    if dec == 0:
        return f"{int(val)}"
    return f"{val:.{dec}f}"


def write_report(
    results: list[tuple[int, dict]],
    spy_metrics: dict,
    cfg: TradingConfig,
    out_path: Path,
) -> None:
    header = (
        "| SMA Days | Sharpe | CAGR | Total Return | MaxDD "
        "| WinRate | AvgWin | AvgLoss | Trades | AvgHold |"
    )
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    rows = [header, sep]

    for sma, m in results:
        marker = " ◀ current" if sma == 20 else ""
        rows.append(
            f"| **{sma}d**{marker} "
            f"| {_fmt(m['sharpe'])} "
            f"| {_fmt(m['cagr'], pct=True)} "
            f"| {_fmt(m['total_return'], pct=True)} "
            f"| {_fmt(m['max_drawdown'], pct=True)} "
            f"| {_fmt(m['win_rate'], pct=True)} "
            f"| {_fmt(m['avg_win'], pct=True)} "
            f"| {_fmt(m['avg_loss'], pct=True)} "
            f"| {_fmt(m['trade_count'], dec=0)} "
            f"| {_fmt(m['avg_hold_days'])}d |"
        )

    rows.append(
        f"| SPY B&H "
        f"| {_fmt(spy_metrics['sharpe'])} "
        f"| {_fmt(spy_metrics['cagr'], pct=True)} "
        f"| {_fmt(spy_metrics['total_return'], pct=True)} "
        f"| {_fmt(spy_metrics['max_drawdown'], pct=True)} "
        f"| — | — | — | — | — |"
    )

    table = "\n".join(rows)

    # Find best and assess vs SMA=20 baseline
    baseline_m = dict(next(m for s, m in results if s == 20))
    valid = [(s, m) for s, m in results if not np.isnan(m["sharpe"])]
    best_sma, best_m = max(valid, key=lambda x: x[1]["sharpe"])

    deltas = []
    for sma, m in results:
        if sma == 20 or np.isnan(m["sharpe"]):
            continue
        ds = m["sharpe"] - baseline_m["sharpe"]
        dc = m["cagr"] - baseline_m["cagr"]
        dd = m["max_drawdown"] - baseline_m["max_drawdown"]
        dt = m["trade_count"] - baseline_m["trade_count"]
        dh = (m.get("avg_hold_days") or 0) - (baseline_m.get("avg_hold_days") or 0)
        verdict = (
            "**noise** (|ΔSharpe| < 0.10)" if abs(ds) < 0.10
            else f"**better** (+{ds:.2f} Sharpe)" if ds > 0
            else f"**worse** ({ds:.2f} Sharpe)"
        )
        deltas.append(
            f"| {sma}d vs 20d | {ds:+.2f} | {dc:+.1%} | {dd:+.1%} "
            f"| {dt:+d} | {dh:+.1f}d | {verdict} |"
        )

    delta_header = "| Comparison | ΔSharpe | ΔCAGR | ΔMaxDD | ΔTrades | ΔAvgHold | Verdict |"
    delta_sep    = "| --- | --- | --- | --- | --- | --- | --- |"
    delta_table  = "\n".join([delta_header, delta_sep] + deltas)

    report = f"""\
# Trailing SMA Period Sweep — Fast+Tight Baseline

**Window:** {cfg.backtest_start} → {cfg.backtest_end} | **Capital:** ${cfg.paper_capital:,.0f}
**Fixed config:** breakout=63d, 20%/pos, 4 max positions, 20% cash floor, 5 bps slippage
**Universe:** 84 stocks | **Swept parameter:** trailing_sma_days ∈ {{{', '.join(str(s) for s, _ in results)}}}

---

## Results Table

{table}

---

## Delta vs SMA=20 (current config)

{delta_table}

---

## Analysis

**Best by Sharpe:** `{best_sma}d` ({best_m['sharpe']:.2f})

**Shorter windows (5–15d):**
- Tighter exit = more churn. Higher trade count, shorter avg hold, typically lower Sharpe because
  you get stopped out of good trends on normal intraday/weekly noise.
- Avg hold drops sharply; transaction costs compound.

**Longer windows (30–50d):**
- Looser exit = rides trends further but gives back more on reversals.
- Max drawdown tends to worsen; fewer trades but larger individual losses.
- SMA=50d is the original config — meaningful comparison point.

**20d (current):**
- Balances noise tolerance vs drawdown control in the 2021-2026 window.

---

## Deployment Note

This sweep covers a single in-sample window (2021-2026, predominantly bullish).
A Sharpe difference < 0.10 between variants is within simulation variance — do not
switch configs based on noise. If a variant shows > 0.15 Sharpe improvement, run
walk-forward OOS validation (`run_walkforward_variants.py`) before deploying.

---
*Generated by run_trailing_sma_sweep.py*
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Report → %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Force re-download data")
    args = parser.parse_args()

    cfg0 = make_fast_tight(20)
    warmup_start = (pd.Timestamp(cfg0.backtest_start) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")

    logger.info("=== Trailing SMA Sweep — fast+tight baseline (breakout=63d) ===")
    logger.info("Loading 84-stock data from cache …")

    bars = download_bars(
        tickers=BASELINE_UNIVERSE,
        start=warmup_start,
        end=cfg0.backtest_end,
        cache_dir=cfg0.data_dir / "bars",
        force_refresh=args.refresh,
    )
    spy_bars = load_spy(
        start=warmup_start,
        end=cfg0.backtest_end,
        cache_dir=cfg0.data_dir / "bars",
        force_refresh=args.refresh,
    )

    bars = {t: _normalize_idx(df) for t, df in bars.items()}
    spy_bars = _normalize_idx(spy_bars)

    if spy_bars.empty:
        logger.error("No SPY data — aborting")
        sys.exit(1)

    spy_metrics = spy_benchmark(spy_bars, cfg0.backtest_start, cfg0.backtest_end, cfg0.paper_capital)
    logger.info("SPY B&H: return=%.1f%% sharpe=%.2f",
                spy_metrics["total_return"] * 100, spy_metrics["sharpe"])

    results: list[tuple[int, dict]] = []
    for sma in SWEEP_VALUES:
        cfg = make_fast_tight(sma)
        label = f"TrailSMA={sma}d"
        logger.info("--- %s ---", label)
        m = run_variant(bars, spy_bars, cfg, BASELINE_UNIVERSE, label=label)
        results.append((sma, m))

    logger.info("\n=== SWEEP RESULTS ===")
    logger.info("%-12s  %7s  %7s  %8s  %6s  %6s  %7s",
                "SMA Days", "Sharpe", "CAGR", "MaxDD", "Trades", "WinRate", "AvgHold")
    logger.info("-" * 65)
    for sma, m in results:
        cur = " <-- current" if sma == 20 else ""
        logger.info(
            "%-12s  %7.2f  %7.1f%%  %8.1f%%  %6d  %6.1f%%  %7.1fd%s",
            f"{sma}d",
            m["sharpe"] if not np.isnan(m["sharpe"]) else 0,
            m["cagr"] * 100 if not np.isnan(m["cagr"]) else 0,
            m["max_drawdown"] * 100 if not np.isnan(m["max_drawdown"]) else 0,
            m["trade_count"],
            (m["win_rate"] or 0) * 100,
            m.get("avg_hold_days") or 0,
            cur,
        )

    out_path = cfg0.outputs_dir / "trailing_sma_sweep.md"
    write_report(results, spy_metrics, cfg0, out_path)
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
