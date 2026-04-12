#!/usr/bin/env python
"""Strategy v0.3 — Mean Reversion on SPDR Sector ETFs.

Changes from v0.2:
  - Universe: 11 SPDR sector ETFs (XLK XLF XLE XLV XLI XLP XLY XLU XLB XLRE XLC)
  - RSI threshold: 35 (loosened from 30)
  - No volume filter (dropped — sweep showed it kills signal flow without quality gain)
  - PRIMARY exit: RSI(14) crosses above 50 (mean-reversion confirmed)
  - Profit target: +8% (backup for outlier rallies; avg wins were ~4.5%)
  - Time stop: 15 days (loosened from 10)
  - Earnings gate: always-pass (ETFs have no earnings)
  - All other params unchanged from v0.2

Outputs:
  outputs/v03_report.md
  outputs/v03_equity_curve.png
  outputs/v03_trade_log.csv

Usage:
  uv run python scripts/run_v03.py
  uv run python scripts/run_v03.py --refresh
"""

from __future__ import annotations

import logging
import sys
import textwrap
import warnings
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import vectorbt as vbt

from trader.backtest.engine import add_time_stop_exits
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
from trader.data.bars import download_bars
from trader.reports.equity_curve import plot_equity_curve
from trader.risk.filters import apply_position_cap, apply_price_filter
from trader.signals.mean_reversion import (
    _rsi,
    _sma,
    compute_entry_signals,
    compute_exit_signals,
    compute_rsi_series,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── v0.3 universe ─────────────────────────────────────────────────────────────
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC"]

# ── v0.3 config overrides (all else from .env / defaults) ────────────────────
V03_OVERRIDES = dict(
    rsi_oversold=35.0,          # loosened from 30
    rsi_exit_level=50.0,        # PRIMARY exit: RSI crosses above 50
    profit_target_pct=0.08,     # +8% backup target (avg wins ~4.5%)
    stop_loss_pct=0.05,         # unchanged
    time_stop_days=15,          # loosened from 10 — ETFs need more room
    use_volume_filter=False,    # dropped — kills signal flow without quality gain
    use_regime_gate=True,       # kept — does its job (better Sharpe, lower DD)
)


def make_v03_config() -> TradingConfig:
    base = TradingConfig.from_env()
    return replace(base, **V03_OVERRIDES)


# ── Exit reason attribution ───────────────────────────────────────────────────
# vectorbt doesn't expose a named exit-type field in 0.28 trades records.
# We infer from: return magnitude (sl/tp) vs signal exits (rsi cross / regime / time).

_SL_TOL = 0.015   # tolerance band around stop-loss for attribution
_TP_TOL = 0.015   # tolerance band around profit-target


def attribute_exit_reason(
    trade_row: pd.Series,
    cfg: TradingConfig,
    spy_bars: pd.DataFrame,
    rsi_by_ticker: dict[str, pd.Series],
) -> str:
    """Infer exit reason for a single trade record."""
    ret = trade_row.get("Return", float("nan"))
    ticker = str(trade_row.get("Column", ""))
    # vbt 0.28 uses Timestamp columns; older used Index
    exit_date = trade_row.get("Exit Timestamp") or trade_row.get("Exit Index")
    entry_date = trade_row.get("Entry Timestamp") or trade_row.get("Entry Index")

    if pd.isna(ret):
        return "unknown"

    # Stop loss: return ≈ -stop_loss_pct (within tolerance, negative)
    if ret < 0 and abs(ret - (-cfg.stop_loss_pct)) <= _SL_TOL:
        return "stop_loss"

    # Profit target: return ≈ +profit_target_pct (within tolerance, positive)
    if ret > 0 and abs(ret - cfg.profit_target_pct) <= _TP_TOL:
        return "profit_target"

    # Signal exits — check conditions on exit date
    if exit_date is not None and not pd.isna(exit_date):
        exit_ts = pd.Timestamp(exit_date)

        # Regime stop: SPY below 200d SMA on exit date
        try:
            spy_close = spy_bars["Close"]
            spy_sma_val = _sma(spy_close, cfg.spy_trend_sma_period)
            spy_row = spy_close.get(exit_ts)
            sma_row = spy_sma_val.get(exit_ts)
            if spy_row is not None and sma_row is not None and not pd.isna(spy_row) and not pd.isna(sma_row):
                if spy_row < sma_row:
                    return "regime_stop"
        except Exception:
            pass

        # RSI crossover: RSI crossed above rsi_exit_level on exit date
        if cfg.rsi_exit_level > 0 and ticker in rsi_by_ticker:
            try:
                rsi_series = rsi_by_ticker[ticker]
                rsi_exit = rsi_series.get(exit_ts)
                rsi_prev = None
                idx_pos = rsi_series.index.get_loc(exit_ts)
                if idx_pos > 0:
                    rsi_prev = rsi_series.iloc[idx_pos - 1]
                if (rsi_exit is not None and not pd.isna(rsi_exit)
                        and rsi_prev is not None and not pd.isna(rsi_prev)
                        and rsi_exit >= cfg.rsi_exit_level
                        and rsi_prev < cfg.rsi_exit_level):
                    return "rsi_crossover"
            except Exception:
                pass

        # Time stop: duration ≈ time_stop_days bars
        if entry_date is not None and not pd.isna(entry_date):
            entry_ts = pd.Timestamp(entry_date)
            try:
                # Count trading days between entry and exit
                if exit_ts in spy_bars.index and entry_ts in spy_bars.index:
                    entry_pos = spy_bars.index.get_loc(entry_ts)
                    exit_pos = spy_bars.index.get_loc(exit_ts)
                    bars_held = exit_pos - entry_pos
                    if abs(bars_held - cfg.time_stop_days) <= 2:
                        return "time_stop"
            except Exception:
                pass

    # Fallback: classify by return direction
    return "signal_exit_other"


def run_v03_backtest(
    bars: dict[str, pd.DataFrame],
    spy_bars: pd.DataFrame,
    cfg: TradingConfig,
    tickers: list[str],
) -> dict:
    """Full v0.3 backtest. Returns metrics + portfolio + rsi_series for trade log."""
    close_dict = {t: bars[t]["Close"] for t in tickers if t in bars}
    open_dict  = {t: bars[t]["Open"]  for t in tickers if t in bars}
    volume_dict = {t: bars[t]["Volume"] for t in tickers if t in bars}

    close = pd.DataFrame(close_dict)
    open_prices = pd.DataFrame(open_dict)
    volume = pd.DataFrame(volume_dict)

    # Normalise all indices to tz-naive date-only (yfinance can return tz-aware)
    def _normalize_idx(df: pd.DataFrame) -> pd.DataFrame:
        idx = pd.to_datetime(df.index).normalize()
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_localize(None)
        df = df.copy()
        df.index = idx
        return df

    close = _normalize_idx(close)
    open_prices = _normalize_idx(open_prices)
    volume = _normalize_idx(volume)
    spy_bars = _normalize_idx(spy_bars)

    # Use intersection of ETF dates and SPY dates as the common index
    common_idx = close.index.intersection(spy_bars.index)
    close = close.reindex(common_idx)
    open_prices = open_prices.reindex(common_idx).ffill(limit=3)
    volume = volume.reindex(common_idx).ffill(limit=3)
    spy_bars_aligned = spy_bars.reindex(common_idx).ffill(limit=3)

    bars_for_signals = {
        t: pd.DataFrame({"Close": close[t], "Volume": volume[t]}, index=common_idx)
        for t in close.columns
    }

    logger.info("Computing entry signals ...")
    entries = compute_entry_signals(bars=bars_for_signals, spy_bars=spy_bars_aligned, cfg=cfg)

    logger.info("Computing exit signals (RSI crossover + regime stop) ...")
    exits_df = compute_exit_signals(bars=bars_for_signals, spy_bars=spy_bars_aligned, cfg=cfg)

    # Align signals to common_idx (they use spy_bars_aligned.index which == common_idx)
    entries = entries.reindex(index=common_idx, columns=close.columns, fill_value=False)
    exits_df = exits_df.reindex(index=common_idx, columns=close.columns, fill_value=False)

    # No price filter needed for ETFs
    entries = apply_position_cap(entries, cfg)
    exits_df = add_time_stop_exits(entries, exits_df, cfg.time_stop_days)

    n_signals = int(entries.sum().sum())
    logger.info("Total entry signals: %d", n_signals)

    if n_signals == 0:
        logger.warning("Zero entry signals — strategy never fires")
        empty = pd.Series(dtype=float)
        return {"total_return": 0, "cagr": 0, "sharpe": float("nan"),
                "sortino": float("nan"), "max_drawdown": 0, "trade_count": 0,
                "win_rate": float("nan"), "avg_win": float("nan"),
                "avg_loss": float("nan"), "avg_hold_days": float("nan"),
                "equity": empty, "portfolio": None, "rsi_series": {}}

    pf = vbt.Portfolio.from_signals(
        close=close,
        open=open_prices,
        entries=entries,
        exits=exits_df,
        price=open_prices,
        size=cfg.target_position_pct,
        size_type="percent",
        sl_stop=cfg.stop_loss_pct,
        tp_stop=cfg.profit_target_pct,
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
    except Exception as exc:
        logger.warning("Trade stat extraction failed: %s", exc)
        trade_stats = {"trade_count": n_signals, "win_rate": float("nan"),
                       "avg_win": float("nan"), "avg_loss": float("nan"),
                       "avg_hold_days": float("nan")}

    # Pre-compute RSI for all tickers (needed for exit attribution)
    rsi_series = compute_rsi_series(bars_for_signals, common_idx, cfg.rsi_period)

    return {
        "total_return": total_return(equity),
        "cagr": annualised_return(equity),
        "sharpe": sharpe_ratio(daily_ret),
        "sortino": sortino_ratio(daily_ret),
        "max_drawdown": max_drawdown(equity),
        "equity": equity,
        "portfolio": pf,
        "rsi_series": rsi_series,
        **trade_stats,
    }


def build_trade_log(
    portfolio: vbt.Portfolio,
    cfg: TradingConfig,
    spy_bars: pd.DataFrame,
    rsi_series: dict[str, pd.Series],
) -> pd.DataFrame:
    """Build trade-level DataFrame with exit reason attribution."""
    try:
        trades_df = portfolio.trades.records_readable.copy()
    except Exception as exc:
        logger.warning("Could not extract trades: %s", exc)
        return pd.DataFrame()

    if trades_df.empty:
        return trades_df

    # Rename for readability — column names vary by vbt version; handle both
    col_map = {
        "Column": "ticker",
        "Entry Index": "entry_date",       # older vbt
        "Exit Index": "exit_date",
        "Entry Timestamp": "entry_date",   # vbt 0.28
        "Exit Timestamp": "exit_date",
        "Avg Entry Price": "entry_price",
        "Avg Exit Price": "exit_price",
        "Entry Price": "entry_price",
        "Exit Price": "exit_price",
        "Size": "shares",
        "P&L": "pnl",
        "PnL": "pnl",
        "Return": "return_pct",
    }
    trades_df = trades_df.rename(columns={k: v for k, v in col_map.items() if k in trades_df.columns})

    # Attribute exit reason
    raw = portfolio.trades.records_readable  # use original column names for attribution
    exit_reasons = []
    for idx in range(len(raw)):
        row = raw.iloc[idx]
        reason = attribute_exit_reason(row, cfg, spy_bars, rsi_series)
        exit_reasons.append(reason)
    trades_df["exit_reason"] = exit_reasons

    # Round numeric columns
    for col in ["entry_price", "exit_price", "pnl", "return_pct"]:
        if col in trades_df.columns:
            trades_df[col] = trades_df[col].round(4)

    # Keep relevant columns
    keep = ["ticker", "entry_date", "exit_date", "entry_price", "exit_price",
            "shares", "pnl", "return_pct", "exit_reason"]
    trades_df = trades_df[[c for c in keep if c in trades_df.columns]]
    return trades_df.sort_values("entry_date").reset_index(drop=True)


def per_etf_summary(trade_log: pd.DataFrame) -> pd.DataFrame:
    """Summarise trade count and P&L by ticker."""
    if trade_log.empty or "ticker" not in trade_log.columns:
        return pd.DataFrame()
    g = trade_log.groupby("ticker").agg(
        trade_count=("pnl", "count"),
        total_pnl=("pnl", "sum"),
        avg_return=("return_pct", "mean"),
        win_rate=("return_pct", lambda x: (x > 0).mean()),
    ).round(4).sort_values("total_pnl", ascending=False)
    return g


def exit_reason_breakdown(trade_log: pd.DataFrame) -> dict[str, float]:
    """Fraction of trades by exit reason."""
    if trade_log.empty or "exit_reason" not in trade_log.columns:
        return {}
    counts = trade_log["exit_reason"].value_counts()
    total = len(trade_log)
    return {reason: count / total for reason, count in counts.items()}


def run_wf_window_v03(
    bars: dict[str, pd.DataFrame],
    spy_bars: pd.DataFrame,
    cfg: TradingConfig,
    window,
) -> dict:
    """Run one walk-forward window with 200-bar warmup for signal indicators."""
    import math
    warmup_days = int(cfg.trend_sma_period * 1.5)
    result = {
        "window_idx": window.window_idx,
        "IS_start": window.is_start,
        "IS_end": window.is_end,
        "OOS_start": window.oos_start,
        "OOS_end": window.oos_end,
    }

    for phase, start, end in [("IS", window.is_start, window.is_end),
                               ("OOS", window.oos_start, window.oos_end)]:
        warmup_start = (pd.Timestamp(start) - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")
        sliced = slice_bars(bars, warmup_start, end)
        spy_sliced = spy_bars.loc[warmup_start:end]
        tickers = list(sliced.keys())

        if not tickers or spy_sliced.empty:
            result[phase] = {"total_return": float("nan"), "cagr": float("nan"),
                             "sharpe": float("nan"), "sortino": float("nan"),
                             "max_drawdown": float("nan"), "trade_count": 0}
            continue

        try:
            m = run_v03_backtest(sliced, spy_sliced, cfg, tickers)
            result[phase] = m
        except Exception as exc:
            logger.error("WF window %d %s failed: %s", window.window_idx, phase, exc)
            result[phase] = {"total_return": float("nan"), "cagr": float("nan"),
                             "sharpe": float("nan"), "sortino": float("nan"),
                             "max_drawdown": float("nan"), "trade_count": 0}

    return result


def write_report(
    cfg: TradingConfig,
    strategy_metrics: dict,
    spy_metrics: dict,
    wf_results: list[dict],
    trade_log: pd.DataFrame,
    output_path: Path,
) -> None:
    exit_breakdown = exit_reason_breakdown(trade_log)
    etf_summary = per_etf_summary(trade_log)

    # Format exit breakdown
    exit_lines = []
    for reason, frac in sorted(exit_breakdown.items(), key=lambda x: -x[1]):
        count = int(round(frac * strategy_metrics.get("trade_count", 0)))
        exit_lines.append(f"| {reason} | {frac:.1%} | {count} |")
    exit_table = (
        "| Exit Reason | % of Trades | Count |\n"
        "|-------------|-------------|-------|\n"
        + "\n".join(exit_lines)
    ) if exit_lines else "_No trades._"

    # Format per-ETF summary
    if not etf_summary.empty:
        etf_lines = []
        for ticker, row in etf_summary.iterrows():
            etf_lines.append(
                f"| {ticker} | {int(row['trade_count'])} | ${row['total_pnl']:.2f} "
                f"| {row['avg_return']:.1%} | {row['win_rate']:.1%} |"
            )
        etf_table = (
            "| Ticker | Trades | Total P&L | Avg Return | Win% |\n"
            "|--------|--------|-----------|------------|------|\n"
            + "\n".join(etf_lines)
        )
    else:
        etf_table = "_No trade data._"

    # Honest assessment
    sharpe = strategy_metrics.get("sharpe", float("nan"))
    spy_sharpe = spy_metrics.get("sharpe", float("nan"))
    trade_count = strategy_metrics.get("trade_count", 0)
    total_ret = strategy_metrics.get("total_return", float("nan"))
    spy_total = spy_metrics.get("total_return", float("nan"))
    mdd = strategy_metrics.get("max_drawdown", float("nan"))
    win_rate = strategy_metrics.get("win_rate", float("nan"))
    avg_win = strategy_metrics.get("avg_win", float("nan"))
    avg_loss = strategy_metrics.get("avg_loss", float("nan"))

    assessment = []
    if trade_count < 100:
        assessment.append(
            f"**TRADE COUNT {trade_count} < 100 minimum.** Statistics are unreliable. "
            "Only 11 ETFs in the universe — this is a signal-frequency problem inherent to the universe size."
        )
    else:
        assessment.append(f"Trade count {trade_count} clears the 100-trade statistical floor.")

    if not np.isnan(sharpe) and not np.isnan(spy_sharpe):
        if sharpe > spy_sharpe:
            assessment.append(
                f"**Sharpe {sharpe:.2f} > SPY {spy_sharpe:.2f}** — outperforms on risk-adjusted basis."
            )
        elif sharpe > 0:
            assessment.append(
                f"Sharpe {sharpe:.2f} is positive but below SPY's {spy_sharpe:.2f}. "
                "Positive risk-adjusted return but doesn't justify the active risk."
            )
        else:
            assessment.append(
                f"**Sharpe {sharpe:.2f} — negative.** Risk-adjusted return is worse than cash."
            )

    if not np.isnan(mdd):
        if abs(mdd) > 0.20:
            assessment.append(
                f"**Max drawdown {mdd:.1%} exceeds the -20% spec limit.** Risk controls insufficient."
            )
        else:
            assessment.append(f"Max drawdown {mdd:.1%} is within the -20% spec limit.")

    if not np.isnan(total_ret) and not np.isnan(spy_total):
        gap = total_ret - spy_total
        if gap < -0.10:
            assessment.append(
                f"**Absolute return {total_ret:.1%} vs SPY {spy_total:.1%} ({gap:+.1%}).** "
                "Significant underperformance on raw return."
            )
        elif total_ret > spy_total:
            assessment.append(
                f"Absolute return {total_ret:.1%} beats SPY {spy_total:.1%} ({gap:+.1%})."
            )

    # RSI crossover exit effectiveness
    rsi_cross_pct = exit_breakdown.get("rsi_crossover", 0.0)
    if rsi_cross_pct > 0:
        assessment.append(
            f"RSI crossover exit fired on {rsi_cross_pct:.1%} of trades — "
            + ("it is the dominant exit mechanism, confirming the redesign is working as intended."
               if rsi_cross_pct > 0.40
               else "other exits (stops/time) are dominating. The RSI crossover rarely confirms recovery before another exit fires.")
        )

    # OOS walk-forward consistency
    oos_sharpes = [r.get("OOS", {}).get("sharpe", float("nan")) for r in wf_results]
    valid_oos = [s for s in oos_sharpes if not np.isnan(s) and np.isfinite(s)]
    if valid_oos:
        positive = sum(1 for s in valid_oos if s > 0)
        assessment.append(
            f"Walk-forward OOS: {positive}/{len(valid_oos)} windows positive Sharpe. "
            + ("Edge appears consistent across market regimes." if positive / len(valid_oos) >= 0.6
               else "**Edge is inconsistent OOS — likely regime-dependent or insufficient sample size.**")
        )

    # v0.4 correlation flag
    assessment.append(
        "**v0.4 note (not implemented):** Sector ETFs are correlated. "
        "Simultaneous XLK + XLC entries double tech/comms exposure beyond the 20%/position limit. "
        "A correlation cap or sector-level position limit should be added before paper trading."
    )

    verdict_para = "\n".join(f"- {a}" for a in assessment)

    # Paper-trading recommendation
    if (not np.isnan(sharpe) and sharpe > 0.50
            and trade_count >= 30
            and not np.isnan(mdd) and abs(mdd) < 0.20):
        recommend = (
            "**Cautious yes for paper trading** — Sharpe positive, drawdown controlled. "
            "But with only 11 ETFs and < 100 trades, the sample is thin. "
            "Paper trade for the minimum 30-day observation period before drawing conclusions. "
            "Implement the correlation cap (v0.4) before going live."
        )
    else:
        recommend = (
            "**Not yet ready for paper trading.** The strategy needs either a larger universe "
            "(more ETFs, or back to equities with looser filters) or a longer observation window "
            "to build statistical confidence. Do not promote until OOS walk-forward shows "
            "consistent positive Sharpe across at least 60% of windows."
        )

    report = textwrap.dedent(f"""\
    # Strategy v0.3 Backtest Report
    **Universe:** 11 SPDR Sector ETFs — {', '.join(SECTOR_ETFS)}
    **Window:** {cfg.backtest_start} -> {cfg.backtest_end} | **Capital:** ${cfg.paper_capital:,.0f}
    **Entry:** Price > 200d SMA, RSI(14) < {cfg.rsi_oversold:.0f}, pullback >= {cfg.pullback_pct:.0%} from 10d high, SPY > 200d SMA
    **Exit:** RSI(14) crosses above {cfg.rsi_exit_level:.0f} (PRIMARY), +{cfg.profit_target_pct:.0%} target, -{cfg.stop_loss_pct:.0%} stop, {cfg.time_stop_days}d time stop, regime stop
    **Costs:** ${cfg.commission:.0f} commission + {cfg.slippage_bps:.0f} bps slippage/side | Volume filter: OFF | Regime gate: ON

    ---

    ## Full-Window Metrics

    {format_metrics_table(strategy_metrics, spy_metrics)}

    ---

    ## Exit Rule Breakdown

    {exit_table}

    ---

    ## Per-ETF Summary

    {etf_table}

    ---

    ## Walk-Forward Results (IS = 2yr / OOS = 6mo / Step = 3mo)

    {format_wf_table(wf_results)}

    ---

    ## Honest Assessment

    {verdict_para}

    ---

    ## Paper-Trading Recommendation

    {recommend}

    ---

    ## v0.3 vs v0.2 Design Changes (Summary)

    | Parameter | v0.2 | v0.3 | Rationale |
    |-----------|------|------|-----------|
    | Universe | 84 S&P 500 stocks | 11 sector ETFs | ETFs revert faster; no earnings gap risk |
    | RSI threshold | < 30 | < 35 | Sweep showed < 30 generates < 30 trades in 5 years |
    | Volume filter | on (1.5x avg) | off | Sweep: kills signals, zero quality improvement |
    | Primary exit | +10% profit target | RSI > 50 crossover | Signal-matched exit — exits when mean-reversion completes |
    | Backup profit target | +10% | +8% | Avg wins in sweep were ~4.5%; 10% was rarely hit |
    | Time stop | 10 days | 15 days | ETFs need more room to complete the reversion |

    ---

    ## Known Limitations

    - **Survivorship bias (mild):** All 11 ETFs exist through the full window; no delisting risk.
    - **Sector correlation (v0.4 to-do):** Simultaneous entries in correlated sectors double-count risk.
    - **Thin sample:** 11-ETF universe limits trade count. Statistical conclusions require caution.
    - **Stop execution:** sl_stop applied at bar close by vectorbt. Real gaps can blow through stops.
    - **2021-2026 window:** Predominantly bullish. Mean reversion strategies are more likely to
      outperform in range-bound or volatility-rich markets.

    ---
    *Generated by v0.3 backtest harness*
    """)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    logger.info("Report saved -> %s", output_path)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Strategy v0.3 backtest — sector ETFs")
    parser.add_argument("--refresh", action="store_true", help="Force re-download data")
    args = parser.parse_args()

    cfg = make_v03_config()
    all_tickers = list(set(SECTOR_ETFS + ["SPY"]))

    logger.info("=" * 60)
    logger.info("Strategy v0.3 — Mean Reversion on Sector ETFs")
    logger.info("Universe: %s", " ".join(SECTOR_ETFS))
    logger.info("RSI threshold: %.0f | Exit RSI: %.0f | Time stop: %d days",
                cfg.rsi_oversold, cfg.rsi_exit_level, cfg.time_stop_days)
    logger.info("Window: %s -> %s", cfg.backtest_start, cfg.backtest_end)
    logger.info("=" * 60)

    # Download data
    logger.info("Loading data ...")
    bars_all = download_bars(
        tickers=all_tickers,
        start=cfg.backtest_start,
        end=cfg.backtest_end,
        cache_dir=cfg.data_dir / "bars",
        force_refresh=args.refresh,
    )

    if "SPY" not in bars_all:
        logger.error("SPY data missing")
        sys.exit(1)

    spy_bars = bars_all.pop("SPY")
    tickers = [t for t in SECTOR_ETFS if t in bars_all]
    logger.info("ETFs loaded: %d / %d", len(tickers), len(SECTOR_ETFS))

    # Full-window backtest
    logger.info("Running full-window backtest ...")
    result = run_v03_backtest(bars_all, spy_bars, cfg, tickers)

    # SPY benchmark
    spy_metrics = spy_benchmark(spy_bars, cfg.backtest_start, cfg.backtest_end, cfg.paper_capital)
    logger.info("SPY B&H: ret=%.1f%%  Sharpe=%.2f", spy_metrics["total_return"]*100, spy_metrics["sharpe"])
    logger.info("v0.3:    ret=%.1f%%  Sharpe=%.2f  trades=%d",
                result.get("total_return", 0)*100,
                result.get("sharpe", float("nan")),
                result.get("trade_count", 0))

    # Trade log
    trade_log = pd.DataFrame()
    if result.get("portfolio") is not None:
        trade_log = build_trade_log(
            result["portfolio"], cfg, spy_bars, result.get("rsi_series", {})
        )
        tl_path = cfg.outputs_dir / "v03_trade_log.csv"
        trade_log.to_csv(tl_path, index=False)
        logger.info("Trade log saved -> %s  (%d trades)", tl_path, len(trade_log))

    # Walk-forward
    logger.info("Running walk-forward ...")
    windows = generate_windows(cfg)
    wf_results = []
    for window in windows:
        wf_result = run_wf_window_v03(bars_all, spy_bars, cfg, window)
        wf_results.append(wf_result)

    # Equity curve
    strategy_equity = result.get("equity")
    if strategy_equity is not None and not strategy_equity.empty:
        spy_equity = spy_metrics["equity"].reindex(strategy_equity.index, method="ffill")
        plot_equity_curve(
            strategy_equity=strategy_equity,
            spy_equity=spy_equity,
            output_path=cfg.outputs_dir / "v03_equity_curve.png",
            title="v0.3 Sector ETF Strategy vs SPY Buy & Hold",
        )

    # Report
    write_report(
        cfg=cfg,
        strategy_metrics=result,
        spy_metrics=spy_metrics,
        wf_results=wf_results,
        trade_log=trade_log,
        output_path=cfg.outputs_dir / "v03_report.md",
    )

    logger.info("=" * 60)
    logger.info("Done. Outputs in %s/", cfg.outputs_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
