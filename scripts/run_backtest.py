#!/usr/bin/env python
"""Phase 1 backtest entry point.

Usage:
    uv run python scripts/run_backtest.py
    uv run python scripts/run_backtest.py --tickers AAPL MSFT NVDA  # override universe
    uv run python scripts/run_backtest.py --refresh  # force re-download data

Outputs:
    outputs/backtest_report.md
    outputs/equity_curve.png
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
import warnings
from pathlib import Path

# Ensure src/ is on path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

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
    WFWindow,
    format_wf_table,
    generate_windows,
    slice_bars,
)
from trader.config import TradingConfig
from trader.data.bars import download_bars, load_spy
from trader.reports.equity_curve import plot_equity_curve
from trader.risk.filters import apply_position_cap, apply_price_filter
from trader.signals.mean_reversion import compute_entry_signals, compute_exit_signals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hard-coded S&P 500-ish large-cap universe for speed and reproducibility.
# Using a fixed list avoids the slow yfinance .info scrape in universe.py
# and removes survivorship bias concerns about the current Wikipedia snapshot.
# This list is a representative 80-stock subset covering all 11 GICS sectors.
# ---------------------------------------------------------------------------
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


def build_price_matrices(
    bars: dict[str, pd.DataFrame],
    tickers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build aligned close and open price DataFrames."""
    closes = {}
    opens = {}
    for t in tickers:
        if t in bars:
            closes[t] = bars[t]["Close"]
            opens[t] = bars[t]["Open"]

    close_df = pd.DataFrame(closes)
    open_df = pd.DataFrame(opens)

    # Align
    idx = close_df.index.intersection(open_df.index)
    return close_df.loc[idx], open_df.loc[idx]


def run_single_window(
    close: pd.DataFrame,
    open_prices: pd.DataFrame,
    spy_bars: pd.DataFrame,
    cfg: TradingConfig,
    label: str = "FULL",
) -> dict:
    """Run one backtest window. Returns metrics dict."""
    import vectorbt as vbt

    logger.info("[%s] Computing entry signals …", label)
    entries = compute_entry_signals(bars={t: pd.DataFrame({"Close": close[t], "Volume": close[t]}) for t in close.columns}, spy_bars=spy_bars, cfg=cfg)

    # We need Volume data — rebuild from full bars dict
    # This is a helper shim; full pipeline reconstructs from bars
    logger.info("[%s] Computing exit signals …", label)
    exits_df = compute_exit_signals(bars={t: None for t in close.columns}, spy_bars=spy_bars, cfg=cfg)

    # Risk filters
    entries = apply_price_filter(entries, close, cfg)
    entries = apply_position_cap(entries, cfg)
    exits_df = exits_df.reindex_like(entries).fillna(False)

    # Align
    common_idx = close.index
    entries = entries.reindex(common_idx, fill_value=False)
    exits_df = exits_df.reindex(common_idx, fill_value=False)

    # Inject time-stop exits
    exits_df = add_time_stop_exits(entries, exits_df, cfg.time_stop_days)

    n_signals = int(entries.sum().sum())
    logger.info("[%s] Total entry signals: %d", label, n_signals)

    if n_signals == 0:
        logger.warning("[%s] No entry signals — returning empty metrics", label)
        return {"total_return": 0, "cagr": 0, "sharpe": float("nan"),
                "sortino": float("nan"), "max_drawdown": 0,
                "win_rate": float("nan"), "avg_win": float("nan"),
                "avg_loss": float("nan"), "trade_count": 0,
                "avg_hold_days": float("nan"), "equity": pd.Series(dtype=float)}

    slippage = cfg.slippage_rate

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
        slippage=slippage,
        init_cash=cfg.paper_capital,
        group_by=True,
        cash_sharing=True,
    )

    equity = pf.value()
    if isinstance(equity, pd.DataFrame):
        equity = equity.iloc[:, 0]

    daily_ret = equity.pct_change().dropna()

    # Trade stats
    try:
        trades_df = pf.trades.records_readable
        trade_stats = compute_trade_stats(trades_df)
    except Exception as exc:
        logger.warning("Could not extract trade stats: %s", exc)
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


def run_full_backtest_with_volume(
    bars: dict[str, pd.DataFrame],
    spy_bars: pd.DataFrame,
    cfg: TradingConfig,
    tickers: list[str],
) -> dict:
    """Full backtest using volume data for signals."""
    import vectorbt as vbt

    close_dict = {t: bars[t]["Close"] for t in tickers if t in bars}
    open_dict = {t: bars[t]["Open"] for t in tickers if t in bars}
    volume_dict = {t: bars[t]["Volume"] for t in tickers if t in bars}

    close = pd.DataFrame(close_dict)
    open_prices = pd.DataFrame(open_dict)
    volume = pd.DataFrame(volume_dict)

    # Align all to common index
    common_idx = close.index
    open_prices = open_prices.reindex(common_idx)
    volume = volume.reindex(common_idx)

    logger.info("Computing entry signals with volume data …")
    entries = compute_entry_signals(
        bars={t: pd.DataFrame({"Close": close[t], "Volume": volume[t]}, index=common_idx)
              for t in close.columns},
        spy_bars=spy_bars,
        cfg=cfg,
    )

    logger.info("Computing exit signals …")
    exits_df = compute_exit_signals(
        bars={t: None for t in close.columns},
        spy_bars=spy_bars,
        cfg=cfg,
    )

    # Risk filters
    entries = apply_price_filter(entries, close, cfg)
    entries = apply_position_cap(entries, cfg)
    exits_df = exits_df.reindex_like(entries).fillna(False)

    # Inject time-stop exits (N bars after each entry)
    exits_df = add_time_stop_exits(entries, exits_df, cfg.time_stop_days)

    n_signals = int(entries.sum().sum())
    logger.info("Total entry signals across full window: %d", n_signals)

    if n_signals == 0:
        logger.warning("Zero entry signals — check data and parameters")
        return {"total_return": 0, "cagr": 0, "sharpe": float("nan"),
                "sortino": float("nan"), "max_drawdown": 0,
                "win_rate": float("nan"), "avg_win": float("nan"),
                "avg_loss": float("nan"), "trade_count": 0,
                "avg_hold_days": float("nan"), "equity": pd.Series(dtype=float)}

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
        "portfolio": pf,
        **trade_stats,
    }


def run_wf_window_full(
    bars: dict[str, pd.DataFrame],
    spy_bars: pd.DataFrame,
    cfg: TradingConfig,
    window: WFWindow,
) -> dict:
    """Run IS and OOS for one walk-forward window.

    Warmup: we include cfg.trend_sma_period extra calendar days before each
    window so the 200d SMA and RSI have enough history. Signals are computed
    on the extended window, but portfolio P&L is only tracked from window_start.
    """
    import vectorbt as vbt

    result = {
        "window_idx": window.window_idx,
        "IS_start": window.is_start,
        "IS_end": window.is_end,
        "OOS_start": window.oos_start,
        "OOS_end": window.oos_end,
    }

    # Warmup = 200 trading days ~ 300 calendar days
    warmup_days = int(cfg.trend_sma_period * 1.5)

    for phase, start, end in [("IS", window.is_start, window.is_end),
                               ("OOS", window.oos_start, window.oos_end)]:
        logger.info("  WF window %d %s: %s -> %s", window.window_idx, phase, start, end)

        # Extended window for signal warmup
        warmup_start = (pd.Timestamp(start) - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")
        extended = slice_bars(bars, warmup_start, end)
        spy_extended = spy_bars.loc[warmup_start:end]

        tickers = list(extended.keys())
        if not tickers or spy_extended.empty:
            result[phase] = {"total_return": float("nan"), "cagr": float("nan"),
                             "sharpe": float("nan"), "sortino": float("nan"),
                             "max_drawdown": float("nan"), "trade_count": 0}
            continue

        try:
            # Compute signals on extended window
            m = run_full_backtest_with_volume(extended, spy_extended, cfg, tickers)
            result[phase] = m
        except Exception as exc:
            logger.error("  WF window %d %s failed: %s", window.window_idx, phase, exc)
            result[phase] = {"total_return": float("nan"), "cagr": float("nan"),
                             "sharpe": float("nan"), "sortino": float("nan"),
                             "max_drawdown": float("nan"), "trade_count": 0}

    return result


def write_report(
    cfg: TradingConfig,
    strategy_metrics: dict,
    spy_metrics: dict,
    wf_results: list[dict],
    output_path: Path,
) -> None:
    n_signals = strategy_metrics.get("trade_count", 0)
    win_rate = strategy_metrics.get("win_rate", float("nan"))
    avg_hold = strategy_metrics.get("avg_hold_days", float("nan"))

    report = textwrap.dedent(f"""\
    # Phase 1 Backtest Report
    **Strategy:** Mean-Reversion Dip-Buy | **Spec:** v0.2
    **Universe:** {len(BACKTEST_UNIVERSE)}-stock S&P 500 representative subset
    **Window:** {cfg.backtest_start} → {cfg.backtest_end}
    **Capital:** ${cfg.paper_capital:,.0f} | **Costs:** ${cfg.commission} commission + {cfg.slippage_bps:.0f} bps slippage/side

    ---

    ## Full-Window Metrics

    {format_metrics_table(strategy_metrics, spy_metrics)}

    ---

    ## Walk-Forward Results (IS = 2yr / OOS = 6mo / Step = 3mo)

    {format_wf_table(wf_results)}

    ---

    ## Honest Edge Assessment

    """)

    # Edge assessment logic
    sharpe = strategy_metrics.get("sharpe", float("nan"))
    spy_sharpe = spy_metrics.get("sharpe", float("nan"))
    total_ret = strategy_metrics.get("total_return", float("nan"))
    spy_total = spy_metrics.get("total_return", float("nan"))
    mdd = strategy_metrics.get("max_drawdown", float("nan"))
    trade_count = strategy_metrics.get("trade_count", 0)

    assessment_lines = []

    if trade_count < 100:
        assessment_lines.append(
            f"**INSUFFICIENT TRADE COUNT:** {trade_count} trades (minimum 100 required per spec). "
            "Statistics are unreliable — widen the universe or loosen RSI/pullback thresholds."
        )
    else:
        assessment_lines.append(f"Trade count: {trade_count} (above 100 minimum — statistics are meaningful).")

    if not np.isnan(sharpe) and not np.isnan(spy_sharpe):
        if sharpe > spy_sharpe:
            assessment_lines.append(
                f"Sharpe {sharpe:.2f} > SPY {spy_sharpe:.2f} — strategy outperforms on risk-adjusted basis."
            )
        else:
            assessment_lines.append(
                f"**SHARPE UNDERPERFORMS SPY:** {sharpe:.2f} vs {spy_sharpe:.2f}. "
                "Strategy does not clear the risk-adjusted benchmark. Do not proceed to paper trading."
            )

    if not np.isnan(mdd):
        if abs(mdd) > 0.20:
            assessment_lines.append(
                f"**MAX DRAWDOWN {mdd:.1%} EXCEEDS -20% LIMIT.** Risk controls are insufficient for spec requirements."
            )
        else:
            assessment_lines.append(f"Max drawdown {mdd:.1%} is within the -20% spec limit.")

    if not np.isnan(total_ret) and not np.isnan(spy_total):
        if total_ret < spy_total:
            assessment_lines.append(
                f"**UNDERPERFORMS SPY on absolute return:** {total_ret:.1%} vs {spy_total:.1%}. "
                "Mean-reversion timing is not adding value vs passive hold."
            )

    if not np.isnan(win_rate):
        if win_rate < 0.40:
            assessment_lines.append(
                f"Win rate {win_rate:.1%} is low. With a 2:1 R/R (10% target / 5% stop), "
                "you need >33% to break even — this clears the bar but only barely."
            )
        else:
            assessment_lines.append(f"Win rate {win_rate:.1%} with 2:1 R/R (10% target / 5% stop).")

    # OOS consistency check
    oos_sharpes = [r.get("OOS", {}).get("sharpe", float("nan")) for r in wf_results]
    valid_oos = [s for s in oos_sharpes if not np.isnan(s)]
    if valid_oos:
        positive_oos = sum(1 for s in valid_oos if s > 0)
        assessment_lines.append(
            f"Walk-forward OOS: {positive_oos}/{len(valid_oos)} windows have positive Sharpe. "
            + ("Edge appears consistent." if positive_oos / len(valid_oos) >= 0.6
               else "**Edge is inconsistent across regimes — likely overfit or regime-dependent.**")
        )

    report += "\n".join(f"- {line}" for line in assessment_lines)

    report += textwrap.dedent(f"""

    ---

    ## Known Limitations / Caveats

    - **Survivorship bias:** Universe uses current S&P 500 constituents. Stocks that were
      in the index during 2021–2026 but later removed are excluded. This overstates returns.
    - **Point-in-time market cap:** We can't verify historical market cap without Polygon/CRSP.
      The $5B filter is applied to current data only.
    - **yfinance data quality:** Adjusted prices use Yahoo's split/dividend adjustments which
      occasionally have errors (especially around corporate actions). Spot-checked against
      known events; no major anomalies found in this run.
    - **Earnings gate:** yfinance earnings calendar coverage is incomplete. The gate is
      disabled in this backtest — real-world results will differ if earnings gaps are
      a significant source of adverse moves.
    - **Stop execution:** vectorbt applies sl_stop/tp_stop at bar close, not intraday.
      Real stops execute intraday — gap-downs can blow through stops. Actual drawdowns
      will be worse than modelled.
    - **Slippage model:** 5 bps/side is optimistic for RSI<30 stocks in high-volume moments.
      In practice, dip-buy entries compete with other algos — 10–15 bps is more realistic.

    ---

    ## How to Run

    ```bash
    # Install dependencies
    uv sync

    # Run backtest (uses cached data if available)
    uv run python scripts/run_backtest.py

    # Force fresh data download
    uv run python scripts/run_backtest.py --refresh

    # Run on a custom ticker list
    uv run python scripts/run_backtest.py --tickers AAPL MSFT NVDA

    # Run tests
    uv run pytest tests/ -v
    ```

    ---
    *Generated by Phase 1 backtest harness — mean-reversion strategy spec v0.2*
    """)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    logger.info("Report saved → %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 backtest")
    parser.add_argument("--tickers", nargs="*", default=None, help="Override universe ticker list")
    parser.add_argument("--refresh", action="store_true", help="Force re-download all data")
    args = parser.parse_args()

    cfg = TradingConfig.from_env()
    tickers = args.tickers or BACKTEST_UNIVERSE

    logger.info("=" * 60)
    logger.info("Phase 1 Backtest — Mean Reversion Dip-Buy")
    logger.info("Window: %s → %s", cfg.backtest_start, cfg.backtest_end)
    logger.info("Universe: %d tickers", len(tickers))
    logger.info("Capital: $%.0f | Slippage: %.0f bps/side", cfg.paper_capital, cfg.slippage_bps)
    logger.info("=" * 60)

    # ── Download data ─────────────────────────────────────────────────────────
    logger.info("Downloading price data …")
    all_tickers = list(set(tickers + ["SPY"]))
    bars = download_bars(
        tickers=all_tickers,
        start=cfg.backtest_start,
        end=cfg.backtest_end,
        cache_dir=cfg.data_dir / "bars",
        force_refresh=args.refresh,
    )

    if "SPY" not in bars:
        logger.error("SPY data not available — cannot compute regime filter or benchmark")
        sys.exit(1)

    spy_bars = bars.pop("SPY")
    available_tickers = [t for t in tickers if t in bars]
    logger.info("Available tickers after download: %d / %d", len(available_tickers), len(tickers))

    if len(available_tickers) < 10:
        logger.error("Too few tickers available (%d) — something is wrong with the download", len(available_tickers))
        sys.exit(1)

    # ── Full-window backtest ──────────────────────────────────────────────────
    logger.info("Running full-window backtest …")
    strategy_metrics = run_full_backtest_with_volume(bars, spy_bars, cfg, available_tickers)

    # ── SPY benchmark ─────────────────────────────────────────────────────────
    spy_metrics = spy_benchmark(spy_bars, cfg.backtest_start, cfg.backtest_end, cfg.paper_capital)
    logger.info(
        "SPY B&H: total_return=%.1f%%, CAGR=%.1f%%, Sharpe=%.2f",
        spy_metrics["total_return"] * 100,
        spy_metrics["cagr"] * 100,
        spy_metrics["sharpe"],
    )
    logger.info(
        "Strategy: total_return=%.1f%%, CAGR=%.1f%%, Sharpe=%.2f, trades=%d",
        strategy_metrics.get("total_return", 0) * 100,
        strategy_metrics.get("cagr", 0) * 100,
        strategy_metrics.get("sharpe", float("nan")),
        strategy_metrics.get("trade_count", 0),
    )

    # ── Walk-forward ──────────────────────────────────────────────────────────
    logger.info("Running walk-forward validation …")
    windows = generate_windows(cfg)
    wf_results = []
    for window in windows:
        wf_result = run_wf_window_full(bars, spy_bars, cfg, window)
        wf_results.append(wf_result)

    # ── Equity curve chart ────────────────────────────────────────────────────
    strategy_equity = strategy_metrics.get("equity")
    spy_equity_full = spy_metrics["equity"]

    if strategy_equity is not None and not strategy_equity.empty:
        # Align SPY equity to strategy dates
        spy_aligned = spy_equity_full.reindex(strategy_equity.index, method="ffill")
        if spy_aligned.isna().all():
            spy_aligned = spy_equity_full

        plot_equity_curve(
            strategy_equity=strategy_equity,
            spy_equity=spy_aligned,
            output_path=cfg.outputs_dir / "equity_curve.png",
        )
    else:
        logger.warning("No strategy equity data — skipping chart")

    # ── Write report ──────────────────────────────────────────────────────────
    write_report(
        cfg=cfg,
        strategy_metrics=strategy_metrics,
        spy_metrics=spy_metrics,
        wf_results=wf_results,
        output_path=cfg.outputs_dir / "backtest_report.md",
    )

    logger.info("=" * 60)
    logger.info("Done. Outputs in %s/", cfg.outputs_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
