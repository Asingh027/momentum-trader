#!/usr/bin/env python
"""Walk-forward validation for the two scaling-sweep winners.

Tests whether the performance of Tight trail (20d) and Fast+tight (63d+20d)
holds up OOS, and whether the improvement survives a 2022 bear-market window.

Decision rule (hardcoded at bottom):
  - 10+/11 OOS windows positive AND avg OOS Sharpe >= 1.5 → switch live config
  - <8/11 OR avg OOS Sharpe < 1.0 → overfit, keep baseline

Usage:
    uv run python scripts/run_walkforward_variants.py

Output: outputs/walkforward_variants.md
No network calls — uses cached parquet data only.
"""

from __future__ import annotations

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
from trader.backtest.walk_forward import generate_windows, slice_bars
from trader.config import TradingConfig
from trader.data.bars import download_bars, load_spy
from trader.risk.filters import apply_position_cap, apply_price_filter
from trader.signals.momentum import compute_entry_signals, compute_exit_signals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

UNIVERSE = [
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

BASE_OVERRIDES = dict(
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
    target_position_pct=0.20,
    max_positions=4,
    cash_floor_pct=0.20,
)

VARIANTS = [
    ("Tight trail (20d)", dict(trailing_sma_days=20)),
    ("Fast+tight (63d+20d)", dict(breakout_lookback=63, trailing_sma_days=20)),
]

# Baseline OOS results from the v1.0 walk-forward report (11/11 windows positive, etc.)
BASELINE_WF = {
    "label": "Baseline (50d trail)",
    "oos_positive": 11,
    "oos_total": 11,
    "avg_oos_sharpe": None,   # filled in below from actual data if available
    "worst_oos_sharpe": None,
    "full_sharpe": 2.24,
}


def make_config(**extra) -> TradingConfig:
    cfg = TradingConfig()
    cfg = dataclasses.replace(cfg, **BASE_OVERRIDES)
    if extra:
        cfg = dataclasses.replace(cfg, **extra)
    return cfg


def _normalize_idx(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = df.index.normalize()
    return df


def _empty_metrics() -> dict:
    return {
        "total_return": float("nan"), "cagr": float("nan"),
        "sharpe": float("nan"), "sortino": float("nan"),
        "max_drawdown": float("nan"), "trade_count": 0,
        "win_rate": float("nan"), "avg_win": float("nan"),
        "avg_loss": float("nan"), "avg_hold_days": float("nan"),
        "equity": pd.Series(dtype=float),
    }


def run_window(
    bars: dict[str, pd.DataFrame],
    spy_bars: pd.DataFrame,
    cfg: TradingConfig,
    tickers: list[str],
    start: str,
    end: str,
) -> dict:
    """Run one IS or OOS window with 300-day warmup for indicator seeding."""
    from dateutil.relativedelta import relativedelta

    warmup_start = (pd.Timestamp(start) - relativedelta(days=300)).strftime("%Y-%m-%d")

    w_bars = slice_bars(bars, warmup_start, end)
    w_spy = spy_bars.loc[spy_bars.index >= pd.Timestamp(warmup_start).normalize()]
    if w_spy.empty:
        return _empty_metrics()

    for t in list(w_bars.keys()):
        w_bars[t] = _normalize_idx(w_bars[t])
    w_spy = _normalize_idx(w_spy)
    common_idx = w_spy.index

    closes, opens = {}, {}
    for t in tickers:
        if t in w_bars:
            closes[t] = w_bars[t]["Close"].reindex(common_idx)
            opens[t] = w_bars[t]["Open"].reindex(common_idx)
    close_df = pd.DataFrame(closes, index=common_idx)
    open_df = pd.DataFrame(opens, index=common_idx)
    if close_df.empty:
        return _empty_metrics()

    entries = compute_entry_signals(w_bars, w_spy, cfg)
    exits_df = compute_exit_signals(w_bars, w_spy, cfg)
    entries = apply_price_filter(entries, close_df, cfg)
    entries = apply_position_cap(entries, cfg)

    # Trim to actual window
    ws = pd.Timestamp(start).normalize()
    we = pd.Timestamp(end).normalize()
    entries = entries.loc[(entries.index >= ws) & (entries.index <= we)]
    exits_df = exits_df.reindex_like(entries).fillna(False)
    close_df = close_df.loc[(close_df.index >= ws) & (close_df.index <= we)]
    open_df = open_df.loc[(open_df.index >= ws) & (open_df.index <= we)]

    if int(entries.sum().sum()) == 0:
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
        trade_stats = compute_trade_stats(pf.trades.records_readable)
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


def run_variant_wf(
    bars: dict[str, pd.DataFrame],
    spy_bars: pd.DataFrame,
    cfg: TradingConfig,
    tickers: list[str],
    label: str,
) -> list[dict]:
    """Run full walk-forward for one config variant. Returns list of window results."""
    windows = generate_windows(cfg)
    results = []
    for w in windows:
        logger.info("[%s] Window %d | IS %s–%s | OOS %s–%s",
                    label, w.window_idx, w.is_start, w.is_end, w.oos_start, w.oos_end)
        is_m = run_window(bars, spy_bars, cfg, tickers, w.is_start, w.is_end)
        oos_m = run_window(bars, spy_bars, cfg, tickers, w.oos_start, w.oos_end)
        logger.info(
            "  IS  sharpe=%.2f trades=%d | OOS sharpe=%.2f trades=%d",
            is_m["sharpe"] if not np.isnan(is_m["sharpe"]) else -99,
            is_m["trade_count"],
            oos_m["sharpe"] if not np.isnan(oos_m["sharpe"]) else -99,
            oos_m["trade_count"],
        )
        results.append({
            "window_idx": w.window_idx,
            "is_start": w.is_start, "is_end": w.is_end,
            "oos_start": w.oos_start, "oos_end": w.oos_end,
            "IS": is_m, "OOS": oos_m,
        })
    return results


def _fmt_sharpe(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.2f}"


def _fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.1%}"


def write_report(
    variant_results: list[tuple[str, list[dict]]],
    spy_metrics: dict,
    cfg: TradingConfig,
    out_path: Path,
) -> None:
    sections = []

    sections.append(f"""\
# Walk-Forward Variant Validation

**Universe:** 84 S&P 500 large-caps (cached only — no network)
**Window:** {cfg.backtest_start} → {cfg.backtest_end}
**IS:** 2 yr | **OOS:** 6 mo | **Step:** 3 mo

Decision rule: Switch live config if **10+/11 OOS windows positive** AND **avg OOS Sharpe ≥ 1.5**.
Keep baseline if **<8/11** OR **avg OOS Sharpe < 1.0**.

---
""")

    summary_rows = []

    for var_label, wf_results in variant_results:
        oos_metrics = [r["OOS"] for r in wf_results]
        valid = [m for m in oos_metrics if not np.isnan(m.get("sharpe", float("nan")))]
        oos_sharpes = [m["sharpe"] for m in valid]
        oos_positive = sum(1 for s in oos_sharpes if s > 0)
        oos_total = len(oos_sharpes)
        avg_sharpe = float(np.mean(oos_sharpes)) if oos_sharpes else float("nan")
        worst_sharpe = float(np.min(oos_sharpes)) if oos_sharpes else float("nan")

        # Find 2022 bear window: OOS covering 2022-10 to 2023-04 (approx window 2-3)
        bear_windows = [r for r in wf_results
                        if r["oos_start"] >= "2022-10-01" and r["oos_start"] <= "2023-04-01"]
        if not bear_windows:
            # Fallback: any window with OOS touching 2022
            bear_windows = [r for r in wf_results
                            if r["oos_start"] <= "2022-10-01" and r["oos_end"] >= "2022-04-01"]
        bear_sharpe = None
        bear_label = "n/a"
        if bear_windows:
            bw = bear_windows[0]
            bear_sharpe = bw["OOS"].get("sharpe", float("nan"))
            bear_label = f"{bw['oos_start']}–{bw['oos_end']}"

        summary_rows.append({
            "label": var_label,
            "oos_positive": oos_positive,
            "oos_total": oos_total,
            "avg_oos_sharpe": avg_sharpe,
            "worst_oos_sharpe": worst_sharpe,
            "bear_window": bear_label,
            "bear_sharpe": bear_sharpe,
        })

        # Per-window table
        header = "| Window | OOS Period | IS Sharpe | IS Trades | OOS Sharpe | OOS Return | OOS MaxDD | OOS Trades |"
        sep =    "|--------|-----------|-----------|-----------|------------|------------|-----------|------------|"
        rows = [header, sep]
        for r in wf_results:
            is_m = r["IS"]
            oos_m = r["OOS"]
            flag = " ✓" if oos_m.get("sharpe", float("nan")) > 0 else " ✗"
            rows.append(
                f"| {r['window_idx']} "
                f"| {r['oos_start']}–{r['oos_end']} "
                f"| {_fmt_sharpe(is_m.get('sharpe'))} "
                f"| {is_m.get('trade_count', 0)} "
                f"| {_fmt_sharpe(oos_m.get('sharpe'))}{flag} "
                f"| {_fmt_pct(oos_m.get('total_return'))} "
                f"| {_fmt_pct(oos_m.get('max_drawdown'))} "
                f"| {oos_m.get('trade_count', 0)} |"
            )
        table = "\n".join(rows)

        sections.append(f"""\
## {var_label}

{table}

**OOS Summary:**
- Positive windows: {oos_positive}/{oos_total}
- Average OOS Sharpe: {_fmt_sharpe(avg_sharpe)}
- Worst OOS Sharpe: {_fmt_sharpe(worst_sharpe)}
- Bear-market window ({bear_label}): {_fmt_sharpe(bear_sharpe)}

---
""")

    # Comparison table
    full_sharpes = {"Tight trail (20d)": 3.63, "Fast+tight (63d+20d)": 3.72}
    comp_header = "| Metric | Baseline (50d trail) | Tight trail (20d) | Fast+tight (63d+20d) |"
    comp_sep =    "| --- | --- | --- | --- |"
    comp_rows = [comp_header, comp_sep]

    def _get(key, label):
        for r in summary_rows:
            if r["label"] == label:
                return r.get(key)
        return None

    def _pos(label):
        r = _get("oos_positive", label)
        t = _get("oos_total", label)
        if r is None or t is None:
            return "n/a"
        return f"{r}/{t}"

    comp_rows.append(
        f"| OOS windows positive | 11/11 "
        f"| {_pos('Tight trail (20d)')} "
        f"| {_pos('Fast+tight (63d+20d)')} |"
    )
    comp_rows.append(
        f"| Avg OOS Sharpe | n/a (baseline run separately) "
        f"| {_fmt_sharpe(_get('avg_oos_sharpe', 'Tight trail (20d)'))} "
        f"| {_fmt_sharpe(_get('avg_oos_sharpe', 'Fast+tight (63d+20d)'))} |"
    )
    comp_rows.append(
        f"| Worst OOS Sharpe | n/a "
        f"| {_fmt_sharpe(_get('worst_oos_sharpe', 'Tight trail (20d)'))} "
        f"| {_fmt_sharpe(_get('worst_oos_sharpe', 'Fast+tight (63d+20d)'))} |"
    )
    comp_rows.append(
        f"| Full-window Sharpe | 2.24 "
        f"| {full_sharpes.get('Tight trail (20d)', 'n/a')} "
        f"| {full_sharpes.get('Fast+tight (63d+20d)', 'n/a')} |"
    )

    bear_t = _get("bear_sharpe", "Tight trail (20d)")
    bear_f = _get("bear_sharpe", "Fast+tight (63d+20d)")
    bear_tl = _get("bear_window", "Tight trail (20d)") or "n/a"
    comp_rows.append(
        f"| Bear window OOS Sharpe ({bear_tl}) | n/a "
        f"| {_fmt_sharpe(bear_t)} "
        f"| {_fmt_sharpe(bear_f)} |"
    )

    comp_table = "\n".join(comp_rows)

    sections.append(f"""\
## Summary Comparison

{comp_table}

---
""")

    # Decision
    decisions = []
    for row in summary_rows:
        label = row["label"]
        pos = row["oos_positive"]
        total = row["oos_total"]
        avg = row["avg_oos_sharpe"]
        if np.isnan(avg):
            decisions.append(f"**{label}:** No OOS data — cannot evaluate.")
            continue
        if pos >= 10 and avg >= 1.5:
            decisions.append(
                f"**{label}: DEPLOY** — {pos}/{total} OOS windows positive, avg OOS Sharpe {avg:.2f}. "
                f"Meets both thresholds. Switch live config."
            )
        elif pos < 8 or avg < 1.0:
            decisions.append(
                f"**{label}: KEEP BASELINE** — {pos}/{total} OOS windows positive, avg OOS Sharpe {avg:.2f}. "
                f"Below threshold. Full-window Sharpe is likely overfit to the 2021–2026 bull run."
            )
        else:
            decisions.append(
                f"**{label}: BORDERLINE** — {pos}/{total} OOS windows positive, avg OOS Sharpe {avg:.2f}. "
                f"Between thresholds. Consider extended paper-trading before deploying."
            )

    decision_text = "\n\n".join(decisions)

    sections.append(f"""\
## Deployment Decision

{decision_text}

### Caveats
- Walk-forward IS windows train on 2021–2024 data that is predominantly bullish. The only true bear test
  is the 2022 drawdown, which appears in 1–2 OOS windows depending on the step offset.
- A 20d trailing stop will whipsaw in choppy markets (mid-2023 sideways grind) — the OOS windows covering
  those periods are the real stress test.
- These results do not account for transaction costs beyond the 5 bps slippage model. Higher turnover
  (20d trail = ~23d avg hold vs 49d for 50d trail) roughly doubles execution cost drag.

---
*Generated by run_walkforward_variants.py — no network calls, cached 84-stock parquet only*
""")

    out_path.write_text("\n".join(sections), encoding="utf-8")
    logger.info("Report → %s", out_path)


def main() -> None:
    cfg = make_config()
    logger.info("=== Walk-Forward Variant Validation ===")
    logger.info("Window: %s → %s | IS=%dyr OOS=%dmo step=%dmo",
                cfg.backtest_start, cfg.backtest_end,
                cfg.wf_is_years, cfg.wf_oos_months, cfg.wf_step_months)

    warmup_start = (
        pd.Timestamp(cfg.backtest_start) - pd.Timedelta(days=400)
    ).strftime("%Y-%m-%d")

    logger.info("Loading cached bar data (no download) …")
    bars = download_bars(
        tickers=UNIVERSE,
        start=warmup_start,
        end=cfg.backtest_end,
        cache_dir=cfg.data_dir / "bars",
        force_refresh=False,
    )
    spy_bars = load_spy(
        start=warmup_start,
        end=cfg.backtest_end,
        cache_dir=cfg.data_dir / "bars",
        force_refresh=False,
    )

    for t in list(bars.keys()):
        bars[t] = _normalize_idx(bars[t])
    spy_bars = _normalize_idx(spy_bars)

    available = [t for t in UNIVERSE if t in bars]
    logger.info("Available tickers: %d / %d", len(available), len(UNIVERSE))

    variant_results = []
    for var_label, overrides in VARIANTS:
        var_cfg = make_config(**overrides)
        logger.info("=== Variant: %s ===", var_label)
        wf = run_variant_wf(bars, spy_bars, var_cfg, available, var_label)
        variant_results.append((var_label, wf))

    out_dir = cfg.outputs_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_report(variant_results, {}, var_cfg, out_dir / "walkforward_variants.md")

    logger.info("=== Walk-forward validation complete ===")
    logger.info("Output: outputs/walkforward_variants.md")


if __name__ == "__main__":
    main()
