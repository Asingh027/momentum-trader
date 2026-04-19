#!/usr/bin/env python
"""Scaling sweep — 6 strategy variants vs baseline.

Variants (all use Aggressive config: 20%/pos, 4 max, 20% floor, 2021-2026):
  Baseline     — 84-stock, 126d breakout, 50d trail, no sector cap
  Full S&P 500 — ~500 stocks, 126d, 50d trail
  Fast break   — 84-stock, 63d breakout, 50d trail
  Tight trail  — 84-stock, 126d, 20d trail
  Fast+tight   — 84-stock, 63d, 20d trail
  Sector-capped— 84-stock, 126d, 50d trail, max 2/GICS sector

Output: outputs/scaling_comparison.md

Usage:
    uv run python scripts/run_scaling_sweep.py
    uv run python scripts/run_scaling_sweep.py --refresh   # force re-download
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Baseline 84-stock universe ────────────────────────────────────────────────
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


def make_config(**extra) -> TradingConfig:
    cfg = TradingConfig()
    cfg = dataclasses.replace(cfg, **V1_OVERRIDES)
    cfg = dataclasses.replace(cfg, **AGGRESSIVE_OVERRIDES)
    if extra:
        cfg = dataclasses.replace(cfg, **extra)
    return cfg


# ── S&P 500 universe fetch ────────────────────────────────────────────────────

def fetch_sp500_universe() -> tuple[list[str], dict[str, str]]:
    """Return (tickers, sector_map) from Wikipedia S&P 500 table.

    Handles BRK.B → BRK-B style dot-to-dash conversion for yfinance.
    """
    import io
    import urllib.request

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    logger.info("Fetching S&P 500 constituent list from Wikipedia …")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8")
        tables = pd.read_html(io.StringIO(html), header=0)
        df = tables[0]
    except Exception as exc:
        logger.error("Failed to fetch S&P 500 list: %s", exc)
        return [], {}

    # Column names vary slightly; normalise
    col_map = {}
    for col in df.columns:
        low = col.lower()
        if "symbol" in low or "ticker" in low:
            col_map["symbol"] = col
        elif "gics sector" in low or "sector" in low:
            col_map["sector"] = col

    if "symbol" not in col_map:
        logger.error("Could not find Symbol column in Wikipedia table. Columns: %s", list(df.columns))
        return [], {}

    sym_col = col_map["symbol"]
    sec_col = col_map.get("sector")

    tickers_raw = df[sym_col].tolist()
    # yfinance: dots → hyphens (BRK.B → BRK-B)
    tickers = [str(t).replace(".", "-") for t in tickers_raw]

    sector_map: dict[str, str] = {}
    if sec_col:
        for raw, ticker in zip(tickers_raw, tickers):
            sector = df.loc[df[sym_col] == raw, sec_col].values
            if len(sector) > 0:
                sector_map[ticker] = str(sector[0])

    logger.info("S&P 500 list: %d tickers, %d with sector data", len(tickers), len(sector_map))
    return tickers, sector_map


def build_baseline_sector_map() -> dict[str, str]:
    """GICS sector mapping for the 84-stock baseline universe (static fallback)."""
    return {
        # Information Technology
        "AAPL": "Information Technology", "MSFT": "Information Technology",
        "NVDA": "Information Technology", "GOOGL": "Communication Services",
        "META": "Communication Services", "AVGO": "Information Technology",
        "AMD": "Information Technology", "INTC": "Information Technology",
        "QCOM": "Information Technology", "TXN": "Information Technology",
        "CRM": "Information Technology", "ADBE": "Information Technology",
        "NOW": "Information Technology", "ORCL": "Information Technology",
        "IBM": "Information Technology", "HPQ": "Information Technology",
        "AMAT": "Information Technology", "KLAC": "Information Technology",
        "LRCX": "Information Technology",
        # Health Care
        "JNJ": "Health Care", "UNH": "Health Care", "LLY": "Health Care",
        "ABBV": "Health Care", "MRK": "Health Care", "PFE": "Health Care",
        "ABT": "Health Care", "TMO": "Health Care", "DHR": "Health Care",
        "MDT": "Health Care", "BMY": "Health Care", "AMGN": "Health Care",
        "GILD": "Health Care",
        # Financials
        "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
        "GS": "Financials", "MS": "Financials", "BLK": "Financials",
        "AXP": "Financials", "USB": "Financials", "PNC": "Financials",
        # Consumer Discretionary
        "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
        "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
        "NKE": "Consumer Discretionary", "LOW": "Consumer Discretionary",
        "TGT": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
        "GM": "Consumer Discretionary", "F": "Consumer Discretionary",
        # Consumer Staples
        "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
        "WMT": "Consumer Staples", "COST": "Consumer Staples", "CL": "Consumer Staples",
        "MO": "Consumer Staples",
        # Industrials
        "CAT": "Industrials", "DE": "Industrials", "HON": "Industrials",
        "UPS": "Industrials", "BA": "Industrials", "GE": "Industrials",
        "LMT": "Industrials", "RTX": "Industrials",
        # Energy
        "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
        # Materials
        "LIN": "Materials", "APD": "Materials", "ECL": "Materials",
        # Utilities
        "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
        # Real Estate
        "AMT": "Real Estate", "PLD": "Real Estate", "EQIX": "Real Estate",
        # Communication Services
        "DIS": "Communication Services", "NFLX": "Communication Services",
        "CMCSA": "Communication Services", "T": "Communication Services",
        "VZ": "Communication Services",
    }


# ── Sector cap filter ─────────────────────────────────────────────────────────

def apply_sector_cap(
    entries: pd.DataFrame,
    sector_map: dict[str, str],
    max_per_sector: int,
) -> pd.DataFrame:
    """Limit signals to max_per_sector per GICS sector per bar.

    Tickers without sector data are bucketed as 'Unknown' and share the cap.
    Ordering within a sector follows column order (alphabetical after apply_position_cap).
    """
    result = entries.copy()
    for date, row in entries.iterrows():
        true_cols = row[row].index.tolist()
        if not true_cols:
            continue
        sector_counts: dict[str, int] = {}
        for ticker in true_cols:
            sector = sector_map.get(ticker, "Unknown")
            count = sector_counts.get(sector, 0)
            if count >= max_per_sector:
                result.loc[date, ticker] = False
            else:
                sector_counts[sector] = count + 1
    return result


# ── Core backtest runner ──────────────────────────────────────────────────────

def _normalize_idx(df: pd.DataFrame) -> pd.DataFrame:
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


def run_variant(
    bars: dict[str, pd.DataFrame],
    spy_bars: pd.DataFrame,
    cfg: TradingConfig,
    tickers: list[str],
    sector_map: dict[str, str] | None = None,
    sector_cap: int | None = None,
    label: str = "",
) -> dict:
    """Run one backtest variant. Returns metrics dict."""
    available = [t for t in tickers if t in bars]
    if not available:
        logger.error("%s: no available tickers", label)
        return _empty()

    common_idx = spy_bars.index
    close_df, open_df = build_price_matrices(bars, available, common_idx)

    start_ts = pd.Timestamp(cfg.backtest_start).normalize()
    end_ts = pd.Timestamp(cfg.backtest_end).normalize()

    logger.info("%s: computing signals (%d tickers) …", label, len(available))
    entries_full = compute_entry_signals(bars, spy_bars, cfg)
    exits_full = compute_exit_signals(bars, spy_bars, cfg)

    entries_full = apply_price_filter(entries_full, close_df, cfg)
    entries_full = apply_position_cap(entries_full, cfg)

    if sector_map and sector_cap:
        entries_full = apply_sector_cap(entries_full, sector_map, sector_cap)

    entries = entries_full.loc[
        (entries_full.index >= start_ts) & (entries_full.index <= end_ts)
    ]
    exits_df = exits_full.reindex_like(entries).fillna(False)
    close_win = close_df.loc[(close_df.index >= start_ts) & (close_df.index <= end_ts)]
    open_win = open_df.loc[(open_df.index >= start_ts) & (open_df.index <= end_ts)]

    n_signals = int(entries.sum().sum())
    logger.info("%s: %d entry signals in window", label, n_signals)
    if n_signals == 0:
        logger.warning("%s: zero signals — skipping", label)
        return _empty()

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
        "n_tickers": len(available),
    }
    logger.info(
        "%s: return=%.1f%% CAGR=%.1f%% sharpe=%.2f mdd=%.1f%% trades=%d avghold=%.1fd",
        label,
        m["total_return"] * 100, m["cagr"] * 100,
        m["sharpe"] if not np.isnan(m["sharpe"]) else -99,
        m["max_drawdown"] * 100,
        m["trade_count"],
        m["avg_hold_days"] if not np.isnan(m.get("avg_hold_days", float("nan"))) else 0,
    )
    return m


def _empty() -> dict:
    return {
        "total_return": float("nan"), "cagr": float("nan"),
        "sharpe": float("nan"), "sortino": float("nan"),
        "max_drawdown": float("nan"), "trade_count": 0,
        "win_rate": float("nan"), "avg_win": float("nan"),
        "avg_loss": float("nan"), "avg_hold_days": float("nan"),
        "equity": pd.Series(dtype=float), "n_tickers": 0,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def _fmt(val, pct=False, dec=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "n/a"
    if pct:
        return f"{val:.1%}"
    if dec == 0:
        return f"{int(val)}"
    return f"{val:.{dec}f}"


def write_report(
    variants: list[tuple[str, str, str, str, str, dict]],  # (label, universe, breakout, trail, sector_cap, metrics)
    spy_metrics: dict,
    cfg: TradingConfig,
    out_path: Path,
) -> None:
    header = (
        "| Variant | Universe | Breakout | Trail | Sector Cap "
        "| Trades | Sharpe | CAGR | Total Return | MaxDD | WinRate | AvgHold |"
    )
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"

    rows = [header, sep]
    for label, universe, breakout, trail, scap, m in variants:
        rows.append(
            f"| {label} | {universe} | {breakout} | {trail} | {scap} "
            f"| {_fmt(m['trade_count'], dec=0)} "
            f"| {_fmt(m['sharpe'])} "
            f"| {_fmt(m['cagr'], pct=True)} "
            f"| {_fmt(m['total_return'], pct=True)} "
            f"| {_fmt(m['max_drawdown'], pct=True)} "
            f"| {_fmt(m['win_rate'], pct=True)} "
            f"| {_fmt(m['avg_hold_days'])}d |"
        )

    spy_row = (
        f"| SPY B&H | — | — | — | — | — "
        f"| {_fmt(spy_metrics['sharpe'])} "
        f"| {_fmt(spy_metrics['cagr'], pct=True)} "
        f"| {_fmt(spy_metrics['total_return'], pct=True)} "
        f"| {_fmt(spy_metrics['max_drawdown'], pct=True)} | — | — |"
    )
    rows.append(spy_row)

    table = "\n".join(rows)

    # Determine which is best (by Sharpe, ignoring n/a)
    ranked = [
        (v[0], m["sharpe"]) for v, m in zip(variants, [v[5] for v in variants])
        if not np.isnan(m["sharpe"])
    ]
    # Fix: unpack properly
    ranked = [(label, m["sharpe"]) for label, _, _, _, _, m in variants if not np.isnan(m["sharpe"])]
    ranked.sort(key=lambda x: x[1], reverse=True)
    best_label, best_sharpe = ranked[0] if ranked else ("n/a", float("nan"))

    baseline_m = variants[0][5]
    assessments = []

    for label, _, _, _, _, m in variants[1:]:
        if np.isnan(m["sharpe"]) or np.isnan(baseline_m["sharpe"]):
            continue
        sharpe_delta = m["sharpe"] - baseline_m["sharpe"]
        cagr_delta = m["cagr"] - baseline_m["cagr"]
        mdd_delta = m["max_drawdown"] - baseline_m["max_drawdown"]
        trade_delta = m["trade_count"] - baseline_m["trade_count"]
        hold_delta = m.get("avg_hold_days", float("nan")) - baseline_m.get("avg_hold_days", float("nan"))

        lines = [f"\n### {label} vs Baseline"]
        lines.append(
            f"Sharpe {sharpe_delta:+.2f} | CAGR {cagr_delta:+.1%} | "
            f"MaxDD {mdd_delta:+.1%} | Trades {trade_delta:+d} | AvgHold {hold_delta:+.1f}d"
        )
        if abs(sharpe_delta) < 0.10:
            lines.append("**Verdict: noise.** Sharpe difference < 0.10 — within simulation variance.")
        elif sharpe_delta > 0:
            lines.append(f"**Verdict: improvement** — Sharpe higher by {sharpe_delta:.2f}.")
        else:
            lines.append(f"**Verdict: worse** — Sharpe lower by {abs(sharpe_delta):.2f}.")
        assessments.append("\n".join(lines))

    assessment_text = "\n".join(assessments) if assessments else "_No comparison data._"

    deploy_rec = (
        f"**Best by Sharpe: `{best_label}` ({best_sharpe:.2f})**. "
    )
    if best_label == "Baseline":
        deploy_rec += (
            "No variant strictly dominates the baseline. Stick with 84-stock universe, "
            "126d breakout, 50d trailing stop. Keep what works."
        )
    else:
        deploy_rec += (
            f"Consider switching if the improvement persists in walk-forward validation. "
            "A single full-window Sharpe improvement is necessary but not sufficient — "
            "run walk-forward OOS windows before deploying."
        )

    report = f"""\
# Strategy Scaling Comparison — v1.0 Momentum (Aggressive Config)

**Window:** {cfg.backtest_start} → {cfg.backtest_end} | **Capital:** ${cfg.paper_capital:,.0f}
**Config:** {cfg.target_position_pct:.0%}/pos, {cfg.max_positions} max, {cfg.cash_floor_pct:.0%} floor, 5 bps slippage

---

## Results Table

{table}

---

## Variant-by-Variant Assessment
{assessment_text}

---

## Deployment Recommendation

{deploy_rec}

### Limitations
- **Survivorship bias:** All universes use current S&P 500 membership. Delisted stocks are invisible.
- **Single window:** 2021-2026 is predominantly bullish. A single Sharpe number in a bull market is insufficient for deployment decisions.
- **Sector cap ordering:** Sector-capped variant drops tickers alphabetically when sector limit is hit — not by RS score. A proper implementation would rank by RS within each sector first.
- **Full S&P 500 data quality:** Some tickers may have short histories, survivorship artifacts, or corporate action noise not fully cleaned by the pipeline.

---
*Generated by run_scaling_sweep.py*
"""
    out_path.write_text(report, encoding="utf-8")
    logger.info("Report → %s", out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Force re-download all data")
    parser.add_argument("--skip-sp500", action="store_true",
                        help="Skip Full S&P 500 variant (saves ~10 min download)")
    args = parser.parse_args()

    cfg = make_config()
    logger.info("=== Scaling Sweep — v1.0 Momentum (Aggressive) ===")
    logger.info("Window: %s → %s | pos=%.0f%% max=%d floor=%.0f%%",
                cfg.backtest_start, cfg.backtest_end,
                cfg.target_position_pct * 100, cfg.max_positions, cfg.cash_floor_pct * 100)

    warmup_start = (pd.Timestamp(cfg.backtest_start) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")

    # ── S&P 500 list + sector data ────────────────────────────────────────
    sp500_tickers, sp500_sector_map = [], {}
    if not args.skip_sp500:
        sp500_tickers, sp500_sector_map = fetch_sp500_universe()

    baseline_sector_map = build_baseline_sector_map()
    # Merge: sp500 sector data takes precedence for any overlap
    merged_sector_map = {**baseline_sector_map, **sp500_sector_map}

    # ── Download baseline 84-stock data ───────────────────────────────────
    logger.info("Loading baseline 84-stock data …")
    bars_84 = download_bars(
        tickers=BASELINE_UNIVERSE,
        start=warmup_start,
        end=cfg.backtest_end,
        cache_dir=cfg.data_dir / "bars",
        force_refresh=args.refresh,
    )
    spy_bars = load_spy(
        start=warmup_start,
        end=cfg.backtest_end,
        cache_dir=cfg.data_dir / "bars",
        force_refresh=args.refresh,
    )

    for t in list(bars_84.keys()):
        bars_84[t] = _normalize_idx(bars_84[t])
    spy_bars = _normalize_idx(spy_bars)

    if spy_bars.empty:
        logger.error("No SPY data — aborting")
        sys.exit(1)

    # ── Download Full S&P 500 data (only new tickers) ─────────────────────
    bars_sp500 = {}
    if sp500_tickers and not args.skip_sp500:
        logger.info("Loading Full S&P 500 data (%d tickers — new ones will be downloaded) …",
                    len(sp500_tickers))
        bars_sp500_raw = download_bars(
            tickers=sp500_tickers,
            start=warmup_start,
            end=cfg.backtest_end,
            cache_dir=cfg.data_dir / "bars",
            force_refresh=args.refresh,
        )
        bars_sp500 = {t: _normalize_idx(df) for t, df in bars_sp500_raw.items()}

    # ── SPY benchmark ─────────────────────────────────────────────────────
    spy_metrics = spy_benchmark(spy_bars, cfg.backtest_start, cfg.backtest_end, cfg.paper_capital)
    logger.info("SPY B&H: return=%.1f%% sharpe=%.2f",
                spy_metrics["total_return"] * 100, spy_metrics["sharpe"])

    # ── Run all variants ──────────────────────────────────────────────────
    variants: list[tuple[str, str, str, str, str, dict]] = []

    # 1. Baseline
    logger.info("--- Variant: Baseline ---")
    m = run_variant(bars_84, spy_bars, make_config(), BASELINE_UNIVERSE, label="Baseline")
    variants.append(("Baseline", f"84", "126d", "50d", "none", m))

    # 2. Full S&P 500
    if bars_sp500:
        logger.info("--- Variant: Full S&P 500 ---")
        m = run_variant(bars_sp500, spy_bars, make_config(),
                        sp500_tickers, label="Full S&P 500")
        variants.append(("Full S&P", f"~{len([t for t in sp500_tickers if t in bars_sp500])}", "126d", "50d", "none", m))
    else:
        logger.info("Skipping Full S&P 500 variant (--skip-sp500 or no data)")
        variants.append(("Full S&P", "skipped", "126d", "50d", "none", _empty()))

    # 3. Fast breakout (63d)
    logger.info("--- Variant: Fast breakout (63d) ---")
    m = run_variant(bars_84, spy_bars, make_config(breakout_lookback=63),
                    BASELINE_UNIVERSE, label="Fast breakout")
    variants.append(("Fast breakout", "84", "63d", "50d", "none", m))

    # 4. Tight trailing stop (20d)
    logger.info("--- Variant: Tight trailing stop (20d) ---")
    m = run_variant(bars_84, spy_bars, make_config(trailing_sma_days=20),
                    BASELINE_UNIVERSE, label="Tight trail")
    variants.append(("Tight trail", "84", "126d", "20d", "none", m))

    # 5. Fast + tight (63d + 20d)
    logger.info("--- Variant: Fast+tight (63d + 20d) ---")
    m = run_variant(bars_84, spy_bars, make_config(breakout_lookback=63, trailing_sma_days=20),
                    BASELINE_UNIVERSE, label="Fast+tight")
    variants.append(("Fast+tight", "84", "63d", "20d", "none", m))

    # 6. Sector-capped (max 2/sector)
    logger.info("--- Variant: Sector-capped (max 2/sector) ---")
    m = run_variant(bars_84, spy_bars, make_config(), BASELINE_UNIVERSE,
                    sector_map=baseline_sector_map, sector_cap=2, label="Sector-capped")
    variants.append(("Sector-capped", "84", "126d", "50d", "max 2", m))

    # ── Write report ──────────────────────────────────────────────────────
    out_dir = cfg.outputs_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_report(variants, spy_metrics, cfg, out_dir / "scaling_comparison.md")

    logger.info("=== Scaling sweep complete ===")
    logger.info("Output: outputs/scaling_comparison.md")


if __name__ == "__main__":
    main()
