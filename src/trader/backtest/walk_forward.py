"""Walk-forward validation.

Rolling windows:
- In-sample (IS): 2 years
- Out-of-sample (OOS): 6 months
- Step: 3 months (quarterly advance)

For each window we run the full backtest pipeline (signals → risk filters →
engine) and report IS vs OOS metrics separately.

NOTE: We do NOT re-optimise parameters between windows — the strategy has fixed
parameters from config. Walk-forward here tests whether the edge is stable
across different market regimes, not whether optimised params transfer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd
from dateutil.relativedelta import relativedelta

from trader.config import TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class WFWindow:
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    window_idx: int


def generate_windows(cfg: TradingConfig) -> list[WFWindow]:
    """Generate all IS/OOS date windows for the backtest period."""
    start = pd.Timestamp(cfg.backtest_start)
    end = pd.Timestamp(cfg.backtest_end)

    windows = []
    idx = 0
    is_start = start

    while True:
        is_end = is_start + relativedelta(years=cfg.wf_is_years)
        oos_start = is_end
        oos_end = oos_start + relativedelta(months=cfg.wf_oos_months)

        if oos_end > end:
            break

        windows.append(WFWindow(
            is_start=is_start.strftime("%Y-%m-%d"),
            is_end=is_end.strftime("%Y-%m-%d"),
            oos_start=oos_start.strftime("%Y-%m-%d"),
            oos_end=oos_end.strftime("%Y-%m-%d"),
            window_idx=idx,
        ))

        is_start = is_start + relativedelta(months=cfg.wf_step_months)
        idx += 1

    logger.info("Walk-forward: %d windows generated", len(windows))
    for w in windows:
        logger.debug("  Window %d: IS %s–%s, OOS %s–%s", w.window_idx, w.is_start, w.is_end, w.oos_start, w.oos_end)

    return windows


def slice_bars(
    bars: dict[str, pd.DataFrame],
    start: str,
    end: str,
) -> dict[str, pd.DataFrame]:
    """Slice all ticker DataFrames to a date window."""
    sliced = {}
    for ticker, df in bars.items():
        s = df.loc[start:end]
        if not s.empty:
            sliced[ticker] = s
    return sliced


def format_wf_table(wf_results: list[dict]) -> str:
    """Return a markdown table of IS vs OOS metrics across windows."""
    if not wf_results:
        return "_No walk-forward results._"

    header = "| Window | Period | Total Return | CAGR | Sharpe | Sortino | Max DD | Trades |"
    sep = "|--------|--------|-------------|------|--------|---------|--------|--------|"
    rows = [header, sep]

    for r in wf_results:
        for phase in ("IS", "OOS"):
            m = r.get(phase, {})
            rows.append(
                f"| {r['window_idx']} | {phase} {r[f'{phase}_start']}–{r[f'{phase}_end']} "
                f"| {m.get('total_return', float('nan')):.1%} "
                f"| {m.get('cagr', float('nan')):.1%} "
                f"| {m.get('sharpe', float('nan')):.2f} "
                f"| {m.get('sortino', float('nan')):.2f} "
                f"| {m.get('max_drawdown', float('nan')):.1%} "
                f"| {m.get('trade_count', 0)} |"
            )

    return "\n".join(rows)
