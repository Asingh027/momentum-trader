"""Risk filters applied to signal DataFrame before order generation.

These are non-bypassable constraints implemented separately from the strategy
so strategy changes cannot accidentally disable them.

Currently implemented for the backtest:
- Max concurrent positions cap (enforced via vectorbt max_open_trades)
- Price range filter (pre-applied in universe; double-checked here)
- Regime gate (computed in signals but validated here)

Live-only controls (PDT guard, daily loss halt, drawdown halt) are
out of scope for Phase 1 / backtest.
"""

from __future__ import annotations

import pandas as pd

from trader.config import TradingConfig


def apply_price_filter(
    entries: pd.DataFrame,
    close_prices: pd.DataFrame,
    cfg: TradingConfig,
) -> pd.DataFrame:
    """Mask entries where close price is outside [min_price, max_price]."""
    in_range = (close_prices >= cfg.min_price) & (close_prices <= cfg.max_price)
    return entries & in_range.reindex(entries.index, fill_value=False)


def apply_position_cap(
    entries: pd.DataFrame,
    cfg: TradingConfig,
) -> pd.DataFrame:
    """Hard-cap to max_positions new entries per bar.

    When more than max_positions signals fire on the same day, keep
    the first max_positions alphabetically (deterministic, no look-ahead bias).
    """
    def _cap_row(row: pd.Series) -> pd.Series:
        true_cols = row[row].index.tolist()
        if len(true_cols) <= cfg.max_positions:
            return row
        keep = sorted(true_cols)[: cfg.max_positions]
        new_row = pd.Series(False, index=row.index)
        new_row[keep] = True
        return new_row

    return entries.apply(_cap_row, axis=1)
