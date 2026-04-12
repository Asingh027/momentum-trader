"""Position sizing.

Rules from spec v0.2:
- 20% of portfolio per position
- Maximum 5 concurrent positions
- Always hold >= 10% cash (no new entries if cash < floor)

In a vectorbt backtest this is handled by:
- size=target_position_pct (as fraction of total equity)
- max_open_trades=max_positions

The cash floor is enforced by sizing: with 5 positions at 20% each = 100%,
but cash_floor of 10% means effective max allocation is 90% → 4.5 full positions.
vectorbt handles this naturally via init_cash + size_type='value_percent'.
"""

from __future__ import annotations

from trader.config import TradingConfig


def position_size_fraction(cfg: TradingConfig) -> float:
    """Fraction of total equity to allocate per trade."""
    return cfg.target_position_pct


def max_concurrent_positions(cfg: TradingConfig) -> int:
    return cfg.max_positions


def effective_max_allocation(cfg: TradingConfig) -> float:
    """Maximum fraction of portfolio deployable (respects cash floor)."""
    return 1.0 - cfg.cash_floor_pct
