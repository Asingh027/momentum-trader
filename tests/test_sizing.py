"""Tests for cash-floor-aware position sizing.

Spec rules:
- available = cash - (portfolio_value × cash_floor_pct)   (clamped to 0)
- notional   = min(portfolio_value × target_position_pct, available)
- With 20% position size and 10% floor on $25k: floor=$2,500, target=$5,000
  - Starting cash $25,000 → available=$22,500 → first 4 full ($5k each, $22,500→$2,500 left),
    5th notional = min($5,000, $2,500) = $2,500
"""

from __future__ import annotations

import pytest

from trader.config import TradingConfig
from trader.risk.sizing import compute_available_capital, compute_position_notional


@pytest.fixture
def cfg() -> TradingConfig:
    return TradingConfig(
        paper_capital=25_000.0,
        target_position_pct=0.20,
        cash_floor_pct=0.10,
        max_positions=5,
    )


# ── compute_available_capital ──────────────────────────────────────────────

class TestComputeAvailableCapital:

    def test_single_entry_full_cash(self, cfg):
        """Fresh account: floor=$2,500, available=$22,500."""
        available = compute_available_capital(
            portfolio_value=25_000.0, cash=25_000.0, cfg=cfg
        )
        assert available == pytest.approx(22_500.0)

    def test_floor_blocks_when_cash_below_floor(self, cfg):
        """Cash exactly at floor → available=0."""
        available = compute_available_capital(
            portfolio_value=25_000.0, cash=2_500.0, cfg=cfg
        )
        assert available == pytest.approx(0.0)

    def test_floor_clamps_to_zero_not_negative(self, cfg):
        """Cash below floor → clamped to 0, not negative."""
        available = compute_available_capital(
            portfolio_value=25_000.0, cash=1_000.0, cfg=cfg
        )
        assert available == 0.0

    def test_floor_uses_portfolio_value_as_base(self, cfg):
        """Floor is based on portfolio_value, not cash."""
        # portfolio=$30k, cash=$10k → floor=$3k, available=$7k
        available = compute_available_capital(
            portfolio_value=30_000.0, cash=10_000.0, cfg=cfg
        )
        assert available == pytest.approx(7_000.0)


# ── compute_position_notional ──────────────────────────────────────────────

class TestComputePositionNotional:

    def test_full_position_when_plenty_of_cash(self, cfg):
        """Available >> target → full target returned."""
        notional = compute_position_notional(
            portfolio_value=25_000.0, cash_available=22_500.0, cfg=cfg
        )
        assert notional == pytest.approx(5_000.0)

    def test_partial_position_when_cash_constrained(self, cfg):
        """Available < target → returns available (partial position)."""
        notional = compute_position_notional(
            portfolio_value=25_000.0, cash_available=2_500.0, cfg=cfg
        )
        assert notional == pytest.approx(2_500.0)

    def test_zero_notional_when_no_cash(self, cfg):
        """Zero available → zero notional (floor blocks entry)."""
        notional = compute_position_notional(
            portfolio_value=25_000.0, cash_available=0.0, cfg=cfg
        )
        assert notional == 0.0

    def test_zero_notional_when_negative_available(self, cfg):
        """Negative available (should not happen but be safe) → zero."""
        notional = compute_position_notional(
            portfolio_value=25_000.0, cash_available=-100.0, cfg=cfg
        )
        assert notional == 0.0


# ── Sequential deduction (simulates the runner loop) ─────────────────────

class TestSequentialEntries:

    def test_five_simultaneous_entries_respect_floor(self, cfg):
        """
        $25k portfolio, 20% position, 10% floor.
        Floor = $2,500. Max deployable = $22,500.
        4 full $5k positions = $20k → $2,500 remaining.
        5th position = min($5k, $2,500) = $2,500.
        Total deployed = $22,500 (not $25,000).
        """
        portfolio_value = 25_000.0
        cash = 25_000.0
        cash_available = compute_available_capital(portfolio_value, cash, cfg)

        notionals = []
        for _ in range(5):
            n = compute_position_notional(portfolio_value, cash_available, cfg)
            if n <= 0:
                break
            notionals.append(n)
            cash_available -= n

        assert len(notionals) == 5
        # First 4 are full size
        assert all(n == pytest.approx(5_000.0) for n in notionals[:4])
        # 5th is partial
        assert notionals[4] == pytest.approx(2_500.0)
        # Total does not breach floor
        assert sum(notionals) == pytest.approx(22_500.0)
        # Remaining available is 0
        assert cash_available == pytest.approx(0.0)

    def test_floor_blocks_6th_entry(self, cfg):
        """After 5 entries that exhaust available capital, 6th notional=0."""
        portfolio_value = 25_000.0
        cash = 25_000.0
        cash_available = compute_available_capital(portfolio_value, cash, cfg)

        for _ in range(5):
            n = compute_position_notional(portfolio_value, cash_available, cfg)
            cash_available -= max(n, 0)

        # 6th attempt
        sixth = compute_position_notional(portfolio_value, cash_available, cfg)
        assert sixth == 0.0
