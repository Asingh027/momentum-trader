"""TradingConfig — all strategy parameters loaded from environment variables.

Load with: cfg = TradingConfig.from_env()
Override: TradingConfig(paper_capital=50000, ...)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_float(key: str, default: float) -> float:
    val = os.environ.get(key, "")
    return float(val.replace("_", "")) if val.strip() else default


def _env_int(key: str, default: int) -> int:
    val = os.environ.get(key, "")
    return int(val.replace("_", "")) if val.strip() else default


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


@dataclass
class TradingConfig:
    # ── Capital & Sizing ─────────────────────────────────────────────────────
    paper_capital: float = 25_000.0
    target_position_pct: float = 0.20
    max_positions: int = 5
    cash_floor_pct: float = 0.10

    # ── Universe ──────────────────────────────────────────────────────────────
    min_market_cap: float = 5_000_000_000.0
    min_price: float = 20.0
    max_price: float = 500.0
    min_addv: float = 50_000_000.0  # avg daily dollar volume (20d)

    # ── Entry Signal ──────────────────────────────────────────────────────────
    trend_sma_period: int = 200
    pullback_lookback: int = 10
    pullback_pct: float = 0.05
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    volume_ratio: float = 1.5
    volume_avg_period: int = 20
    earnings_buffer_days: int = 5

    # ── Exit ──────────────────────────────────────────────────────────────────
    profit_target_pct: float = 0.10
    stop_loss_pct: float = 0.05
    time_stop_days: int = 10
    spy_trend_sma_period: int = 200

    # ── Costs ─────────────────────────────────────────────────────────────────
    commission: float = 0.0
    slippage_bps: float = 5.0  # per side

    # ── Backtest Window ───────────────────────────────────────────────────────
    backtest_start: str = "2021-04-01"
    backtest_end: str = "2026-04-01"

    # ── Walk-Forward ──────────────────────────────────────────────────────────
    wf_is_years: int = 2
    wf_oos_months: int = 6
    wf_step_months: int = 3

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_dir: Path = field(default_factory=lambda: Path("data"))
    outputs_dir: Path = field(default_factory=lambda: Path("outputs"))

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.outputs_dir = Path(self.outputs_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def slippage_rate(self) -> float:
        """One-way slippage as a fraction (e.g. 5 bps = 0.0005)."""
        return self.slippage_bps / 10_000.0

    @classmethod
    def from_env(cls) -> "TradingConfig":
        return cls(
            paper_capital=_env_float("PAPER_CAPITAL", 25_000.0),
            target_position_pct=_env_float("TARGET_POSITION_PCT", 0.20),
            max_positions=_env_int("MAX_POSITIONS", 5),
            cash_floor_pct=_env_float("CASH_FLOOR_PCT", 0.10),
            min_market_cap=_env_float("MIN_MARKET_CAP", 5_000_000_000.0),
            min_price=_env_float("MIN_PRICE", 20.0),
            max_price=_env_float("MAX_PRICE", 500.0),
            min_addv=_env_float("MIN_ADDV", 50_000_000.0),
            trend_sma_period=_env_int("TREND_SMA_PERIOD", 200),
            pullback_lookback=_env_int("PULLBACK_LOOKBACK", 10),
            pullback_pct=_env_float("PULLBACK_PCT", 0.05),
            rsi_period=_env_int("RSI_PERIOD", 14),
            rsi_oversold=_env_float("RSI_OVERSOLD", 30.0),
            volume_ratio=_env_float("VOLUME_RATIO", 1.5),
            volume_avg_period=_env_int("VOLUME_AVG_PERIOD", 20),
            earnings_buffer_days=_env_int("EARNINGS_BUFFER_DAYS", 5),
            profit_target_pct=_env_float("PROFIT_TARGET_PCT", 0.10),
            stop_loss_pct=_env_float("STOP_LOSS_PCT", 0.05),
            time_stop_days=_env_int("TIME_STOP_DAYS", 10),
            spy_trend_sma_period=_env_int("SPY_TREND_SMA_PERIOD", 200),
            commission=_env_float("COMMISSION", 0.0),
            slippage_bps=_env_float("SLIPPAGE_BPS", 5.0),
            backtest_start=_env_str("BACKTEST_START", "2021-04-01"),
            backtest_end=_env_str("BACKTEST_END", "2026-04-01"),
            wf_is_years=_env_int("WF_IS_YEARS", 2),
            wf_oos_months=_env_int("WF_OOS_MONTHS", 6),
            wf_step_months=_env_int("WF_STEP_MONTHS", 3),
            data_dir=Path(_env_str("DATA_DIR", "data")),
            outputs_dir=Path(_env_str("OUTPUTS_DIR", "outputs")),
        )
