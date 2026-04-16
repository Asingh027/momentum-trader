# Trader — Automated Swing Trading System

Mean-reversion dip-buy strategy on S&P 500. Phase 1: data pipeline + backtest harness only.
No paper execution, no Alpaca API calls.

## Framework Choice: vectorbt

**Why vectorbt over bt or backtesting.py:**
- Vectorized operations on pandas — runs a 5-year daily backtest in seconds, not minutes
- Built-in walk-forward support and portfolio-level metrics
- Native portfolio construction with per-trade entry/exit signals as boolean arrays
- Sharpe, Sortino, drawdown, win rate all computed in one call
- Active maintenance, NumPy/pandas-native (no hidden magic)

**Trade-off:** Steeper learning curve than backtesting.py; requires thinking in signal arrays rather than event callbacks.

## Quick Start

```bash
# 1. Install dependencies (requires uv)
uv sync

# 2. Copy and configure environment
cp .env.example .env
# edit .env if desired (defaults match spec v0.2)

# 3. Run the backtest
uv run python scripts/run_backtest.py

# Outputs written to outputs/
```

## Project Structure

```
trader/
├── pyproject.toml
├── .env.example
├── src/trader/
│   ├── config.py              # TradingConfig dataclass (all params from env)
│   ├── data/
│   │   ├── universe.py        # S&P 500 universe fetch + filtering
│   │   └── bars.py            # OHLCV download + parquet cache
│   ├── signals/
│   │   ├── base.py            # Signal protocol / base class
│   │   └── mean_reversion.py  # Entry & exit signal computation
│   ├── risk/
│   │   ├── sizing.py          # Position sizing (20%/pos, 5 max, 10% cash floor)
│   │   └── filters.py        # Risk filters (earnings gate, regime gate)
│   ├── backtest/
│   │   ├── engine.py          # vectorbt portfolio construction
│   │   ├── walk_forward.py    # Rolling IS/OOS walk-forward
│   │   └── metrics.py         # Metrics extraction + SPY benchmark
│   └── reports/
│       └── equity_curve.py    # Equity curve + drawdown chart
├── scripts/
│   └── run_backtest.py        # Entry point
├── tests/
│   └── test_signals.py
└── data/                      # gitignored — parquet cache
```

## Strategy Parameters (v0.2)

All loaded from `.env` / environment variables. See `.env.example` for full list.

| Parameter | Default | Description |
|-----------|---------|-------------|
| Capital | $25,000 | Paper trading capital |
| Position size | 20% | Per-position allocation |
| Max positions | 5 | Concurrent positions |
| Cash floor | 10% | Minimum cash reserve |
| Trend filter | Price > 200d SMA | Long-term trend |
| Pullback | ≥5% from 10d high | Entry condition |
| RSI | < 30 (14-period) | Oversold |
| Volume | ≥1.5× 20d avg | Capitulation |
| Profit target | +10% | Exit condition |
| Stop loss | -5% | Exit condition |
| Time stop | 10 trading days | Exit condition |
| Slippage | 5 bps/side | Cost model |

## Docker Deployment

The trader runs as a containerized cron job alongside the health-mcp stack.
See the [health-mcp repo](https://github.com/avneet/health-mcp) for the full
`docker-compose.yml`.

**What runs in the container:**

| Schedule (ET)         | Command                                       |
|-----------------------|-----------------------------------------------|
| 4:30 PM Mon–Fri       | `run_trading.py` — full EOD run               |
| 10:30–15:30 Mon–Fri   | `run_trading.py --monitor-only` — intraday    |

**Required files on the host** (mounted at `/data` inside the container):

| File / dir                     | Purpose                         |
|--------------------------------|---------------------------------|
| `alpaca.env`                   | Alpaca API credentials          |
| `trader.db`                    | SQLite decision log (auto-created) |
| `daily_reports/`               | Markdown daily reports (auto-created) |
| `task_logs/`                   | Cron job logs (auto-created)    |

**Environment variables** (set in docker-compose, can also be used locally):

| Variable          | Docker default        | Description                        |
|-------------------|-----------------------|------------------------------------|
| `ALPACA_ENV_PATH` | `/data/alpaca.env`    | Path to Alpaca credentials file    |
| `DB_PATH`         | `/data/trader.db`     | Path to SQLite decision log        |
| `REPORTS_DIR`     | `/data/daily_reports` | Directory for markdown reports     |

**Build and run (from the health-mcp repo):**

```bash
docker compose up -d --build trader
docker compose logs -f trader

# Manual trigger (dry-run — no orders)
docker compose exec trader uv run python scripts/run_trading.py --dry-run

# Account status check
docker compose exec trader uv run python scripts/run_trading.py --status

# Verify timezone is ET
docker compose exec trader date
```

**Task Scheduler cleanup** — once Docker is verified, remove the old scheduled tasks:

```powershell
schtasks /Delete /TN "TraderEOD" /F
schtasks /Delete /TN "TraderMonitor_1030" /F
schtasks /Delete /TN "TraderMonitor_1130" /F
schtasks /Delete /TN "TraderMonitor_1230" /F
schtasks /Delete /TN "TraderMonitor_1330" /F
schtasks /Delete /TN "TraderMonitor_1430" /F
schtasks /Delete /TN "TraderMonitor_1530" /F
```

## Running Tests

```bash
uv run pytest tests/ -v
```
