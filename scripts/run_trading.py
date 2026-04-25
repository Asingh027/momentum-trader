#!/usr/bin/env python
"""Live trading CLI entry point.

Usage:
    uv run python scripts/run_trading.py                    # full daily run (4:30 PM ET)
    uv run python scripts/run_trading.py --dry-run          # signals only, no orders placed
    uv run python scripts/run_trading.py --status           # print account + positions, exit
    uv run python scripts/run_trading.py --monitor-only     # intraday risk check (hourly)
    uv run python scripts/run_trading.py --monitor-only --dry-run  # monitor without orders

Modes:
    (default)        Full EOD run: exits + entries, daily report written.
    --monitor-only   Risk-only: checks hard stops and regime gate on open positions.
                     Never originates entries. Silent if all positions are safe.
    --status         Account snapshot only; exits immediately.

Credentials are read from:
    C:\\Users\\Avneet\\Documents\\Trading Helper\\alpaca.env  (paper, default)
    C:\\Users\\Avneet\\Documents\\Trading Helper\\alpaca_live.env  (live, with --live)
Override with: --env-file /path/to/other.env
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is on path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Momentum v1.0 live trading runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute signals and log decisions, but do NOT place any orders.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current account status and positions, then exit.",
    )
    parser.add_argument(
        "--monitor-only", "--intraday",
        action="store_true",
        dest="monitor_only",
        help=(
            "Intraday risk-only mode. Checks hard stops and regime gate on open "
            "positions. Never originates entries. Exits silently if all clear."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Trade against the LIVE Alpaca account (api.alpaca.markets). "
            "Default is paper (paper-api.alpaca.markets). "
            "Loads alpaca_live.env unless --env-file overrides it."
        ),
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to Alpaca credentials env file (default: Trading Helper/alpaca.env).",
    )
    args = parser.parse_args()

    # Resolve env file: explicit --env-file wins; else live vs paper default
    env_file = args.env_file
    if env_file is None and args.live:
        env_file = Path(r"C:\Users\Avneet\Documents\Trading Helper\alpaca_live.env")

    if args.monitor_only:
        from trader.monitor import run_intraday_monitor
        run_intraday_monitor(dry_run=args.dry_run, env_path=env_file, live=args.live)
    else:
        from trader.runner import run_daily
        run_daily(
            dry_run=args.dry_run,
            env_path=env_file,
            status_only=args.status,
            live=args.live,
        )


if __name__ == "__main__":
    main()
