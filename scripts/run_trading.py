#!/usr/bin/env python
"""Live trading CLI entry point.

Usage:
    uv run python scripts/run_trading.py               # full daily run
    uv run python scripts/run_trading.py --dry-run     # signals only, no orders placed
    uv run python scripts/run_trading.py --status      # print account + positions, exit

The --dry-run flag is critical for the first paper week. Use it to verify
what the bot WOULD do before enabling live order placement.

Credentials are read from:
    C:\\Users\\Avneet\\Documents\\Trading Helper\\alpaca.env
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
        "--env-file",
        type=Path,
        default=None,
        help="Path to Alpaca credentials env file (default: Trading Helper/alpaca.env).",
    )
    args = parser.parse_args()

    from trader.runner import run_daily

    run_daily(
        dry_run=args.dry_run,
        env_path=args.env_file,
        status_only=args.status,
    )


if __name__ == "__main__":
    main()
