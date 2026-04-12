"""Daily report writer.

v1: writes markdown to C:\\Users\\Avneet\\Documents\\Trading Helper\\daily_reports\\{date}.md
Phase 3 will add Telegram/email.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_REPORTS_DIR = Path(r"C:\Users\Avneet\Documents\Trading Helper\daily_reports")


def get_reports_dir() -> Path:
    import os
    env = os.environ.get("REPORTS_DIR", "")
    return Path(env) if env.strip() else _DEFAULT_REPORTS_DIR


def write_daily_report(
    report_date: Optional[date] = None,
    portfolio_value: float = 0.0,
    portfolio_pct_change: float = 0.0,
    cash: float = 0.0,
    positions: Optional[list[dict]] = None,
    actions: Optional[list[dict]] = None,
    daily_pnl_pct: float = 0.0,
    drawdown_pct: float = 0.0,
    pdt_count: int = 0,
    kill_switch_status: str = "OK",
    dry_run: bool = False,
) -> Path:
    """Write the daily trading report to the reports directory.

    Returns the path of the written file.
    """
    if report_date is None:
        report_date = date.today()

    positions = positions or []
    actions = actions or []

    # Format positions table
    pos_lines = []
    for p in positions:
        symbol = p.get("symbol", "")
        shares = p.get("qty", 0.0)
        entry = p.get("avg_entry_price", 0.0)
        current = p.get("current_price", 0.0)
        upnl = p.get("unrealized_pl", 0.0)
        upnl_pct = p.get("unrealized_plpc", 0.0)
        days = p.get("days_held", 0)
        pos_lines.append(
            f"- **{symbol}** | {shares:.4f} shares @ ${entry:.2f} | "
            f"Current ${current:.2f} | {'+'if upnl>=0 else ''}{upnl_pct*100:.1f}% (${upnl:+.2f}) | {days}d held"
        )
    positions_block = "\n".join(pos_lines) if pos_lines else "_No open positions._"

    # Format actions
    action_lines = []
    for a in actions:
        action = a.get("action", "")
        ticker = a.get("ticker", "")
        reason = a.get("reason", "")
        suffix = " [DRY RUN]" if a.get("dry_run") else ""
        action_lines.append(f"- **{action.upper()} {ticker}** — {reason}{suffix}")
    actions_block = "\n".join(action_lines) if action_lines else "_No actions today._"

    # Risk status
    kill_icon = "OK" if kill_switch_status == "OK" else f"TRIPPED: {kill_switch_status}"
    dry_run_notice = "\n> **DRY RUN MODE** — No orders were placed.\n" if dry_run else ""

    content = f"""\
## Trading Report — {report_date.strftime('%A, %B %d, %Y')}
{dry_run_notice}
**Portfolio:** ${portfolio_value:,.2f} ({portfolio_pct_change:+.2f}% today)
**Cash:** ${cash:,.2f}

### Positions ({len(positions)})
{positions_block}

### Today's Actions
{actions_block}

### Risk Status
- Daily P&L: {daily_pnl_pct:+.2f}%
- Drawdown from peak: {drawdown_pct:.2f}%
- PDT count (rolling 5d): {pdt_count}/3
- Kill switches: {kill_icon}
"""

    reports_dir = get_reports_dir()
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / f"{report_date.isoformat()}.md"
    output_path.write_text(content, encoding="utf-8")
    logger.info("Daily report written to %s", output_path)
    return output_path
