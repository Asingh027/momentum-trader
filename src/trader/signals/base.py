"""Base signal protocol.

All signal classes must implement compute_entries and compute_exits,
returning boolean Series aligned to the price index.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class SignalProtocol(Protocol):
    def compute_entries(self, bars: dict[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
        """Return boolean DataFrame (dates × tickers) — True = entry signal on that date."""
        ...

    def compute_exits(self, bars: dict[str, pd.DataFrame], entries: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Return boolean DataFrame (dates × tickers) — True = exit signal on that date."""
        ...
