# Strategy v1.0 Backtest Report
**Universe:** 84 S&P 500 large-caps (same as prior phases)
**Window:** 2021-04-01 -> 2026-04-01 | **Capital:** $25,000
**Entry:** Price > 126d high, Price > 50d SMA, Golden Cross (50>200d), RS vs SPY >= +5% over 63d, SPY > 200d SMA
**Exit:** Price < 50d SMA (PRIMARY trailing stop), -8% hard stop, regime stop
**Costs:** $0 commission + 5 bps slippage/side | Volume filter: OFF | Regime gate: ON

---

## Full-Window Metrics

| Metric | Strategy | SPY B&H |
| --- | --- | --- |
| Total Return | 183.8% | 73.8% |
| CAGR | 23.2% | 11.7% |
| Sharpe Ratio | 2.09 | 0.74 |
| Sortino Ratio | 2.61 | 1.01 |
| Max Drawdown | -6.6% | -24.5% |
| Win Rate | 60.6% | — |
| Avg Win | 11.0% | — |
| Avg Loss | -3.2% | — |
| Trade Count | 404 | — |
| Avg Hold (days) | 47.6 | — |

---

## Exit Rule Breakdown

| Exit Reason | % of Trades | Count |
| --- | --- | --- |
| trailing_sma | 71.3% | 288 |
| regime_stop | 23.3% | 94 |
| hard_stop | 3.5% | 14 |
| signal_exit_other | 2.0% | 8 |

---

## Per-Ticker Summary

| Ticker | Trades | Total P&L | Avg Return | Win% |
| --- | --- | --- | --- | --- |
| HD | 7 | $-269.62 | -1.5% | 28.6% |
| LOW | 6 | $-161.78 | -2.1% | 33.3% |
| MDT | 4 | $-117.42 | -0.7% | 25.0% |
| PNC | 5 | $-108.30 | 1.6% | 60.0% |
| HON | 2 | $-78.81 | -2.3% | 50.0% |
| PFE | 2 | $-55.68 | -1.3% | 50.0% |
| VZ | 4 | $-39.59 | -0.1% | 50.0% |
| WFC | 5 | $-32.52 | -0.5% | 40.0% |
| TXN | 2 | $-31.21 | -3.9% | 50.0% |
| HPQ | 4 | $-28.93 | 1.8% | 25.0% |
| PG | 2 | $-27.35 | -1.8% | 0.0% |
| ORCL | 9 | $-26.75 | 2.5% | 44.4% |
| UPS | 3 | $5.16 | -1.3% | 33.3% |
| ECL | 7 | $15.97 | 0.7% | 57.1% |
| SBUX | 3 | $31.27 | 0.1% | 33.3% |
| DIS | 2 | $34.79 | 2.6% | 50.0% |
| CMCSA | 1 | $45.56 | 4.7% | 100.0% |
| PEP | 3 | $48.40 | 1.0% | 66.7% |
| TGT | 2 | $58.13 | 3.7% | 50.0% |
| PLD | 2 | $59.61 | 2.9% | 50.0% |
| LIN | 4 | $63.77 | -0.3% | 50.0% |
| ABT | 4 | $69.98 | 1.7% | 50.0% |
| KLAC | 6 | $85.60 | 1.2% | 50.0% |
| MCD | 4 | $85.90 | -0.2% | 50.0% |
| MO | 7 | $87.48 | 2.4% | 42.9% |
| UNH | 1 | $107.60 | 2.0% | 100.0% |
| AMT | 2 | $115.46 | 2.5% | 50.0% |
| USB | 3 | $119.47 | 2.7% | 66.7% |
| CL | 4 | $165.95 | 2.9% | 50.0% |
| MS | 9 | $192.53 | 2.0% | 66.7% |
| DHR | 3 | $202.84 | 1.4% | 100.0% |
| KO | 5 | $217.92 | 2.2% | 100.0% |
| DUK | 5 | $219.31 | 2.5% | 40.0% |
| NEE | 4 | $244.26 | 4.1% | 100.0% |
| APD | 4 | $266.78 | 2.5% | 50.0% |
| T | 2 | $266.92 | 9.9% | 100.0% |
| GS | 3 | $276.45 | 2.7% | 66.7% |
| DE | 5 | $279.44 | 3.6% | 100.0% |
| CAT | 6 | $309.25 | 15.6% | 66.7% |
| SLB | 4 | $329.14 | 5.2% | 100.0% |
| QCOM | 2 | $329.48 | 20.2% | 100.0% |
| AMGN | 7 | $342.29 | 2.3% | 71.4% |
| ADBE | 1 | $367.57 | 30.8% | 100.0% |
| AAPL | 5 | $371.70 | 3.0% | 40.0% |
| BAC | 10 | $372.14 | -0.0% | 30.0% |
| AMD | 5 | $397.85 | 10.5% | 80.0% |
| SO | 6 | $431.20 | 3.5% | 50.0% |
| MRK | 3 | $465.61 | 7.5% | 33.3% |
| JPM | 8 | $493.31 | 4.4% | 50.0% |
| CVX | 6 | $527.15 | 4.0% | 66.7% |
| GM | 6 | $570.09 | 9.1% | 83.3% |
| AXP | 10 | $614.25 | 4.8% | 70.0% |
| AMZN | 6 | $618.11 | 6.8% | 50.0% |
| LMT | 4 | $657.93 | 3.9% | 25.0% |
| JNJ | 3 | $678.04 | 14.3% | 66.7% |
| TSLA | 5 | $679.89 | 3.5% | 40.0% |
| COP | 6 | $728.86 | 6.4% | 83.3% |
| LLY | 3 | $771.33 | 10.3% | 100.0% |
| NFLX | 9 | $791.13 | 4.5% | 55.6% |
| BMY | 6 | $793.72 | 5.5% | 83.3% |
| GILD | 7 | $794.29 | 6.3% | 71.4% |
| INTC | 6 | $830.03 | 6.9% | 66.7% |
| IBM | 8 | $896.05 | 5.2% | 62.5% |
| MSFT | 3 | $1,091.44 | 16.6% | 100.0% |
| ABBV | 10 | $1,140.31 | 3.1% | 80.0% |
| XOM | 7 | $1,201.40 | 4.9% | 71.4% |
| CRM | 3 | $1,230.83 | 16.7% | 100.0% |
| RTX | 9 | $1,291.19 | 6.2% | 77.8% |
| BA | 7 | $1,388.92 | -0.6% | 42.9% |
| NVDA | 9 | $1,551.95 | 18.5% | 77.8% |
| NOW | 5 | $1,632.27 | 7.5% | 40.0% |
| WMT | 7 | $1,635.54 | 5.8% | 42.9% |
| GOOGL | 7 | $1,706.48 | 9.5% | 71.4% |
| AVGO | 12 | $2,104.70 | 9.5% | 58.3% |
| AMAT | 11 | $2,270.92 | 10.4% | 45.5% |
| META | 5 | $2,914.32 | 19.4% | 80.0% |
| LRCX | 7 | $3,349.73 | 19.8% | 85.7% |
| GE | 10 | $4,887.20 | 15.0% | 70.0% |

---

## Walk-Forward Results (IS = 2yr / OOS = 6mo / Step = 3mo)

| Window | Period | Total Return | CAGR | Sharpe | Sortino | Max DD | Trades |
|--------|--------|-------------|------|--------|---------|--------|--------|
| 0 | IS 2021-04-01–2023-04-01 | 18.0% | 8.6% | 1.31 | 0.97 | -5.4% | 88 |
| 0 | OOS 2023-04-01–2023-10-01 | 17.1% | 37.9% | 2.37 | 4.47 | -4.8% | 58 |
| 1 | IS 2021-07-01–2023-07-01 | 38.3% | 17.6% | 1.85 | 1.92 | -5.4% | 114 |
| 1 | OOS 2023-07-01–2024-01-01 | 7.1% | 15.0% | 1.05 | 1.29 | -6.3% | 66 |
| 2 | IS 2021-10-01–2023-10-01 | 39.6% | 18.2% | 1.70 | 1.95 | -5.4% | 137 |
| 2 | OOS 2023-10-01–2024-04-01 | 20.8% | 46.2% | 3.01 | 4.64 | -4.0% | 56 |
| 3 | IS 2022-01-01–2024-01-01 | 50.1% | 22.7% | 1.94 | 2.43 | -6.6% | 166 |
| 3 | OOS 2024-01-01–2024-07-01 | 16.4% | 35.8% | 2.99 | 5.04 | -3.7% | 75 |
| 4 | IS 2022-04-01–2024-04-01 | 60.4% | 26.6% | 2.09 | 2.78 | -6.6% | 165 |
| 4 | OOS 2024-04-01–2024-10-01 | 10.8% | 22.6% | 2.27 | 3.03 | -3.1% | 75 |
| 5 | IS 2022-07-01–2024-07-01 | 62.1% | 27.3% | 2.07 | 3.00 | -6.6% | 188 |
| 5 | OOS 2024-07-01–2025-01-01 | 9.8% | 20.6% | 1.66 | 1.79 | -6.0% | 69 |
| 6 | IS 2022-10-01–2024-10-01 | 72.2% | 31.3% | 2.26 | 3.38 | -6.6% | 221 |
| 6 | OOS 2024-10-01–2025-04-01 | 5.6% | 11.7% | 1.01 | 1.46 | -4.4% | 70 |
| 7 | IS 2023-01-01–2025-01-01 | 82.3% | 35.1% | 2.38 | 3.70 | -6.6% | 235 |
| 7 | OOS 2025-01-01–2025-07-01 | 8.6% | 18.3% | 1.94 | 2.22 | -3.4% | 44 |
| 8 | IS 2023-04-01–2025-04-01 | 70.0% | 30.4% | 2.33 | 3.58 | -6.4% | 230 |
| 8 | OOS 2025-04-01–2025-10-01 | 22.5% | 49.9% | 4.15 | 6.27 | -2.7% | 38 |
| 9 | IS 2023-07-01–2025-07-01 | 56.4% | 25.1% | 1.99 | 2.56 | -6.3% | 223 |
| 9 | OOS 2025-07-01–2026-01-01 | 13.8% | 29.4% | 2.14 | 3.40 | -3.6% | 64 |
| 10 | IS 2023-10-01–2025-10-01 | 79.0% | 33.8% | 2.78 | 3.92 | -4.9% | 211 |
| 10 | OOS 2025-10-01–2026-04-01 | 19.4% | 42.9% | 2.10 | 3.12 | -6.6% | 74 |

---

## Honest Assessment

- Trade count 404 >= 100 minimum. Statistics are meaningful.
- Sharpe 2.09 — above 0.50 viable threshold.
- Max drawdown -6.6% is within the -20% spec limit.
- Absolute return 183.8% vs SPY 73.8% (+110.0%).
- Win rate 60.6%, avg win 11.0%, avg loss -3.2%.
- Walk-forward OOS: 11/11 windows positive Sharpe (100%). Edge appears consistent OOS.
---

## Paper-Trading Recommendation

**Ready for paper trading consideration.** OOS walk-forward shows consistent positive Sharpe. Run minimum 3 months paper before any live capital.

---

## v1.0 vs v0.3 Design Changes (Summary)

| Parameter | v0.3 | v1.0 | Rationale |
|-----------|------|------|-----------|
| Strategy family | Mean reversion dip-buy | Momentum / trend-following | Mean reversion had negative expectancy in 2021-2026 bull run |
| Universe | 11 sector ETFs | 84 S&P 500 large-caps | More signals; stock-level momentum cleaner than sector ETFs |
| Entry trigger | RSI < 35 oversold | Price > 126d high (breakout) | Buy strength, not weakness |
| Trend filter | Price > 200d SMA | Golden cross (50d > 200d) + Price > 200d + Price > 50d | Multi-confirm uptrend |
| RS filter | None | 63d return outperforms SPY by >= 5pp | Only enter strongest relative performers |
| Primary exit | RSI > 50 crossover | Price < 50d SMA (trailing) | Ride the trend; exit only when trend breaks |
| Profit target | +8% | None | Momentum runs can exceed 8%; cutting winners was a bug |
| Stop loss | -5% | -8% hard stop | Trend strategies need wider stops |
| Time stop | 15 days | None | Good trends last months, not weeks |

---

## Known Limitations

- **Survivorship bias:** Universe uses current S&P 500 constituents. Stocks that were removed (bankruptcy, acquisition, de-listing) between 2021-2026 are excluded. Momentum strategies are particularly vulnerable — failed breakouts in delisted stocks are invisible.
- **Lookahead in universe:** 2021 entry uses stocks that are in the S&P 500 in 2026.
- **Stop execution:** sl_stop applied at bar close by vectorbt. Real gaps can blow through hard stops.
- **2021-2026 window:** Predominantly bullish with two sharp corrections (2022 bear, 2025 correction). Momentum strategies should outperform in this regime.
- **Golden cross lag:** 50d/200d golden cross is a lagging signal. Many large moves have already started when it fires.

---
*Generated by v1.0 backtest harness*
