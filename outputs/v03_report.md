    # Strategy v0.3 Backtest Report
    **Universe:** 11 SPDR Sector ETFs — XLK, XLF, XLE, XLV, XLI, XLP, XLY, XLU, XLB, XLRE, XLC
    **Window:** 2021-04-01 -> 2026-04-01 | **Capital:** $25,000
    **Entry:** Price > 200d SMA, RSI(14) < 35, pullback >= 5% from 10d high, SPY > 200d SMA
    **Exit:** RSI(14) crosses above 50 (PRIMARY), +8% target, -5% stop, 15d time stop, regime stop
    **Costs:** $0 commission + 5 bps slippage/side | Volume filter: OFF | Regime gate: ON

    ---

    ## Full-Window Metrics

    | Metric | Strategy | SPY B&H |
| --- | --- | --- |
| Total Return | -6.1% | 73.8% |
| CAGR | -1.2% | 11.7% |
| Sharpe Ratio | -0.61 | 0.74 |
| Sortino Ratio | -0.25 | 1.01 |
| Max Drawdown | -6.6% | -24.5% |
| Win Rate | 35.7% | — |
| Avg Win | 1.6% | — |
| Avg Loss | -2.9% | — |
| Trade Count | 28 | — |
| Avg Hold (days) | 12.0 | — |

    ---

    ## Exit Rule Breakdown

    | Exit Reason | % of Trades | Count |
|-------------|-------------|-------|
| rsi_crossover | 42.9% | 12 |
| regime_stop | 21.4% | 6 |
| time_stop | 17.9% | 5 |
| stop_loss | 14.3% | 4 |
| signal_exit_other | 3.6% | 1 |

    ---

    ## Per-ETF Summary

    | Ticker | Trades | Total P&L | Avg Return | Win% |
|--------|--------|-----------|------------|------|
| XLC | 1 | $102.99 | 2.6% | 100.0% |
| XLP | 1 | $63.11 | 1.3% | 100.0% |
| XLF | 2 | $40.25 | 0.6% | 50.0% |
| XLU | 2 | $-48.80 | -0.6% | 0.0% |
| XLB | 1 | $-135.94 | -4.5% | 0.0% |
| XLV | 2 | $-152.79 | -1.6% | 50.0% |
| XLRE | 2 | $-186.00 | -3.6% | 0.0% |
| XLI | 3 | $-224.15 | -1.9% | 0.0% |
| XLY | 6 | $-455.01 | -1.4% | 33.3% |
| XLK | 8 | $-522.17 | -1.2% | 50.0% |

    ---

    ## Walk-Forward Results (IS = 2yr / OOS = 6mo / Step = 3mo)

    | Window | Period | Total Return | CAGR | Sharpe | Sortino | Max DD | Trades |
|--------|--------|-------------|------|--------|---------|--------|--------|
| 0 | IS 2021-04-01–2023-04-01 | -1.5% | -0.7% | -0.89 | -0.15 | -1.5% | 4 |
| 0 | OOS 2023-04-01–2023-10-01 | -0.0% | -0.0% | -0.01 | -0.00 | -1.1% | 5 |
| 1 | IS 2021-07-01–2023-07-01 | -1.2% | -0.5% | -0.65 | -0.13 | -1.6% | 5 |
| 1 | OOS 2023-07-01–2024-01-01 | -0.2% | -0.2% | -0.11 | -0.04 | -1.2% | 4 |
| 2 | IS 2021-10-01–2023-10-01 | -1.5% | -0.6% | -0.54 | -0.16 | -2.0% | 9 |
| 2 | OOS 2023-10-01–2024-04-01 | -0.3% | -0.2% | -0.17 | -0.04 | -0.9% | 2 |
| 3 | IS 2022-01-01–2024-01-01 | -1.4% | -0.5% | -0.42 | -0.13 | -2.1% | 9 |
| 3 | OOS 2024-01-01–2024-07-01 | 1.0% | 0.8% | 0.94 | 0.63 | -0.6% | 4 |
| 4 | IS 2022-04-01–2024-04-01 | 0.0% | 0.0% | 0.02 | 0.01 | -1.2% | 5 |
| 4 | OOS 2024-04-01–2024-10-01 | -0.2% | -0.2% | -0.09 | -0.04 | -2.0% | 6 |
| 5 | IS 2022-07-01–2024-07-01 | 1.1% | 0.4% | 0.34 | 0.13 | -1.2% | 9 |
| 5 | OOS 2024-07-01–2025-01-01 | -2.3% | -1.8% | -0.76 | -0.22 | -3.8% | 6 |
| 6 | IS 2022-10-01–2024-10-01 | -0.2% | -0.1% | -0.04 | -0.01 | -2.0% | 11 |
| 6 | OOS 2024-10-01–2025-04-01 | -3.2% | -2.4% | -1.09 | -0.38 | -3.2% | 6 |
| 7 | IS 2023-01-01–2025-01-01 | -1.3% | -0.5% | -0.22 | -0.08 | -3.8% | 15 |
| 7 | OOS 2025-01-01–2025-07-01 | -0.6% | -0.5% | -0.28 | -0.17 | -1.8% | 6 |
| 8 | IS 2023-04-01–2025-04-01 | -3.4% | -1.2% | -0.56 | -0.25 | -5.1% | 17 |
| 8 | OOS 2025-04-01–2025-10-01 | 0.0% | 0.0% | nan | nan | 0.0% | 0 |
| 9 | IS 2023-07-01–2025-07-01 | -3.6% | -1.3% | -0.60 | -0.26 | -5.1% | 16 |
| 9 | OOS 2025-07-01–2026-01-01 | -0.1% | -0.1% | -0.04 | -0.01 | -1.3% | 3 |
| 10 | IS 2023-10-01–2025-10-01 | -3.7% | -1.3% | -0.63 | -0.25 | -5.1% | 14 |
| 10 | OOS 2025-10-01–2026-04-01 | -1.1% | -0.8% | -0.42 | -0.18 | -1.8% | 7 |

    ---

    ## Honest Assessment

    - **TRADE COUNT 28 < 100 minimum.** Statistics are unreliable. Only 11 ETFs in the universe — this is a signal-frequency problem inherent to the universe size.
- **Sharpe -0.61 — negative.** Risk-adjusted return is worse than cash.
- Max drawdown -6.6% is within the -20% spec limit.
- **Absolute return -6.1% vs SPY 73.8% (-79.9%).** Significant underperformance on raw return.
- RSI crossover exit fired on 42.9% of trades — it is the dominant exit mechanism, confirming the redesign is working as intended.
- Walk-forward OOS: 1/10 windows positive Sharpe. **Edge is inconsistent OOS — likely regime-dependent or insufficient sample size.**
- **v0.4 note (not implemented):** Sector ETFs are correlated. Simultaneous XLK + XLC entries double tech/comms exposure beyond the 20%/position limit. A correlation cap or sector-level position limit should be added before paper trading.

    ---

    ## Paper-Trading Recommendation

    **Not yet ready for paper trading.** The strategy needs either a larger universe (more ETFs, or back to equities with looser filters) or a longer observation window to build statistical confidence. Do not promote until OOS walk-forward shows consistent positive Sharpe across at least 60% of windows.

    ---

    ## v0.3 vs v0.2 Design Changes (Summary)

    | Parameter | v0.2 | v0.3 | Rationale |
    |-----------|------|------|-----------|
    | Universe | 84 S&P 500 stocks | 11 sector ETFs | ETFs revert faster; no earnings gap risk |
    | RSI threshold | < 30 | < 35 | Sweep showed < 30 generates < 30 trades in 5 years |
    | Volume filter | on (1.5x avg) | off | Sweep: kills signals, zero quality improvement |
    | Primary exit | +10% profit target | RSI > 50 crossover | Signal-matched exit — exits when mean-reversion completes |
    | Backup profit target | +10% | +8% | Avg wins in sweep were ~4.5%; 10% was rarely hit |
    | Time stop | 10 days | 15 days | ETFs need more room to complete the reversion |

    ---

    ## Known Limitations

    - **Survivorship bias (mild):** All 11 ETFs exist through the full window; no delisting risk.
    - **Sector correlation (v0.4 to-do):** Simultaneous entries in correlated sectors double-count risk.
    - **Thin sample:** 11-ETF universe limits trade count. Statistical conclusions require caution.
    - **Stop execution:** sl_stop applied at bar close by vectorbt. Real gaps can blow through stops.
    - **2021-2026 window:** Predominantly bullish. Mean reversion strategies are more likely to
      outperform in range-bound or volatility-rich markets.

    ---
    *Generated by v0.3 backtest harness*
