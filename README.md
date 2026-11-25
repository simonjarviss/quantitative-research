# Quantitative Research

**Simon Jarvis** · Portfolio Manager, QUANTT (Queen's University Algorithmic & Network Trading Team)

Research papers and the code behind them, written in 2026 for QUANTT's systematic trading desk.
Each paper is a self-contained PDF. The code that produced the empirical results is in
[`code/`](code/). The data are licensed or cached from vendors and are not included.
Results appear only as aggregate tables and charts in the papers.

## Papers

| Paper | What it covers | Pages |
|---|---|---|
| [Forced Flows at the Index Close](reports/index-rebalancing/index-rebalancing.pdf) | Point-in-time event studies of index rebalances in Canada, Europe and the United States (Russell, S&P/TSX Composite, STOXX 600, the DAX family and MSCI): who must trade, what is capturable after the announcement, capacity, and forward paper forecasts of the 2026 reviews | 16 |
| [Execution Capacity, Market Beta and an Intraday Close Test](reports/cef-desk-studies/cef-desk-studies.pdf) | Three studies for a market-neutral closed-end-fund discount book: the size and forecastability of the closing auction, the market beta of a second sleeve and the hedge chosen, and a leveraged-ETF close test that was set aside | 14 |
| [Delta-Hedged Gamma on SPY and QQQ Options](reports/options-gamma/options-gamma.pdf) | The variance risk premium, dealer gamma and regime signals in the literature, the option data available to a small desk, and a proposed test | 11 |
| [Free Public Data for Systematic Strategies](reports/free-data-sources/free-data-sources.pdf) | A survey of 73 free data sources across rates, credit, commodities, options, events, funds and alternative data: access rules, publication lags, point-in-time hazards and 33 strategy playbooks | 14 |

Earlier work: [Macro Regime-Based Asset Rotation](https://github.com/simonjarviss/macro-regime-rotation/blob/main/report/macro_regime_rotation_report.pdf)
(co-author), a four-regime macro classification and a walk-forward multi-asset rotation.

## Code

| Folder | Contents |
|---|---|
| [`code/index_rebalance_research/12_studies/`](code/index_rebalance_research/12_studies/) | One folder per event study, each with a `run_all.sh` that rebuilds it from the data pulls |
| [`code/index_rebalance_research/13_shadow/`](code/index_rebalance_research/13_shadow/) | Forward paper forecasts of the 2026 Russell, MSCI, S&P/TSX and European reviews |
| [`code/index_rebalance_research/16_flow_engine/`](code/index_rebalance_research/16_flow_engine/) | Net passive flow estimates per rebalance and their validation |
| [`code/index_rebalance_research/14_borrow_data/`](code/index_rebalance_research/14_borrow_data/) | Borrow-data collectors |
| [`code/index_rebalance_research/raw/scripts/`](code/index_rebalance_research/raw/scripts/) | Closing-auction share of European volume from venue samples |
| [`code/desk/`](code/desk/) | The closing-auction, market-beta and leveraged-ETF studies, with their tests |
| [`code/r2/`](code/r2/) | A read-only client for the desk's data warehouse; credentials come from the environment and are not in this repository |

The closed-end-fund studies import the desk's public research harness from
[quanttqueensu/CreditTrading](https://github.com/quanttqueensu/CreditTrading). The papers are
compiled from their LaTeX sources in `reports/` with [Tectonic](https://tectonic-typesetting.github.io/).

## Notes

All backtests are gross of transaction costs unless a paper says otherwise. Paper-account results
are simulated fills at a broker, not live trading. Nothing here is investment advice.

© 2026 Simon Jarvis. All rights reserved. Shared for reading; no licence to reuse the code or text is granted.
