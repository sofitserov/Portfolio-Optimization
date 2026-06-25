# Portfolio-Optimization

## Introduction

This project explores portfolio optimization strategies for a fixed set of
six Vanguard mutual funds:

`VCADX`, `VSIAX`, `VTCLX`, `VTIAX`, `VTSAX`, `VUIAX`

Using ~10 years of daily NAV history (2015–2026, pulled via `yfinance`), it
compares three approaches to choosing portfolio weights:

1. **Mean-variance optimization** (`frontier.py`) — classic Markowitz:
   maximize Sharpe ratio subject to a per-fund weight cap, and trace the
   efficient frontier.
2. **Correlation-minimization rebalancing** (`backtest.py`) — a walk-forward
   strategy that periodically rebalances into whichever funds were least
   correlated over the trailing window.
3. **Correlation-minimization + mean-reversion** (`backtest.py`) — adds a
   penalty that pushes away from funds currently outperforming their
   own historical average, and toward funds lagging it.

Both walk-forward strategies are backtested out-of-sample against a static
equal-weight benchmark. `frontier.py` also tracks how correlation between
these funds changes over time.

## Files

- `frontier.py` — downloads price data, runs max-Sharpe optimization with a
  weight cap, plots the efficient frontier, and plots a sliding 60-day
  pairwise correlation heatmap.
- `backtest.py` — walk-forward backtest for the correlation-minimization and
  correlation+mean-reversion strategies; prints/plots weight history and
  compares performance against an equal-weight benchmark.

## Findings

- **Uncapped max-Sharpe** concentrates 100% into one fund (`VTCLX`) — a known
  failure since the Sharpe objective is highly sensitive to noisy
  return estimates. A 35% per-fund cap fixes this (`VTCLX`/`VTSAX` 35%,
  `VUIAX` 30%) at a Sharpe cost (~0.48 vs. ~0.52 uncapped).
- **Correlation Changes** — pairs involving `VCADX` swing between
  negative and positive, while the 5 equity funds stay highly 
  correlated with each other, spiking to ~1.0 during the 2020 crash.
- **Pure correlation-minimization underperforms.** It consistently
  overweights `VCADX` (least-correlated, but also lowest-return) and trails
  the equal-weight benchmark (6.6% vs. 10.1% return, Sharpe 0.19 vs. 0.40).
  Minimizing correlation alone has no return signal to push back with.
- **Adding mean-reversion helps, but the effect is sensitive to `λ`.** Sweeping
  the reversion penalty weight (`reversion_lambda`) shows performance rising
  with `λ`, peaking around `λ≈50–100` (11.3% return, Sharpe ~0.49, beating
  the benchmark), then degrading as `λ` grows further and the correlation
  term gets drowned out entirely.
- **Removing `VCADX` from the universe** closes most of the gap with the
  benchmark (9.85% vs. 11.59% return) but raises volatility and drawdown a
  lot (11%→16% vol, -30%→-38% drawdown) — `VCADX` was acting as a low-vol
  anchor, not just a drag on returns.

## Notes

- Expected returns are estimated from historical means, which are noisy and
  not a reliable predictor of future returns — small changes in the input
  data can swing either strategy's allocation significantly.
- **`λ≈50–100` was chosen against the full 11-year backtest
  window — the same data used to evaluate it.** With only ~46 rebalances,
  this risks overfitting. Next step: pick `λ` on an in-sample/train period
  and validate on a held-out/walk-forward period before trusting it.
- The risk-free rate (4.5%) is a placeholder and should be replaced with a
  rate matched to the historical period being analyzed.
- All 6 funds are broad equity index funds aside from `VCADX`, so they are
  likely highly correlated with each other; diversification benefit here is
  limited compared to mixing in bonds or other asset classes.
