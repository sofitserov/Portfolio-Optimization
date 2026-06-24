# Portfolio-Optimization

## Introduction

This project explores portfolio optimization strategies for a fixed set of
six Vanguard mutual funds:

`VCADX`, `VSIAX`, `VTCLX`, `VTIAX`, `VTSAX`, `VUIAX`

Using ~10 years of daily NAV history (2015–2026, pulled via `yfinance`), it
compares two different approaches to choosing portfolio weights:

1. **Mean-variance optimization** (`main.py`) — the classic Markowitz
   approach: maximize the Sharpe ratio subject to a per-fund weight cap, and
   trace the resulting efficient frontier.
2. **Correlation-minimization rebalancing** (`backtest.py`) — a walk-forward
   strategy that periodically rebalances into whichever funds were least
   correlated with each other over the trailing window, tested out-of-sample
   against a static equal-weight benchmark.

Along the way, the project also looks at how correlation between these funds
changes over time, since both strategies depend on that assumption holding
reasonably steady.

## Files

- `main.py` — downloads price data, computes returns/expected
  returns/covariance, runs max-Sharpe optimization with a weight cap, plots
  the efficient frontier, and plots a rolling 60-day pairwise correlation
  heatmap.
- `backtest.py` — implements the walk-forward correlation-minimization
  backtest, prints/plots the weight history over time, and compares
  performance against an equal-weight benchmark.

## Findings

**Mean-variance optimization (`main.py`):** an uncapped max-Sharpe
optimization pushed 100% of the portfolio into a single fund (`VTCLX`). This
is a known failure mode — the Sharpe ratio objective is highly sensitive to
small errors in historical mean return estimates, so the optimizer chases
whichever fund had the best risk-adjusted historical performance rather than
diversifying. Adding a 35% per-fund weight cap produced a more realistic
allocation (`VTCLX` 35%, `VTSAX` 35%, `VUIAX` 30%, rest 0%), with ~12.5%
expected annual return, ~16.7% volatility, and a Sharpe ratio of ~0.48 (vs.
~0.52 uncapped) — a modest cost for avoiding total concentration in one fund.

**Rolling correlation (`main.py`):** correlation between these funds is not
static. Pairs involving `VCADX` swing between strongly negative and mildly
positive depending on the period, while the 5 equity funds stay consistently
highly correlated with each other, with a visible spike to near-1.0 across
the board during the 2020 COVID crash.

**Correlation-minimization backtest (`backtest.py`):** this strategy
consistently overweighted `VCADX` (the least-correlated, but also
lowest-return, fund) and underperformed the simple equal-weight benchmark
over the full backtest (6.6% vs. 10.1% annualized return, Sharpe 0.19 vs.
0.40), despite lower volatility. This illustrates a key limitation of
minimizing correlation alone: the objective has no return signal, so it can
sacrifice return for diversification that isn't actually compensated. A
future iteration should combine correlation minimization with a return or
Sharpe constraint rather than using it standalone.

## Notes / caveats

- Expected returns are estimated from historical means, which are noisy and
  not a reliable predictor of future returns — small changes in the input
  data can swing either strategy's allocation significantly.
- The risk-free rate (4.5%) is a placeholder and should be replaced with a
  rate matched to the historical period being analyzed.
- All 6 funds are broad equity index funds (aside from `VCADX`), so they are
  likely highly correlated with each other; the diversification benefit here
  is limited compared to mixing in bonds or other asset classes.
