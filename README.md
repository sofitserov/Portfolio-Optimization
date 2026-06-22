# Portfolio-Optimization

## Introduction

This project applies basic mean-variance portfolio optimization (Markowitz theory)
to a set of six Vanguard mutual funds:

`VCADX`, `VSIAX`, `VTCLX`, `VTIAX`, `VTSAX`, `VUIAX`

Using ~10 years of daily price history (2015-01-01 to 2025-01-01), the script:

1. Computes daily returns and annualizes the expected return and covariance
   for each fund.
2. Solves for the portfolio weights that maximize the Sharpe ratio
   (`(return - risk_free_rate) / volatility`), assuming no shorting and a
   per-fund weight cap.
3. Traces the efficient frontier (minimum volatility for a range of target
   returns) and plots it alongside the individual funds and the optimal
   portfolio.

## Findings

With an uncapped max-Sharpe optimization, the solver pushed 100% of the
portfolio into a single fund (`VTCLX`). This is a known failure mode of
mean-variance optimization: the Sharpe ratio objective is highly sensitive to
small errors in the historical mean return estimates, so the optimizer
chases whichever fund had the best risk-adjusted historical performance
rather than producing a diversified allocation.

Adding a 35% per-fund weight cap produced a more diversified, realistic
allocation:

| Fund   | Weight |
|--------|--------|
| VCADX  | 0.00   |
| VSIAX  | 0.00   |
| VTCLX  | 0.35   |
| VTIAX  | 0.00   |
| VTSAX  | 0.35   |
| VUIAX  | 0.30   |

- Expected annual return: ~12.5%
- Annual volatility: ~16.7%
- Sharpe ratio: ~0.48 (using a 4.5% risk-free rate)

This comes at a modest cost to the Sharpe ratio (down from ~0.52 uncapped),
which illustrates the typical diversification trade-off: spreading risk
across more funds slightly reduces the best-case risk-adjusted return, but
avoids concentrating the whole portfolio in one holding.

## Notes / caveats

- Expected returns are estimated from historical means, which are noisy and
  not a reliable predictor of future returns — small changes in the input
  data can swing the optimizer's allocation significantly.
- The risk-free rate (4.5%) is a placeholder and should be replaced with a
  rate matched to the historical period being analyzed.
- All 6 funds are broad equity index funds, so they are likely highly
  correlated; the diversification benefit here is limited compared to mixing
  in bonds or other asset classes.
