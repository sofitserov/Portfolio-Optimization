# Portfolio Optimization

An exploration of portfolio construction strategies on Vanguard mutual funds (2015–2026), built to understand *when* and *why* optimization adds value over a simple equal-weight baseline.

---

## The Question

Can a systematic optimizer — one that looks at historical correlations, volatility, and returns — build a better long-term portfolio than just splitting evenly across all funds? And if it can in-sample, does that hold when it's actually tested on data it never saw?

---

## Setup

- **Universe:** 6 original Vanguard equity funds + 14 lower-risk candidates (bonds, balanced, dividend, real estate) — 20 funds total
- **Data:** Daily NAV prices 2015–2026 via `yfinance`
- **Baseline:** Equal-weight across all funds in the universe, rebalanced periodically

---

## What We Tried

### Sharpe Maximization (`frontier.py`, `frontier_candidates.py`)

Classic Markowitz mean-variance optimization: find the weights that maximize return per unit of risk. On the original 6 equity funds it concentrated into just 3, producing a higher Sharpe (0.51) than equal-weight (0.40) but almost identical drawdown (-34.9% vs -32.4%) — not a meaningful improvement given the concentration risk.

On the full 20-fund universe the optimizer was more interesting — it mixed equity and bond funds, cutting volatility dramatically and improving Sharpe to 0.59 vs equal-weight's 0.37:

| | Optimal | Equal-weight |
|---|---|---|
| Annual return | 9.26% | 6.74% |
| Volatility | 8.05% | 6.08% |
| Sharpe | 0.59 | 0.37 |
| Max drawdown | -9.94% | -8.09% |

**But these numbers use the full dataset to both fit and evaluate the weights.** To test whether the optimizer's picks actually generalize, we trained on the first 70% of the data and evaluated on the remaining 30% without retraining:

- The optimal portfolio **did** achieve higher returns than equal-weight out-of-sample
- But it came with a **larger drawdown**, a direct consequence of concentrating into only 4 funds
- This is the fundamental tension: the optimizer finds a high-Sharpe combination historically, but concentration means any single fund underperforming hits the portfolio hard

### Walk-Forward Correlation Minimization (`backtest.py`, `corr_min_candidates.py`)

Instead of maximizing Sharpe, minimize the weighted average pairwise correlation between holdings — rewarding diversification directly. Run as a true walk-forward backtest: every 60 days, look back at recent correlations, solve for new weights, hold for the next 60 days. No look-ahead.

On the equity-only universe it underperformed equal-weight — all 6 funds are highly correlated so the optimizer had little to work with. On the mixed 20-fund universe it did worse: it avoided equity entirely because equity funds are correlated with each other and bonds aren't. It minimized correlation perfectly while sacrificing all equity return.

To fix this, we added a blended objective that penalizes correlation *and* rewards Sharpe:

```
L(w) = α × (correlation term) − (1 − α) × Sharpe(w)
```

Return estimates used a 252-day lookback (vs 60-day for correlation) — short-window return estimates are essentially noise and poisoned earlier runs of the blended objective. Even with the fix, equal-weight won across all values of α in the walk-forward test.

We also added a **mean-reversion overlay** (`reversion_lambda`): tilt away from funds that have recently outperformed their own long-run average. With only ~40 effective rebalance decisions over 11 years, any tuned lambda is mostly fitting noise — we added a sweep and train/test split to diagnose this.

### Risk Parity (`risk_parity.py`)

Equalize each fund's *contribution to portfolio variance*. High-vol equity gets a small weight; low-vol bonds get more — but nothing is zeroed out. Solved via the log-barrier formulation with an analytical gradient (the naive squared-difference objective stalls numerically on real covariance matrices):

```
minimize  w^T Σ w − Σ log(wᵢ)
```

Risk parity correctly spread weight (equity ~2–5%, bonds ~10–15%) and produced stable allocations, but still didn't beat equal-weight in realized walk-forward returns.

---

## The Honest Conclusion

Equal-weight is a surprisingly strong baseline — consistent with DeMiguel et al. (2009) *"Optimal Versus Naive Diversification"*. Every walk-forward test confirmed it. The one place the Sharpe optimizer showed a genuine edge (higher out-of-sample returns on the mixed universe) came with larger drawdown and high concentration risk — a tradeoff, not a free lunch.

The deeper issue: every optimization strategy requires estimating something from noisy historical data. With ~40 rebalance decisions over 11 years, that estimation error routinely dominates the signal. The optimizer that "won" in-sample was largely fitting to noise.

**Optimization adds real value when:**
- The universe is large enough that equal-weight creates unnecessary concentration in correlated clusters
- You have a specific risk target (volatility cap, drawdown limit) rather than a return target
- Expected returns come from a factor model or fundamental signal — not short-window historical means

---

## Stack
Python · NumPy · pandas · scipy · yfinance · matplotlib
