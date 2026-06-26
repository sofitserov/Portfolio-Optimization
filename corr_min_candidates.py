"""
corr_min_candidates.py

Walk-forward correlation-minimization backtest on the lower-risk candidate
fund universe. No mean-reversion overlay — pure correlation minimization only.
Compare results to universe_exploration.py and frontier_candidates.py.
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

from frontier import compute_returns
from backtest import (
    run_backtest,
    equal_weight_benchmark,
    performance_summary,
    print_performance,
    plot_backtest,
    plot_weight_history,
)

TICKERS = {
    # Original equity universe
    "VCADX": "CA Muni Bond",
    "VSIAX": "Small-Cap Value",
    "VTCLX": "Tax-Mgd Cap Apprec",
    "VTIAX": "Total Intl Stock",
    "VTSAX": "Total Stock Market",
    "VUIAX": "Utilities",
    # Lower-risk candidates
    "VBTLX": "Total Bond Market",
    "VBILX": "Intermediate Bond",
    "VBLTX": "Long-Term Bond",
    "VTABX": "Total Intl Bond",
    "VWIUX": "Intermediate Tax-Exempt",
    "VWLUX": "Long-Term Tax-Exempt",
    "VFSUX": "Short-Term Investment-Grade",
    "VBIAX": "Balanced Index",
    "VWINX": "Wellesley Income",
    "VWELX": "Wellington",
    "VDIGX": "Dividend Growth",
    "VDADX": "Dividend Appreciation",
    "VGSLX": "Real Estate Index",
    "VEIRX": "Equity Income",
}

START = "2015-01-01"
END = "2026-01-01"
LOOKBACK = 60          # correlation window — short captures recent regime
RETURNS_LOOKBACK = 252 # return/covariance window — needs a full year to be meaningful
REBALANCE_PERIOD = 60
MAX_WEIGHT = 0.35
RISK_FREE_RATE = 0.045

# sharpe_alpha controls the blend between objectives:
#   1.0 = pure correlation minimization (ignores returns)
#   0.0 = pure Sharpe maximization (ignores correlations)
#   values in between trade off the two
SHARPE_ALPHA = 0.5


def main():
    tickers = list(TICKERS.keys())
    data = yf.download(tickers, start=START, end=END)["Close"].dropna()

    available = [t for t in tickers if t in data.columns]
    dropped = [t for t in tickers if t not in available]
    if dropped:
        print(f"Dropped (no data): {dropped}")
    data = data[available]

    returns = compute_returns(data)

    print(f"Universe: {len(available)} funds, {len(returns)} trading days")
    print(f"Lookback: {LOOKBACK}d, rebalance: every {REBALANCE_PERIOD}d, max weight per fund: {MAX_WEIGHT:.0%}\n")

    benchmark_returns = None
    named_returns = {}
    alpha_values = [1.0, 0.7, 0.5, 0.3, 0.0]

    for alpha in alpha_values:
        label = {1.0: "Corr-min only (α=1.0)", 0.0: "Sharpe only (α=0.0)"}.get(
            alpha, f"Blended α={alpha}"
        )
        strat_returns, weights = run_backtest(
            returns, LOOKBACK, REBALANCE_PERIOD, MAX_WEIGHT,
            reversion_lambda=0.0,
            sharpe_alpha=alpha,
            risk_free_rate=RISK_FREE_RATE,
            returns_lookback=RETURNS_LOOKBACK,
        )
        if benchmark_returns is None:
            benchmark_returns = equal_weight_benchmark(returns.loc[strat_returns.index])
        named_returns[label] = strat_returns
        print_performance(label, performance_summary(strat_returns))

    print_performance("Equal-weight benchmark", performance_summary(benchmark_returns))
    named_returns["Equal-Weight Benchmark"] = benchmark_returns

    plot_backtest(named_returns)


if __name__ == "__main__":
    main()
