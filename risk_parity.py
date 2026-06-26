"""
risk_parity.py

Walk-forward risk parity backtest on the combined universe (original equity
funds + lower-risk candidates). Risk parity equalizes each fund's contribution
to total portfolio variance — high-vol equity gets less weight, low-vol bonds
get more, but nothing is zeroed out. Compare against correlation minimization
and equal-weight to see which approach suits the mixed universe best.
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

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
LOOKBACK = 60
REBALANCE_PERIOD = 60
MAX_WEIGHT = 0.35


# ---------------------------------------------------------------------------
# Risk parity solver
# ---------------------------------------------------------------------------

def risk_parity_weights(cov_matrix: pd.DataFrame, max_weight: float = 1.0) -> np.ndarray:
    """
    Find weights that equalize each asset's contribution to total portfolio variance.

    Uses the log-barrier formulation:
        minimize  w^T Σ w  -  Σ log(wᵢ)
    solved without an equality constraint, then normalized to sum to 1.
    The -log term mathematically guarantees equal risk contributions at the
    optimum and is far better scaled than the squared-difference objective
    (which operates on tiny daily covariance values and stalls numerically).

    max_weight is applied as a post-processing cap after normalization.
    """
    n = len(cov_matrix)
    sigma = cov_matrix.values * 252  # annualize for scaling

    def objective(w):
        return w @ sigma @ w - np.sum(np.log(w))

    def gradient(w):
        return 2.0 * (sigma @ w) - 1.0 / w

    result = minimize(
        objective,
        x0=np.full(n, 1 / n),
        method="L-BFGS-B",
        jac=gradient,
        bounds=tuple((1e-8, None) for _ in range(n)),
        options={"ftol": 1e-15, "gtol": 1e-10},
    )
    w = np.maximum(result.x, 0)
    w /= w.sum()
    if max_weight < 1.0:
        for _ in range(100):
            prev = w.copy()
            w = np.minimum(w, max_weight)
            w /= w.sum()
            if np.allclose(w, prev, atol=1e-9):
                break
    return w


def run_risk_parity_backtest(
    returns: pd.DataFrame,
    lookback: int = 60,
    rebalance_period: int = 60,
    max_weight: float = 1.0,
) -> tuple:
    tickers = returns.columns
    portfolio_returns = []
    portfolio_dates = []
    weight_history = []

    start = lookback
    while start < len(returns):
        cov_matrix = returns.iloc[start - lookback:start].cov()
        weights = risk_parity_weights(cov_matrix, max_weight)

        hold_window = returns.iloc[start:start + rebalance_period]
        portfolio_returns.extend(hold_window.values @ weights)
        portfolio_dates.extend(hold_window.index)
        weight_history.append({"date": hold_window.index[0], **dict(zip(tickers, weights))})

        start += rebalance_period

    strategy_returns = pd.Series(portfolio_returns, index=portfolio_dates, name="strategy_return")
    weights_df = pd.DataFrame(weight_history).set_index("date")
    return strategy_returns, weights_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tickers = list(TICKERS.keys())
    data = yf.download(tickers, start=START, end=END)["Close"].dropna()
    available = [t for t in tickers if t in data.columns]
    data = data[available]
    returns = compute_returns(data)

    print(f"Universe: {len(available)} funds, {len(returns)} trading days")

    rp_returns, rp_weights = run_risk_parity_backtest(
        returns, LOOKBACK, REBALANCE_PERIOD, MAX_WEIGHT,
    )
    corr_returns, _ = run_backtest(
        returns, LOOKBACK, REBALANCE_PERIOD, MAX_WEIGHT,
        reversion_lambda=0.0,
    )
    benchmark_returns = equal_weight_benchmark(returns.loc[rp_returns.index])

    print_performance("Risk parity", performance_summary(rp_returns))
    print_performance("Correlation-minimization", performance_summary(corr_returns.loc[rp_returns.index]))
    print_performance("Equal-weight benchmark", performance_summary(benchmark_returns))

    print("\nRisk parity weight history (each row = one rebalance):")
    print(rp_weights.round(4).to_string())

    print("\nWeight range per fund (min → max):")
    for ticker in rp_weights.columns:
        lo, hi = rp_weights[ticker].min(), rp_weights[ticker].max()
        avg = rp_weights[ticker].mean()
        print(f"  {ticker:8s} ({TICKERS.get(ticker, ''):30s}): {lo:.4f} → {hi:.4f}  (avg {avg:.4f})")

    # Line chart — shows individual fund weight changes more clearly than stacked area
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    for ticker in rp_weights.columns:
        ax.plot(rp_weights.index, rp_weights[ticker], marker="o", markersize=3,
                label=f"{ticker} ({TICKERS.get(ticker, '')})")
    ax.set_ylabel("Weight")
    ax.set_xlabel("Rebalance date")
    ax.set_title("Risk Parity Weights Over Time — Combined Universe")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plot_backtest({
        "Risk Parity": rp_returns,
        "Correlation-Minimization": corr_returns.loc[rp_returns.index],
        "Equal-Weight Benchmark": benchmark_returns,
    })


if __name__ == "__main__":
    main()
