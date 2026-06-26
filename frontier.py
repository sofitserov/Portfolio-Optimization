import itertools

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
# import pypfopt as pf  # not installed / not yet used
from scipy.optimize import minimize

# QUartely/ yearly rebalance
# Double check dividend inclusion in the prices data from Yahoo Finance (should be included in the adjusted close)
# Mean reversion over a longer time period?
# Impact of taxes vs tax-free funds
# Look for stocks with Betas closer to 0? Use seperate File

# Priorities:
# 
# Combination of sharpe optimization/mean reversion and the correlation minimization
# Mean reversion element?



def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def compute_expected_returns(returns: pd.DataFrame) -> pd.Series:
    return returns.mean() * 252


def compute_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov() * 252


def portfolio_performance(weights: np.ndarray, expected_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float):
    port_return = np.dot(weights, expected_returns)
    port_vol = np.sqrt(weights @ cov_matrix @ weights)
    sharpe = (port_return - risk_free_rate) / port_vol
    return port_return, port_vol, sharpe


def negative_sharpe(weights: np.ndarray, expected_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float) -> float:
    _, _, sharpe = portfolio_performance(weights, expected_returns, cov_matrix, risk_free_rate)
    return -sharpe


def optimize_portfolio(expected_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float, max_weight: float = 1.0):
    num_assets = len(expected_returns)
    args = (expected_returns, cov_matrix, risk_free_rate)
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, max_weight) for _ in range(num_assets))
    initial_guess = np.array([1 / num_assets] * num_assets)

    result = minimize(
        negative_sharpe,
        initial_guess,
        args=args,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result.x


def portfolio_volatility(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    return np.sqrt(weights @ cov_matrix @ weights)


def efficient_frontier(expected_returns: pd.Series, cov_matrix: pd.DataFrame, max_weight: float = 1.0, num_points: int = 50):
    """
    Trace the efficient frontier by minimizing volatility for a range of
    target returns. Returns (volatilities, target_returns).
    """
    num_assets = len(expected_returns)
    bounds = tuple((0, max_weight) for _ in range(num_assets))
    initial_guess = np.array([1 / num_assets] * num_assets)

    target_returns = np.linspace(expected_returns.min(), expected_returns.max(), num_points)
    volatilities = []

    for target in target_returns:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, target=target: np.dot(w, expected_returns) - target},
        )
        result = minimize(
            portfolio_volatility,
            initial_guess,
            args=(cov_matrix,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        volatilities.append(result.fun if result.success else np.nan)

    return np.array(volatilities), target_returns


def plot_efficient_frontier(volatilities, target_returns, optimal_vol, optimal_return, fund_vols, fund_returns, tickers):
    plt.figure(figsize=(10, 6))
    plt.plot(volatilities, target_returns, "b--", label="Efficient Frontier")
    plt.scatter(fund_vols, fund_returns, c="gray", marker="o", label="Individual Funds")
    for ticker, vol, ret in zip(tickers, fund_vols, fund_returns):
        plt.annotate(ticker, (vol, ret))
    plt.scatter(optimal_vol, optimal_return, c="red", marker="*", s=200, label="Optimal Portfolio")
    plt.xlabel("Volatility (Std. Dev.)")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_fund_growth(data: pd.DataFrame, tickers: list):
    """Plot normalized growth of $1 invested, for the given tickers, over the full date range."""
    normalized = data[tickers] / data[tickers].iloc[0]
    plt.figure(figsize=(10, 6))
    for ticker in tickers:
        plt.plot(normalized.index, normalized[ticker], label=ticker)
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.title(f"{' vs '.join(tickers)} — Growth Over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()


def windowed_correlations(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Slide a `window`-day lookback forward one trading day at a time and
    compute the pairwise correlation matrix within each window.

    Returns a DataFrame indexed by window end-date, one column per fund pair.
    """
    pairs = list(itertools.combinations(returns.columns, 2))
    records = []
    index = []

    for start in range(0, len(returns) - window + 1):
        chunk = returns.iloc[start:start + window]
        corr = chunk.corr()
        records.append({f"{a}-{b}": corr.loc[a, b] for a, b in pairs})
        index.append(chunk.index[-1])

    return pd.DataFrame(records, index=index)


def plot_windowed_correlations(corr_over_time: pd.DataFrame, window: int = 60):
    """Heatmap of fund-pair correlation (rows) across a sliding window (columns)."""
    matrix = corr_over_time.T
    col_labels = [d.strftime("%Y-%m-%d") for d in matrix.columns]

    max_ticks = 40
    tick_step = max(1, len(col_labels) // max_ticks)
    tick_positions = range(0, len(col_labels), tick_step)

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(matrix.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([col_labels[i] for i in tick_positions], rotation=90, fontsize="small")
    ax.set_xlabel("Window End Date")
    ax.set_title(f"Pairwise Fund Correlation — Sliding {window}-Day Window")
    fig.colorbar(im, ax=ax, label="Correlation")
    plt.tight_layout()
    plt.show()


def main():
    tickers = ['VCADX', 'VSIAX', 'VTCLX', 'VTIAX', 'VTSAX', 'VUIAX']
    data = yf.download(tickers, start='2015-01-01', end='2026-01-01')['Close']

    print(data.corr())
    plot_fund_growth(data, ['VCADX', 'VTIAX'])

    RFR = 0.045 # to be adjusted later with a more specific yield for that period
    MAX_WEIGHT = 0.35 # per-fund cap to avoid all-or-nothing concentration
    returns = compute_returns(data)
    exp_rets = compute_expected_returns(returns)
    cov = compute_covariance(returns)
    optimal_weights = optimize_portfolio(exp_rets, cov, RFR, MAX_WEIGHT)

    corr_over_time = windowed_correlations(returns, window=60)
    plot_windowed_correlations(corr_over_time, window=60)

    port_return, port_vol, sharpe = portfolio_performance(optimal_weights, exp_rets, cov, RFR)

    ew_weights = np.array([1 / len(tickers)] * len(tickers))
    ew_return, ew_vol, ew_sharpe = portfolio_performance(ew_weights, exp_rets, cov, RFR)

    returns = compute_returns(data)
    def max_drawdown(weights):
        cum = (1 + returns.values @ weights).cumprod()
        return (cum / np.maximum.accumulate(cum) - 1).min()

    opt_dd = max_drawdown(optimal_weights)
    ew_dd  = max_drawdown(ew_weights)

    print("\nOptimal weights:")
    for ticker, weight in zip(tickers, optimal_weights):
        if weight > 0.001:
            print(f"  {ticker}: {weight:.2%}")

    col = 22
    print(f"\n{'':>{col}}  {'Optimal':>10}  {'Equal-weight':>12}")
    print(f"{'Expected annual return':>{col}}  {port_return:>10.2%}  {ew_return:>12.2%}")
    print(f"{'Annual volatility':>{col}}  {port_vol:>10.2%}  {ew_vol:>12.2%}")
    print(f"{'Sharpe ratio':>{col}}  {sharpe:>10.4f}  {ew_sharpe:>12.4f}")
    print(f"{'Max drawdown':>{col}}  {opt_dd:>10.2%}  {ew_dd:>12.2%}")

    volatilities, target_returns = efficient_frontier(exp_rets, cov, MAX_WEIGHT)
    fund_vols = np.sqrt(np.diag(cov))
    plot_efficient_frontier(volatilities, target_returns, port_vol, port_return, fund_vols, exp_rets.values, tickers)


if __name__ == "__main__":
    main()
