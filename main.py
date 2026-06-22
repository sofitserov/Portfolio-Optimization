

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
# import pypfopt as pf  # not installed / not yet used
from scipy.optimize import minimize


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


def main():
    tickers = ['VCADX', 'VSIAX', 'VTCLX', 'VTIAX', 'VTSAX', 'VUIAX']
    data = yf.download(tickers, start='2015-01-01', end='2025-01-01')['Close']

    RFR = 0.045 # to be adjusted later with a more specific yield for that period
    MAX_WEIGHT = 0.35 # per-fund cap to avoid all-or-nothing concentration
    returns = compute_returns(data)
    exp_rets = compute_expected_returns(returns)
    cov = compute_covariance(returns)
    optimal_weights = optimize_portfolio(exp_rets, cov, RFR, MAX_WEIGHT)

    port_return, port_vol, sharpe = portfolio_performance(optimal_weights, exp_rets, cov, RFR)

    print("Optimal weights:")
    for ticker, weight in zip(tickers, optimal_weights):
        print(f"  {ticker}: {weight:.4f}")
    print(f"Expected annual return: {port_return:.4f}")
    print(f"Annual volatility: {port_vol:.4f}")
    print(f"Sharpe ratio: {sharpe:.4f}")

    volatilities, target_returns = efficient_frontier(exp_rets, cov, MAX_WEIGHT)
    fund_vols = np.sqrt(np.diag(cov))
    plot_efficient_frontier(volatilities, target_returns, port_vol, port_return, fund_vols, exp_rets.values, tickers)


if __name__ == "__main__":
    main()
