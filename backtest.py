import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from main import compute_returns


def minimize_correlation(corr_matrix: pd.DataFrame, max_weight: float = 1.0) -> np.ndarray:
    """
    Solve for weights that minimize the weighted-average pairwise correlation
    between holdings: minimize w^T (R - I) w, where R is the correlation
    matrix (subtracting I drops the self-correlation diagonal, which is
    always 1 and would otherwise dominate the objective).
    """
    num_assets = len(corr_matrix)
    off_diag = corr_matrix.values - np.eye(num_assets)

    def objective(w):
        return w @ off_diag @ w

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, max_weight) for _ in range(num_assets))
    initial_guess = np.array([1 / num_assets] * num_assets)

    result = minimize(
        objective,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result.x


def run_backtest(returns: pd.DataFrame, lookback: int = 60, rebalance_period: int = 60, max_weight: float = 1.0):
    """
    Walk forward through `returns`: every `rebalance_period` trading days,
    look back `lookback` days, solve for weights that minimize pairwise
    correlation, then hold those weights for the next `rebalance_period`
    days before rebalancing again.

    Returns (strategy_returns, weight_history) where strategy_returns is a
    Series of realized daily portfolio returns and weight_history is a
    DataFrame of the weights used at each rebalance date.
    """
    tickers = returns.columns
    portfolio_returns = []
    portfolio_dates = []
    weight_history = []

    start = lookback
    while start < len(returns):
        lookback_window = returns.iloc[start - lookback:start]
        corr_matrix = lookback_window.corr()
        weights = minimize_correlation(corr_matrix, max_weight)

        hold_window = returns.iloc[start:start + rebalance_period]
        hold_returns = hold_window.values @ weights

        portfolio_returns.extend(hold_returns)
        portfolio_dates.extend(hold_window.index)
        weight_history.append({"date": hold_window.index[0], **dict(zip(tickers, weights))})

        start += rebalance_period

    strategy_returns = pd.Series(portfolio_returns, index=portfolio_dates, name="strategy_return")
    weights_df = pd.DataFrame(weight_history).set_index("date")
    return strategy_returns, weights_df


def equal_weight_benchmark(returns: pd.DataFrame) -> pd.Series:
    """Static equal-weight buy-and-hold benchmark over the same period."""
    weights = np.array([1 / returns.shape[1]] * returns.shape[1])
    return pd.Series(returns.values @ weights, index=returns.index, name="benchmark_return")


def performance_summary(daily_returns: pd.Series, risk_free_rate: float = 0.045) -> dict:
    annual_return = daily_returns.mean() * 252
    annual_vol = daily_returns.std() * np.sqrt(252)
    sharpe = (annual_return - risk_free_rate) / annual_vol

    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    max_drawdown = drawdown.min()

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
    }


def plot_backtest(strategy_returns: pd.Series, benchmark_returns: pd.Series):
    strategy_curve = (1 + strategy_returns).cumprod()
    benchmark_curve = (1 + benchmark_returns).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(strategy_curve.index, strategy_curve, label="Correlation-Minimization Strategy")
    plt.plot(benchmark_curve.index, benchmark_curve, label="Equal-Weight Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.title("Backtest: Correlation-Minimization Rebalancing vs. Equal-Weight Benchmark")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_weight_history(weight_history: pd.DataFrame):
    """Stacked area chart of portfolio weight per fund at each rebalance date."""
    plt.figure(figsize=(10, 6))
    plt.stackplot(weight_history.index, weight_history.T.values, labels=weight_history.columns)
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.title("Portfolio Weights Over Time (Correlation-Minimization Strategy)")
    plt.legend(loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.show()


def main():
    tickers = ['VCADX', 'VSIAX', 'VTCLX', 'VTIAX', 'VTSAX', 'VUIAX']
    data = yf.download(tickers, start='2015-01-01', end='2026-01-01')['Close']
    returns = compute_returns(data)

    LOOKBACK = 60
    REBALANCE_PERIOD = 60
    MAX_WEIGHT = 0.35

    strategy_returns, weight_history = run_backtest(returns, LOOKBACK, REBALANCE_PERIOD, MAX_WEIGHT)
    benchmark_returns = equal_weight_benchmark(returns.loc[strategy_returns.index])

    strategy_stats = performance_summary(strategy_returns)
    benchmark_stats = performance_summary(benchmark_returns)

    print("Strategy weight history:")
    print(weight_history.round(4))

    print("\nStrategy performance:")
    print(f"  Annual return: {strategy_stats['annual_return']:.2%}")
    print(f"  Annual volatility: {strategy_stats['annual_volatility']:.2%}")
    print(f"  Sharpe ratio: {strategy_stats['sharpe_ratio']:.4f}")
    print(f"  Max drawdown: {strategy_stats['max_drawdown']:.2%}")

    print("\nBenchmark performance:")
    print(f"  Annual return: {benchmark_stats['annual_return']:.2%}")
    print(f"  Annual volatility: {benchmark_stats['annual_volatility']:.2%}")
    print(f"  Sharpe ratio: {benchmark_stats['sharpe_ratio']:.4f}")
    print(f"  Max drawdown: {benchmark_stats['max_drawdown']:.2%}")

    plot_weight_history(weight_history)
    plot_backtest(strategy_returns, benchmark_returns)


if __name__ == "__main__":
    main()
