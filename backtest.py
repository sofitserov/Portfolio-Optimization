import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from frontier import compute_returns


def mean_reversion_scores(returns_so_far: pd.DataFrame, short_window: int = 60, long_window: int = 252) -> pd.Series:
    """
    Per-fund z-score of recent average return vs. that fund's own longer-run
    historical average return: z = (recent_mean - long_run_mean) / long_run_std.

    A high positive z means the fund has recently been outperforming its own
    history (a mean-reversion candidate to move away from); a negative z
    means it's recently underperformed its own history.
    """
    recent_mean = returns_so_far.iloc[-short_window:].mean()
    long_run = returns_so_far.iloc[-long_window:]
    return (recent_mean - long_run.mean()) / long_run.std()


def minimize_correlation(corr_matrix: pd.DataFrame, max_weight: float = 1.0,
                          reversion_scores: pd.Series = None, reversion_lambda: float = 0.0) -> np.ndarray:
    """
    Solve for weights that minimize the weighted-average pairwise correlation
    between holdings: minimize w^T (R - I) w, where R is the correlation
    matrix (subtracting I drops the self-correlation diagonal, which is
    always 1 and would otherwise dominate the objective).

    If `reversion_scores` is given, adds a `reversion_lambda * (w . scores)`
    penalty, pushing weight away from funds currently overextended relative
    to their own historical average return, and toward funds that have
    recently lagged their own average.
    """
    num_assets = len(corr_matrix)
    off_diag = corr_matrix.values - np.eye(num_assets)

    def objective(w):
        corr_term = w @ off_diag @ w
        if reversion_scores is not None and reversion_lambda:
            corr_term += reversion_lambda * np.dot(w, reversion_scores.values)
        return corr_term

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


def run_backtest(returns: pd.DataFrame, lookback: int = 60, rebalance_period: int = 60, max_weight: float = 1.0,
                  reversion_lambda: float = 0.0, reversion_short: int = 60, reversion_long: int = 252):
    """
    Walk forward through `returns`: every `rebalance_period` trading days,
    look back `lookback` days, solve for weights that minimize pairwise
    correlation (optionally penalized by mean-reversion z-scores), then hold
    those weights for the next `rebalance_period` days before rebalancing
    again.

    Mean-reversion scoring needs `reversion_long` days of prior history, so
    it's skipped (equivalent to reversion_lambda=0) until enough history has
    accumulated.

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

        reversion_scores = None
        if reversion_lambda and start >= reversion_long:
            reversion_scores = mean_reversion_scores(returns.iloc[:start], reversion_short, reversion_long)

        weights = minimize_correlation(corr_matrix, max_weight, reversion_scores, reversion_lambda)

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


def plot_backtest(named_returns: dict):
    """named_returns: dict of {label: daily_returns_series} to plot as growth-of-$1 curves."""
    plt.figure(figsize=(10, 6))
    for label, daily_returns in named_returns.items():
        curve = (1 + daily_returns).cumprod()
        plt.plot(curve.index, curve, label=label)
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.title("Backtest: Strategy Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_weight_history(weight_history: pd.DataFrame, title: str = "Portfolio Weights Over Time"):
    """Stacked area chart of portfolio weight per fund at each rebalance date."""
    plt.figure(figsize=(10, 6))
    plt.stackplot(weight_history.index, weight_history.T.values, labels=weight_history.columns)
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.title(title)
    plt.legend(loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.show()


def print_performance(label: str, stats: dict):
    print(f"\n{label} performance:")
    print(f"  Annual return: {stats['annual_return']:.2%}")
    print(f"  Annual volatility: {stats['annual_volatility']:.2%}")
    print(f"  Sharpe ratio: {stats['sharpe_ratio']:.4f}")
    print(f"  Max drawdown: {stats['max_drawdown']:.2%}")


def main():
    tickers = ['VCADX', 'VSIAX', 'VTCLX', 'VTIAX', 'VTSAX', 'VUIAX']
    data = yf.download(tickers, start='2015-01-01', end='2026-01-01')['Close']
    returns = compute_returns(data)

    LOOKBACK = 60
    REBALANCE_PERIOD = 60
    MAX_WEIGHT = 0.35
    REVERSION_LAMBDA = 0.5
    REVERSION_SHORT = 60
    REVERSION_LONG = 252

    corr_returns, corr_weights = run_backtest(returns, LOOKBACK, REBALANCE_PERIOD, MAX_WEIGHT)
    combo_returns, combo_weights = run_backtest(
        returns, LOOKBACK, REBALANCE_PERIOD, MAX_WEIGHT,
        reversion_lambda=REVERSION_LAMBDA, reversion_short=REVERSION_SHORT, reversion_long=REVERSION_LONG,
    )
    benchmark_returns = equal_weight_benchmark(returns.loc[corr_returns.index])

    print("Correlation-minimization weight history:")
    print(corr_weights.round(4))
    print("\nCorrelation + mean-reversion weight history:")
    print(combo_weights.round(4))

    print_performance("Correlation-minimization", performance_summary(corr_returns))
    print_performance("Correlation + mean-reversion", performance_summary(combo_returns))
    print_performance("Equal-weight benchmark", performance_summary(benchmark_returns))

    plot_weight_history(corr_weights, "Portfolio Weights Over Time — Correlation-Minimization")
    plot_weight_history(combo_weights, "Portfolio Weights Over Time — Correlation + Mean-Reversion")
    plot_backtest({
        "Correlation-Minimization": corr_returns,
        "Correlation + Mean-Reversion": combo_returns,
        "Equal-Weight Benchmark": benchmark_returns,
    })


if __name__ == "__main__":
    main()
