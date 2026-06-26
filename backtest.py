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
                          reversion_scores: pd.Series = None, reversion_lambda: float = 0.0,
                          expected_returns: pd.Series = None, cov_matrix: pd.DataFrame = None,
                          sharpe_alpha: float = 1.0, risk_free_rate: float = 0.0) -> np.ndarray:
    """
    Solve for weights that minimize the weighted-average pairwise correlation
    between holdings: minimize w^T (R - I) w, where R is the correlation
    matrix (subtracting I drops the self-correlation diagonal, which is
    always 1 and would otherwise dominate the objective).

    If `reversion_scores` is given, adds a `reversion_lambda * (w . scores)`
    penalty, pushing weight away from funds currently overextended relative
    to their own historical average return, and toward funds that have
    recently lagged their own average.

    If `expected_returns` and `cov_matrix` are given, blends in a Sharpe
    penalty via `sharpe_alpha`:
        L(w) = sharpe_alpha * corr_term - (1 - sharpe_alpha) * Sharpe(w)
    sharpe_alpha=1.0 is pure correlation minimization; 0.0 is pure Sharpe
    maximization; values in between trade off the two objectives.
    """
    num_assets = len(corr_matrix)
    off_diag = corr_matrix.values - np.eye(num_assets)
    use_sharpe = (expected_returns is not None and cov_matrix is not None and sharpe_alpha < 1.0)

    def objective(w):
        corr_term = w @ off_diag @ w
        if reversion_scores is not None and reversion_lambda:
            corr_term += reversion_lambda * np.dot(w, reversion_scores.values)
        if use_sharpe:
            port_return = np.dot(w, expected_returns.values)
            port_vol = np.sqrt(w @ cov_matrix.values @ w + 1e-10)
            sharpe = (port_return - risk_free_rate) / port_vol
            return sharpe_alpha * corr_term - (1.0 - sharpe_alpha) * sharpe
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
                  reversion_lambda: float = 0.0, reversion_short: int = 60, reversion_long: int = 252,
                  sharpe_alpha: float = 1.0, risk_free_rate: float = 0.0, returns_lookback: int = 252):
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

        exp_rets, cov = None, None
        if sharpe_alpha < 1.0 and start >= returns_lookback:
            returns_window = returns.iloc[start - returns_lookback:start]
            exp_rets = returns_window.mean() * 252
            cov = returns_window.cov() * 252

        weights = minimize_correlation(corr_matrix, max_weight, reversion_scores, reversion_lambda,
                                       exp_rets, cov, sharpe_alpha, risk_free_rate)

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


def sweep_lambda(
    returns: pd.DataFrame,
    lambda_grid: list,
    lookback: int = 60,
    rebalance_period: int = 60,
    max_weight: float = 1.0,
    reversion_short: int = 60,
    reversion_long: int = 252,
    eval_start=None,
) -> pd.DataFrame:
    """
    Run the full backtest for each lambda in lambda_grid and collect performance stats.
    If eval_start is a date, only returns from that date forward count toward stats —
    this normalizes the comparison window so all lambdas are judged post-warmup.
    """
    rows = []
    for lam in lambda_grid:
        strat, _ = run_backtest(
            returns, lookback, rebalance_period, max_weight,
            reversion_lambda=lam,
            reversion_short=reversion_short,
            reversion_long=reversion_long,
        )
        if eval_start is not None:
            strat = strat[strat.index >= eval_start]
        stats = performance_summary(strat)
        stats["lambda"] = lam
        rows.append(stats)
    return pd.DataFrame(rows).set_index("lambda")


def plot_lambda_sensitivity(sweep_df: pd.DataFrame):
    """
    Sharpe ratio vs reversion_lambda.
    A broad plateau means the choice is robust; a lone spike signals overfit.
    """
    baseline = sweep_df.loc[0.0, "sharpe_ratio"] if 0.0 in sweep_df.index else None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sweep_df.index, sweep_df["sharpe_ratio"], marker="o")
    if baseline is not None:
        ax.axhline(baseline, color="gray", linestyle="--", label="lambda=0 (no reversion)")
    ax.set_xlabel("reversion_lambda")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Sharpe vs reversion_lambda — plateau=robust, spike=overfit")
    ax.legend()
    plt.tight_layout()
    plt.show()


def out_of_sample_eval(
    returns: pd.DataFrame,
    train_end: str,
    lambda_grid: list,
    lookback: int = 60,
    rebalance_period: int = 60,
    max_weight: float = 1.0,
    reversion_short: int = 60,
    reversion_long: int = 252,
):
    """
    Split returns at train_end. Pick the best lambda by Sharpe on the train period,
    then compare all lambdas on the test period (using full history for lookback).

    Returns (best_lambda, train_sweep_df, test_sweep_df).
    The test sweep includes lambda=0 as the no-reversion baseline — if the tuned
    lambda doesn't beat it out of sample, the reversion overlay adds nothing.
    """
    train_returns = returns[returns.index <= train_end]
    test_start = returns.index[returns.index > train_end][0]

    train_sweep = sweep_lambda(
        train_returns, lambda_grid, lookback, rebalance_period, max_weight,
        reversion_short, reversion_long,
    )
    best_lambda = float(train_sweep["sharpe_ratio"].idxmax())

    # Use full returns so the test period has correct lookback/reversion history;
    # slice to test period only when computing stats.
    test_sweep = sweep_lambda(
        returns, lambda_grid, lookback, rebalance_period, max_weight,
        reversion_short, reversion_long, eval_start=test_start,
    )
    return best_lambda, train_sweep, test_sweep


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

    # -- Baseline backtest --
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

    # -- Lambda overfit diagnostics --
    # All lambdas are evaluated on the same post-warmup window so early
    # rebalances (where reversion was skipped for every lambda) don't add
    # shared noise that makes differences harder to see.
    LAMBDA_GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    TRAIN_END = "2020-12-31"
    post_warmup_start = returns.index[REVERSION_LONG] if len(returns) > REVERSION_LONG else returns.index[0]

    print(f"\n--- Lambda sensitivity (full period, evaluated from {post_warmup_start.date()} post-warmup) ---")
    print("Broad plateau = robust choice; lone spike = likely overfit.\n")
    full_sweep = sweep_lambda(
        returns, LAMBDA_GRID, LOOKBACK, REBALANCE_PERIOD, MAX_WEIGHT,
        REVERSION_SHORT, REVERSION_LONG, eval_start=post_warmup_start,
    )
    print(full_sweep[["sharpe_ratio", "annual_return", "max_drawdown"]].round(4))
    plot_lambda_sensitivity(full_sweep)

    print(f"\n--- Out-of-sample eval (train ≤ {TRAIN_END}, test after) ---")
    best_lam, train_sw, test_sw = out_of_sample_eval(
        returns, TRAIN_END, LAMBDA_GRID, LOOKBACK, REBALANCE_PERIOD, MAX_WEIGHT,
        REVERSION_SHORT, REVERSION_LONG,
    )
    print(f"Best lambda in-sample: {best_lam}")
    print("\nIn-sample Sharpe by lambda:")
    print(train_sw[["sharpe_ratio"]].round(4))
    print("\nOut-of-sample Sharpe by lambda (lambda=0 row is the no-reversion baseline):")
    print(test_sw[["sharpe_ratio", "annual_return", "max_drawdown"]].round(4))
    print(
        "\nNote: lookback, rebalance_period, max_weight, and reversion windows were also "
        "chosen on this data — fixing lambda alone does not eliminate in-sample bias."
    )

    plot_weight_history(corr_weights, "Portfolio Weights Over Time — Correlation-Minimization")
    plot_weight_history(combo_weights, "Portfolio Weights Over Time — Correlation + Mean-Reversion")
    plot_backtest({
        "Correlation-Minimization": corr_returns,
        "Correlation + Mean-Reversion": combo_returns,
        "Equal-Weight Benchmark": benchmark_returns,
    })


if __name__ == "__main__":
    main()
