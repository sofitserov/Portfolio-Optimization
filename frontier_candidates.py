"""
frontier_candidates.py

Runs the same Sharpe-maximization and efficient-frontier analysis as
frontier.py, but on the lower-risk candidate fund universe identified in
universe_exploration.py. Use this alongside frontier.py to compare optimal
portfolios across the two universes.
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from frontier import (
    compute_returns,
    compute_expected_returns,
    compute_covariance,
    portfolio_performance,
    optimize_portfolio,
    efficient_frontier,
    plot_efficient_frontier,
    plot_fund_growth,
    windowed_correlations,
    plot_windowed_correlations,
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
RFR = 0.045
MAX_WEIGHT = 0.35


def main():
    tickers = list(TICKERS.keys())
    data = yf.download(tickers, start=START, end=END)["Close"].dropna()

    # Drop any tickers that failed
    available = [t for t in tickers if t in data.columns and data[t].notna().all()]
    dropped = [t for t in tickers if t not in available]
    if dropped:
        print(f"Dropped (no data): {dropped}")
    data = data[available]

    returns = compute_returns(data)
    exp_rets = compute_expected_returns(returns)
    cov = compute_covariance(returns)

    print("\nCorrelation matrix:")
    print(returns.corr().round(3))

    optimal_weights = optimize_portfolio(exp_rets, cov, RFR, MAX_WEIGHT)
    port_return, port_vol, sharpe = portfolio_performance(optimal_weights, exp_rets, cov, RFR)

    print("\nOptimal weights:")
    for ticker, weight in zip(available, optimal_weights):
        if weight > 0.001:
            print(f"  {ticker:8s} ({TICKERS.get(ticker, ''):30s}): {weight:.2%}")

    n = len(available)
    ew_weights = np.array([1 / n] * n)
    ew_return, ew_vol, ew_sharpe = portfolio_performance(ew_weights, exp_rets, cov, RFR)

    def max_drawdown(weights):
        cum = (1 + returns.values @ weights).cumprod()
        return (cum / np.maximum.accumulate(cum) - 1).min()

    opt_dd = max_drawdown(optimal_weights)
    ew_dd  = max_drawdown(ew_weights)

    col = 26
    print(f"\n{'':>{col}}  {'Optimal':>10}  {'Equal-weight':>12}")
    print(f"{'Expected annual return':>{col}}  {port_return:>10.2%}  {ew_return:>12.2%}")
    print(f"{'Annual volatility':>{col}}  {port_vol:>10.2%}  {ew_vol:>12.2%}")
    print(f"{'Sharpe ratio':>{col}}  {sharpe:>10.4f}  {ew_sharpe:>12.4f}")
    print(f"{'Max drawdown':>{col}}  {opt_dd:>10.2%}  {ew_dd:>12.2%}")

    volatilities, target_returns = efficient_frontier(exp_rets, cov, MAX_WEIGHT)
    fund_vols = np.sqrt(np.diag(cov.values))
    plot_efficient_frontier(
        volatilities, target_returns, port_vol, port_return,
        fund_vols, exp_rets.values, available,
    )

    # corr_over_time = windowed_correlations(returns, window=60)
    # plot_windowed_correlations(corr_over_time, window=60)

    plot_fund_growth(data, available)

    # ------------------------------------------------------------------
    # Train / test split — the honest out-of-sample check
    # Fit optimal weights on the first 70% of data, hold fixed for
    # the remaining 30%, then compare realized returns to equal-weight.
    # ------------------------------------------------------------------
    split = int(len(returns) * 0.70)
    train_returns = returns.iloc[:split]
    test_returns  = returns.iloc[split:]
    train_end  = returns.index[split - 1].date()
    test_start = returns.index[split].date()

    print(f"\n--- Train / test split ---")
    print(f"Train: {returns.index[0].date()} → {train_end}  ({split} days)")
    print(f"Test:  {test_start} → {returns.index[-1].date()}  ({len(test_returns)} days)")

    train_exp_rets = compute_expected_returns(train_returns)
    train_cov      = compute_covariance(train_returns)
    train_weights  = optimize_portfolio(train_exp_rets, train_cov, RFR, MAX_WEIGHT)

    print("\nWeights fitted on training data:")
    for ticker, weight in zip(available, train_weights):
        if weight > 0.001:
            print(f"  {ticker:8s} ({TICKERS.get(ticker, ''):30s}): {weight:.2%}")

    def realized_stats(daily_rets):
        ann_ret = daily_rets.mean() * 252
        ann_vol = daily_rets.std() * np.sqrt(252)
        sharpe  = (ann_ret - RFR) / ann_vol
        cum     = (1 + daily_rets).cumprod()
        max_dd  = (cum / np.maximum.accumulate(cum) - 1).min()
        return ann_ret, ann_vol, sharpe, max_dd

    n = len(available)
    ew = np.array([1 / n] * n)

    opt_daily = pd.Series(test_returns.values @ train_weights, index=test_returns.index)
    ew_daily  = pd.Series(test_returns.values @ ew,            index=test_returns.index)

    opt_r, opt_v, opt_s, opt_d = realized_stats(opt_daily)
    ew_r,  ew_v,  ew_s,  ew_d  = realized_stats(ew_daily)

    print(f"\nRealized performance on test period ({test_start} → {returns.index[-1].date()}):")
    print(f"\n{'':>{col}}  {'Optimal (OOS)':>14}  {'Equal-weight':>12}")
    print(f"{'Annual return':>{col}}  {opt_r:>14.2%}  {ew_r:>12.2%}")
    print(f"{'Annual volatility':>{col}}  {opt_v:>14.2%}  {ew_v:>12.2%}")
    print(f"{'Sharpe ratio':>{col}}  {opt_s:>14.4f}  {ew_s:>12.4f}")
    print(f"{'Max drawdown':>{col}}  {opt_d:>14.2%}  {ew_d:>12.2%}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot((1 + opt_daily).cumprod(), label="Optimal (trained on 70%)")
    ax.plot((1 + ew_daily).cumprod(),  label="Equal-weight")
    ax.set_title(f"Out-of-sample growth of $1  ({test_start} → {returns.index[-1].date()})")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
