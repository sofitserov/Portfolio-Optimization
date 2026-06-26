"""
universe_exploration.py

Exploratory analysis of lower-risk Vanguard mutual funds as candidates for
expanding the portfolio universe. Run this standalone to help decide which
funds to add to backtest.py.
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

from frontier import compute_returns

# ---------------------------------------------------------------------------
# Universe definition
# ---------------------------------------------------------------------------

EXISTING = {
    "VCADX": "CA Muni Bond",
    "VSIAX": "Small-Cap Value",
    "VTCLX": "Tax-Mgd Cap Apprec",
    "VTIAX": "Total Intl Stock",
    "VTSAX": "Total Stock Market",
    "VUIAX": "Utilities",
}

CANDIDATES = {
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

ALL_TICKERS = {**EXISTING, **CANDIDATES}
START = "2015-01-01"
END = "2026-01-01"
RISK_FREE = 0.045


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(tickers: dict, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(list(tickers.keys()), start=start, end=end)["Close"]
    # Drop any tickers that failed to download
    failed = raw.columns[raw.isna().all()].tolist()
    if failed:
        print(f"Skipping (no data): {failed}")
        raw = raw.drop(columns=failed)
    return raw.dropna()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def risk_return_table(returns: pd.DataFrame, names: dict, rfr: float) -> pd.DataFrame:
    annual_ret = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = (annual_ret - rfr) / annual_vol
    max_dd = (
        (1 + returns).cumprod()
        .apply(lambda s: (s / s.cummax() - 1).min())
    )
    df = pd.DataFrame({
        "Name": [names.get(t, t) for t in returns.columns],
        "Ann. Return": annual_ret.values,
        "Ann. Vol": annual_vol.values,
        "Sharpe": sharpe.values,
        "Max Drawdown": max_dd.values,
    }, index=returns.columns)
    return df.sort_values("Ann. Vol")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(returns: pd.DataFrame, names: dict, title: str):
    corr = returns.corr()
    labels = [f"{t}\n{names.get(t, '')}" for t in corr.columns]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.9), max(8, n * 0.8)))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")

    fig.colorbar(im, ax=ax, label="Correlation")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_risk_return(stats: pd.DataFrame, existing_tickers: list):
    fig, ax = plt.subplots(figsize=(10, 6))
    for ticker, row in stats.iterrows():
        color = "steelblue" if ticker in existing_tickers else "darkorange"
        ax.scatter(row["Ann. Vol"], row["Ann. Return"], color=color, s=80, zorder=3)
        ax.annotate(
            ticker, (row["Ann. Vol"], row["Ann. Return"]),
            textcoords="offset points", xytext=(5, 3), fontsize=8,
        )

    # Legend proxies
    ax.scatter([], [], color="steelblue", label="Existing universe")
    ax.scatter([], [], color="darkorange", label="Candidate additions")

    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlabel("Annual Volatility")
    ax.set_ylabel("Annual Return")
    ax.set_title("Risk vs Return — existing + candidates")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_rolling_corr_with_market(
    returns: pd.DataFrame,
    market: str,
    candidates: list,
    names: dict,
    window: int = 126,
):
    """Rolling 6-month correlation of each candidate against the market proxy."""
    if market not in returns.columns:
        print(f"Market proxy {market} not in returns, skipping rolling corr plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    for ticker in candidates:
        if ticker not in returns.columns:
            continue
        roll = returns[ticker].rolling(window).corr(returns[market])
        ax.plot(roll.index, roll, label=f"{ticker} ({names.get(ticker, '')})", alpha=0.8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel(f"Rolling {window}-day correlation with {market}")
    ax.set_title(f"Rolling correlation vs {market} ({names.get(market, 'market proxy')})")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_growth(prices: pd.DataFrame, names: dict, title: str):
    normalized = prices / prices.iloc[0]
    fig, ax = plt.subplots(figsize=(12, 5))
    for ticker in normalized.columns:
        style = "-" if ticker in EXISTING else "--"
        ax.plot(normalized.index, normalized[ticker], linestyle=style,
                label=f"{ticker} ({names.get(ticker, '')})", alpha=0.85)
    ax.set_ylabel("Growth of $1")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Downloading data...")
    data = load_data(ALL_TICKERS, START, END)
    available = {t: ALL_TICKERS[t] for t in data.columns if t in ALL_TICKERS}
    existing_available = [t for t in EXISTING if t in data.columns]
    candidate_available = [t for t in CANDIDATES if t in data.columns]

    returns = compute_returns(data)

    # Risk/return table
    stats = risk_return_table(returns, available, RISK_FREE)
    pd.set_option("display.float_format", "{:.2%}".format)
    print("\nRisk/return summary (sorted by volatility):")
    print(stats.to_string())
    pd.reset_option("display.float_format")

    # Correlation heatmap — candidates only, to see how they relate to each other
    plot_correlation_heatmap(
        returns[candidate_available], CANDIDATES,
        "Candidate fund correlations",
    )

    # Correlation heatmap — full universe (existing + candidates)
    plot_correlation_heatmap(
        returns, available,
        "Full universe correlations (existing + candidates)",
    )

    # Risk/return scatter
    plot_risk_return(stats, existing_available)

    # Rolling correlation of each candidate against VTSAX (total market proxy)
    plot_rolling_corr_with_market(
        returns, market="VTSAX",
        candidates=candidate_available,
        names=CANDIDATES,
    )

    # Growth of $1 for the full universe
    plot_growth(data, available, "Growth of $1 — existing (solid) vs candidates (dashed)")


if __name__ == "__main__":
    main()
