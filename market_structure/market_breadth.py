#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S&P 500 market breadth based on rolling log-price regression extremes.

This single script replaces the former dashboard_marche.py, lancer.py and
regression_largeur.py workflow. It has no GUI and creates only two project
outputs next to the script:

    breadth_extremes.png
    breadth_net.png

Method
------
1. Fetch the current S&P 500 constituent list.
2. Download adjusted WEEKLY prices directly from Yahoo Finance.
3. Fit a rolling linear regression to log prices for every constituent.
4. Express the latest regression residual in standard-deviation units.
5. Measure the share of stocks above +threshold sigma and below -threshold sigma.
6. Save the two breadth figures.

Performance notes
-----------------
- Weekly data are downloaded directly instead of downloading daily data first.
- Downloads are batched and multithreaded.
- Rolling regressions are computed from vectorized rolling sums instead of
  calling np.polyfit separately for every stock and every rolling window.
- A temporary system cache is used for repeated runs without adding cache files
  or folders to the project directory.

Dependencies:
    pip install numpy pandas requests yfinance matplotlib lxml

Usage:
    python market_breadth.py
    python market_breadth.py --refresh
    python market_breadth.py --window 500 --threshold 1.5
"""

from __future__ import annotations

import argparse
import io
import math
import tempfile
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_EXTREMES = SCRIPT_DIR / "breadth_extremes.png"
FIGURE_NET = SCRIPT_DIR / "breadth_net.png"

START_DATE = "1995-01-01"
DEFAULT_WINDOW = 500
DEFAULT_THRESHOLD = 1.5
MIN_PERIODS_FRACTION = 0.45
MIN_PERIODS_FLOOR = 80
CACHE_MAX_AGE_HOURS = 20
DOWNLOAD_BATCH_SIZE = 100

CACHE_DIR = Path(tempfile.gettempdir()) / "market_breadth_cache"
PRICE_CACHE = CACHE_DIR / "weekly_prices.pkl"

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

FALLBACK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "GOOGL", "META", "NFLX",
    "AMZN", "TSLA", "HD", "COST", "WMT", "PG", "JPM", "BAC", "V",
    "MA", "LLY", "UNH", "JNJ", "XOM", "CVX", "CAT", "GE", "LIN",
    "NEE", "PLD",
]


# -----------------------------------------------------------------------------
# Data retrieval
# -----------------------------------------------------------------------------

def cache_is_fresh(path: Path, max_age_hours: int = CACHE_MAX_AGE_HOURS) -> bool:
    if not path.exists():
        return False
    age_hours = (pd.Timestamp.now().timestamp() - path.stat().st_mtime) / 3600.0
    return age_hours <= max_age_hours


def get_sp500_tickers() -> list[str]:
    """Return the current S&P 500 constituents, with a small fallback universe."""
    try:
        response = requests.get(
            SP500_URL,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        response.raise_for_status()
        table = pd.read_html(io.StringIO(response.text))[0]
        tickers = (
            table["Symbol"]
            .astype(str)
            .str.strip()
            .str.replace(".", "-", regex=False)
            .dropna()
            .drop_duplicates()
            .tolist()
        )
        if len(tickers) < 400:
            raise RuntimeError("Unexpectedly small S&P 500 universe.")
        return tickers
    except Exception as exc:
        print(f"S&P 500 constituent download failed ({exc}). Using fallback universe.")
        return FALLBACK_TICKERS.copy()


def extract_close(raw: pd.DataFrame, requested: list[str]) -> pd.DataFrame:
    """Normalize yfinance output to a simple Close-price DataFrame."""
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        level1 = raw.columns.get_level_values(1)

        if "Close" in level0:
            close = raw["Close"].copy()
        elif "Close" in level1:
            close = raw.xs("Close", axis=1, level=1, drop_level=True).copy()
        else:
            return pd.DataFrame()
    elif "Close" in raw.columns:
        close = raw[["Close"]].copy()
        close.columns = [requested[0]]
    else:
        return pd.DataFrame()

    if isinstance(close, pd.Series):
        close = close.to_frame(name=requested[0])

    close.columns = [str(column) for column in close.columns]
    close = close.apply(pd.to_numeric, errors="coerce")

    if isinstance(close.index, pd.DatetimeIndex) and close.index.tz is not None:
        close.index = close.index.tz_localize(None)

    return close.sort_index()


def download_batch(symbols: list[str]) -> pd.DataFrame:
    raw = yf.download(
        symbols,
        start=START_DATE,
        interval="1wk",
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=True,
        group_by="column",
    )
    return extract_close(raw, symbols)


def download_weekly_prices(tickers: Iterable[str]) -> pd.DataFrame:
    """Download weekly adjusted closes in resilient batches."""
    symbols = list(dict.fromkeys(tickers))
    frames: list[pd.DataFrame] = []

    for start in range(0, len(symbols), DOWNLOAD_BATCH_SIZE):
        batch = symbols[start : start + DOWNLOAD_BATCH_SIZE]
        end = start + len(batch)
        print(f"Downloading weekly prices {start + 1}-{end}/{len(symbols)}...")
        try:
            close = download_batch(batch)
            if not close.empty:
                frames.append(close)
        except Exception as exc:
            print(f"Batch skipped after download error: {exc}")

    if not frames:
        raise RuntimeError("No Yahoo Finance prices could be downloaded.")

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated(keep="last")]

    missing = [
        ticker
        for ticker in symbols
        if ticker not in prices.columns or not prices[ticker].notna().any()
    ]

    if missing:
        print(f"Retrying {len(missing)} missing ticker(s)...")
        for start in range(0, len(missing), 20):
            batch = missing[start : start + 20]
            try:
                close = download_batch(batch)
                if not close.empty:
                    prices = pd.concat([prices, close], axis=1)
                    prices = prices.loc[:, ~prices.columns.duplicated(keep="last")]
            except Exception:
                continue

    selected = [ticker for ticker in symbols if ticker in prices.columns]
    prices = prices[selected].sort_index()

    # Preserve the original logic: once a stock has observations, missing weekly
    # values are carried forward, but values before its history starts remain NaN.
    prices = prices.ffill().dropna(how="all")

    if prices.empty:
        raise RuntimeError("Downloaded data contain no usable prices.")

    return prices


def load_weekly_prices(tickers: list[str], refresh: bool) -> pd.DataFrame:
    """Use a temporary cache when it is fresh and still covers most constituents."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not refresh and cache_is_fresh(PRICE_CACHE):
        try:
            cached = pd.read_pickle(PRICE_CACHE)
            if isinstance(cached, pd.DataFrame) and not cached.empty:
                available = [ticker for ticker in tickers if ticker in cached.columns]
                coverage = len(available) / max(len(tickers), 1)
                if coverage >= 0.90:
                    print(f"Using fresh weekly-price cache ({coverage:.0%} universe coverage).")
                    return cached[available].sort_index().ffill().dropna(how="all")
        except Exception:
            pass

    prices = download_weekly_prices(tickers)
    try:
        prices.to_pickle(PRICE_CACHE)
    except Exception:
        pass
    return prices


# -----------------------------------------------------------------------------
# Vectorized rolling regression
# -----------------------------------------------------------------------------

def rolling_regression_zscores(
    log_prices: pd.DataFrame,
    window: int,
    min_periods: int,
) -> pd.DataFrame:
    """
    Compute each stock's rolling regression residual z-score using rolling sums.

    This is algebraically equivalent to repeatedly fitting y = a + b*x and then
    dividing the latest residual by the sample standard deviation of all residuals
    in the window, but avoids Python-level np.polyfit calls.
    """
    if log_prices.empty:
        return pd.DataFrame(index=log_prices.index, columns=log_prices.columns)

    y = log_prices.astype(float)
    x = pd.Series(np.arange(len(y), dtype=float), index=y.index)
    x2 = x * x
    valid = y.notna().astype(float)

    rolling = dict(window=window, min_periods=min_periods)

    n = y.rolling(**rolling).count()
    sy = y.rolling(**rolling).sum()
    syy = (y * y).rolling(**rolling).sum()

    sx = valid.mul(x, axis=0).rolling(**rolling).sum()
    sxx = valid.mul(x2, axis=0).rolling(**rolling).sum()
    sxy = y.mul(x, axis=0).rolling(**rolling).sum()

    denominator = n * sxx - sx * sx
    denominator = denominator.where(denominator.abs() > 1e-12)

    slope = (n * sxy - sx * sy) / denominator
    intercept = (sy - slope * sx) / n

    fitted_latest = intercept.add(slope.mul(x, axis=0), fill_value=np.nan)
    latest_residual = y - fitted_latest

    # For OLS with an intercept: SSE = sum(y^2) - a*sum(y) - b*sum(x*y).
    sse = syy - intercept * sy - slope * sxy
    sse = sse.clip(lower=0.0)

    # Match the original script: residual.std(ddof=1), not the n-2 regression SEE.
    residual_variance = sse / (n - 1.0)
    residual_std = np.sqrt(residual_variance.where(n > 1.0))

    zscores = latest_residual / residual_std.replace(0.0, np.nan)
    zscores = zscores.where(n >= min_periods)
    return zscores


def compute_breadth(
    weekly_prices: pd.DataFrame,
    window: int,
    threshold: float,
) -> pd.DataFrame:
    """Compute positive, negative and net regression-extreme breadth."""
    min_periods = max(MIN_PERIODS_FLOOR, int(window * MIN_PERIODS_FRACTION))

    log_prices = np.log(weekly_prices.where(weekly_prices > 0))
    zscores = rolling_regression_zscores(log_prices, window, min_periods)

    valid = zscores.notna().sum(axis=1)
    denominator = valid.replace(0, np.nan)

    above = zscores.gt(threshold).sum(axis=1).div(denominator).mul(100.0)
    below = zscores.lt(-threshold).sum(axis=1).div(denominator).mul(100.0)

    breadth = pd.DataFrame(
        {
            "Date": zscores.index,
            "Above": above,
            "Below": below,
            "Net": above - below,
            "Valid": valid,
        }
    )
    return breadth.dropna(subset=["Above", "Below", "Net"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def configure_date_axis(ax: plt.Axes, dates: pd.Series) -> None:
    span_years = max((dates.iloc[-1] - dates.iloc[0]).days / 365.25, 1.0)
    interval = 2 if span_years >= 12 else 1
    ax.xaxis.set_major_locator(mdates.YearLocator(base=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", alpha=0.22)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_extremes_figure(df: pd.DataFrame, window: int, threshold: float) -> Path:
    latest = df.iloc[-1]
    start_year = int(df["Date"].dt.year.min())
    end_year = int(df["Date"].dt.year.max())

    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor="white")
    ax.plot(df["Date"], df["Above"], linewidth=1.6, label=f"Above +{threshold:.1f}σ")
    ax.plot(df["Date"], df["Below"], linewidth=1.6, label=f"Below -{threshold:.1f}σ")

    ax.set_title(
        f"S&P 500 regression breadth — extreme participation ({start_year}-{end_year})",
        loc="left",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_ylabel("Share of constituents (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper left", ncols=2)
    configure_date_axis(ax, df["Date"])

    ax.text(
        0.995,
        0.97,
        f"Latest: +{threshold:.1f}σ {latest['Above']:.1f}%   |   -{threshold:.1f}σ {latest['Below']:.1f}%
"
        f"Valid constituents: {int(latest['Valid'])}   |   Window: {window} weeks",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
    )

    fig.tight_layout()
    fig.savefig(FIGURE_EXTREMES, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return FIGURE_EXTREMES


def save_net_figure(df: pd.DataFrame, window: int, threshold: float) -> Path:
    latest = df.iloc[-1]
    start_year = int(df["Date"].dt.year.min())
    end_year = int(df["Date"].dt.year.max())

    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor="white")
    ax.plot(df["Date"], df["Net"], linewidth=1.65)
    ax.fill_between(df["Date"], df["Net"], 0, alpha=0.16)
    ax.axhline(0, linewidth=0.9, linestyle=(0, (3, 3)))

    ax.set_title(
        f"S&P 500 regression breadth — net extreme balance ({start_year}-{end_year})",
        loc="left",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_ylabel("Above +σ minus below -σ (percentage points)")
    ax.set_ylim(-100, 100)
    configure_date_axis(ax, df["Date"])

    ax.text(
        0.995,
        0.97,
        f"Latest net: {latest['Net']:+.1f} pp
"
        f"Threshold: ±{threshold:.1f}σ   |   Window: {window} weeks",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
    )

    fig.tight_layout()
    fig.savefig(FIGURE_NET, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return FIGURE_NET


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S&P 500 regression-breadth analysis with two PNG outputs."
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help=f"Rolling regression window in weeks (default: {DEFAULT_WINDOW}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Extreme residual threshold in sigma (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore the temporary price cache and download fresh data.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.window < 20:
        raise SystemExit("--window must be at least 20 weeks.")
    if not math.isfinite(args.threshold) or args.threshold <= 0:
        raise SystemExit("--threshold must be a positive number.")

    print("Loading S&P 500 universe...")
    tickers = get_sp500_tickers()
    print(f"Universe: {len(tickers)} ticker(s)")

    prices = load_weekly_prices(tickers, refresh=args.refresh)
    print(f"Usable price series: {prices.shape[1]}")

    print("Computing vectorized rolling regressions...")
    breadth = compute_breadth(prices, args.window, args.threshold)
    if breadth.empty:
        raise RuntimeError("No breadth observations could be calculated.")

    figure_1 = save_extremes_figure(breadth, args.window, args.threshold)
    figure_2 = save_net_figure(breadth, args.window, args.threshold)

    latest = breadth.iloc[-1]
    print(f"Latest date: {latest['Date'].date()}")
    print(f"Above +{args.threshold:.1f}σ: {latest['Above']:.2f}%")
    print(f"Below -{args.threshold:.1f}σ: {latest['Below']:.2f}%")
    print(f"Net breadth: {latest['Net']:+.2f} pp")
    print(f"Saved: {figure_1}")
    print(f"Saved: {figure_2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
