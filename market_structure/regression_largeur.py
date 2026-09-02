# -*- coding: utf-8 -*-
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".regression_cache"
PRICE_CACHE = CACHE_DIR / "weekly_prices_1995.csv"
RESULT_CACHE = CACHE_DIR / "regression_extremes_l500_s15.csv"

START_DATE = "1995-01-01"
WINDOW = 500
THRESHOLD = 1.5
INDEX_LABEL = "S&P 500"

FALLBACK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "GOOGL", "META", "NFLX",
    "AMZN", "TSLA", "HD", "COST", "WMT", "PG", "JPM", "BAC", "V",
    "MA", "LLY", "UNH", "JNJ", "XOM", "CVX", "CAT", "GE", "LIN",
    "NEE", "PLD",
]


def _cache_fresh(path: Path, max_age_hours: int = 20) -> bool:
    if not path.exists():
        return False
    age_hours = (pd.Timestamp.now().timestamp() - path.stat().st_mtime) / 3600
    return age_hours <= max_age_hours


def get_sp500_tickers() -> list[str]:
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
        table = pd.read_html(io.StringIO(html))[0]
        return (
            table["Symbol"]
            .astype(str)
            .str.replace(".", "-", regex=False)
            .dropna()
            .drop_duplicates()
            .tolist()
        )
    except Exception:
        return FALLBACK_TICKERS.copy()


def download_weekly_prices(tickers: Iterable[str] | None = None, force: bool = False) -> pd.DataFrame:
    CACHE_DIR.mkdir(exist_ok=True)
    if not force and _cache_fresh(PRICE_CACHE):
        prices = pd.read_csv(PRICE_CACHE, index_col=0)
        prices.index = pd.to_datetime(prices.index, errors="coerce")
        return prices.dropna(how="all")

    import yfinance as yf

    try:
        yf.set_tz_cache_location(str(CACHE_DIR / "yfinance_tz"))
    except Exception:
        pass

    symbols = list(tickers) if tickers is not None else get_sp500_tickers()
    data = yf.download(
        symbols,
        start=START_DATE,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
        group_by="ticker",
    )
    if data.empty:
        raise RuntimeError("Téléchargement Yahoo Finance vide.")

    if isinstance(data.columns, pd.MultiIndex):
        close = pd.DataFrame(
            {ticker: data[(ticker, "Close")] for ticker in symbols if (ticker, "Close") in data.columns}
        )
    else:
        close = data[["Close"]].rename(columns={"Close": symbols[0]})

    weekly = close.resample("W-FRI").last().ffill().dropna(how="all")
    weekly.to_csv(PRICE_CACHE)
    return weekly


def _last_regression_z(values: np.ndarray, min_periods: int) -> float:
    mask = np.isfinite(values)
    if mask.sum() < min_periods:
        return np.nan
    y = values[mask]
    x = np.arange(len(values), dtype=float)[mask]
    if len(y) < 3:
        return np.nan
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (intercept + slope * x)
    std = residual.std(ddof=1)
    if not std or np.isnan(std):
        return np.nan
    return float(residual[-1] / std)


def compute_extreme_breadth(
    window: int = WINDOW,
    threshold: float = THRESHOLD,
    force: bool = False,
) -> pd.DataFrame:
    CACHE_DIR.mkdir(exist_ok=True)
    if not force and _cache_fresh(RESULT_CACHE):
        result = pd.read_csv(RESULT_CACHE)
        result["Date"] = pd.to_datetime(result["Date"], errors="coerce")
        return result.dropna(subset=["Date"]).sort_values("Date")

    weekly = download_weekly_prices(force=force)
    log_prices = np.log(weekly.replace(0, np.nan))
    min_periods = max(80, int(window * 0.45))

    zscores = log_prices.rolling(window=window, min_periods=min_periods).apply(
        lambda values: _last_regression_z(values, min_periods=min_periods),
        raw=True,
    )
    valid = zscores.notna().sum(axis=1).replace(0, np.nan)
    above = (zscores > threshold).sum(axis=1) / valid * 100
    below = (zscores < -threshold).sum(axis=1) / valid * 100

    result = pd.DataFrame(
        {
            "Date": zscores.index,
            "Above": above,
            "Below": below,
            "Net": above - below,
            "Valid": valid,
            "Window": window,
            "Threshold": threshold,
        }
    ).dropna(subset=["Above", "Below", "Net"])
    result.to_csv(RESULT_CACHE, index=False)
    return result


def plot_extreme_breadth(data: pd.DataFrame | None = None) -> None:
    import matplotlib.pyplot as plt

    df = data if data is not None else compute_extreme_breadth()
    if df.empty:
        raise RuntimeError("Aucune donnée disponible pour tracer les graphiques.")

    start = int(pd.to_datetime(df["Date"]).dt.year.min())
    end = int(pd.to_datetime(df["Date"]).dt.year.max())
    window = int(df["Window"].dropna().iloc[-1])
    threshold = float(df["Threshold"].dropna().iloc[-1])

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df["Date"], df["Above"], label=f"Au-dessus de +{threshold:.1f}σ", linewidth=1.4)
    ax1.plot(df["Date"], df["Below"], label=f"Au-dessous de -{threshold:.1f}σ", linewidth=1.4)
    ax1.set_title(
        f"Part des actions au-delà de ±{threshold:.1f}σ du canal de régression "
        f"(hebdo, L={window}) - {start}-{end}"
    )
    ax1.set_ylabel("Pourcentage d'actions")
    ax1.set_xlabel("Date")
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.22)
    ax1.legend()

    fig2, ax2 = plt.subpots(figsize=(12, 6))
    ax2.plot(df["Date"], df["Net"], linewidth=1.4)
    ax2.fill_between(df["Date"], df["Net"], 0, alpha=0.25)
    ax2.axhline(0, color="black", linewidth=0.9)
    ax2.set_title(
        f"Différence (% au-dessus - % au-dessous) - {start}-{end} "
        f"(hebdo, L={window}, ±{threshold:.1f}σ)"
    )
    ax2.set_ylabel("Différence (points de %)")
    ax2.set_xlabel("Date")
    ax2.set_ylim(-100, 100)
    ax2.grid(True, alpha=0.22)

    plt.show()


def main() -> None:
    plot_extreme_breadth()


if __name__ == "__main__":
    main()
