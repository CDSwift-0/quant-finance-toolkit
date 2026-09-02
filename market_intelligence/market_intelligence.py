#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Intelligence local.

Commande:
    python3 ~/Desktop/market_intelligence.py
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import math
import os
import queue
import re
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".market_intelligence_cache"
PRICE_CACHE = CACHE_DIR / "market_prices_10y.csv"
PUT_CALL_CACHE = CACHE_DIR / "spx_put_call_open_interest.csv"
US_CDS_CACHE = CACHE_DIR / "us_cds_5y.csv"
PERIODS = {"3 mois": 63, "6 mois": 126, "1 an": 252, "3 ans": 756, "5 ans": 1260, "10 ans": 2520}
SPX_OPEN_INTEREST_URL = "https://proxy.optionsanalysissuite.com/scanner/history"
SPX_OPEN_INTEREST_DAYS = 7000
US_CDS_URL = "https://www.worldgovernmentbonds.com/cds-historical-data/united-states/5-years/"

ASSETS = {
    "SPY": ("S&P 500", "Indices"),
    "QQQ": ("Nasdaq 100", "Indices"),
    "IWM": ("Russell 2000", "Indices"),
    "TLT": ("Obligations longues", "Macro"),
    "GLD": ("Or", "Macro"),
    "UUP": ("Dollar", "Macro"),
    "USO": ("Pétrole", "Macro"),
    "^MOVE": ("MOVE Index", "Macro"),
    "XLK": ("Technology", "Secteurs"),
    "XLC": ("Communication", "Secteurs"),
    "XLY": ("Consumer Discretionary", "Secteurs"),
    "XLF": ("Financials", "Secteurs"),
    "XLI": ("Industrials", "Secteurs"),
    "XLV": ("Health Care", "Secteurs"),
    "XLE": ("Energy", "Secteurs"),
    "XLP": ("Consumer Staples", "Secteurs"),
    "XLU": ("Utilities", "Secteurs"),
    "XLB": ("Materials", "Secteurs"),
    "XLRE": ("Real Estate", "Secteurs"),
    "^VIX": ("VIX", "Volatilité"),
    "^TNX": ("Taux US 10 ans", "Macro"),
}

SECTORS = {
    "XLK": "Technology",
    "XLC": "Communication",
    "XLY": "Consumer Discretionary",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLV": "Health Care",
    "XLE": "Energy",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
}

CROSS_ASSETS = {
    "SPY": "Actions US",
    "QQQ": "Nasdaq",
    "IWM": "Small caps",
    "TLT": "Obligations",
    "GLD": "Or",
    "UUP": "Dollar",
    "USO": "Pétrole",
}

CYCLICAL = ["XLK", "XLY", "XLF", "XLI", "XLE"]
DEFENSIVE = ["XLV", "XLP", "XLU"]

SOURCE_STATE = {"prices": "unknown", "put_call": "unknown", "cds": "unknown"}
_RUNTIME_BUNDLE: Optional[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = None
_RUNTIME_BUNDLE_AT = 0.0
_RUNTIME_BUNDLE_LOCK = threading.Lock()
RUNTIME_BUNDLE_TTL = 300


def ensure_cache() -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    os.environ["YFINANCE_CACHE_DIR"] = str(CACHE_DIR / "yfinance")


def read_price_cache(path: Path, max_age_hours: int = 18) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        age_hours = (pd.Timestamp.now().timestamp() - path.stat().st_mtime) / 3600
        if age_hours > max_age_hours:
            return None
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, errors="coerce")
        return df.dropna(how="all")
    except Exception:
        return None


def close_from_yfinance(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        level_zero = set(raw.columns.get_level_values(0))
        level_one = set(raw.columns.get_level_values(1))
        if "Close" in level_zero:
            closes = raw["Close"].copy()
        elif "Adj Close" in level_zero:
            closes = raw["Adj Close"].copy()
        elif "Close" in level_one:
            closes = raw.xs("Close", axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        column = "Close" if "Close" in raw.columns else raw.columns[-1]
        closes = raw[[column]].rename(columns={column: tickers[0]})
    closes.index = pd.to_datetime(closes.index, errors="coerce").tz_localize(None)
    closes = closes[~closes.index.isna()].sort_index()
    return closes.apply(pd.to_numeric, errors="coerce").dropna(how="all")


def download_prices(tickers: list[str]) -> pd.DataFrame:
    """Download market prices, preferring fresh cache, then live data, then stale cache."""
    ensure_cache()
    cached = read_price_cache(PRICE_CACHE)
    if cached is not None:
        available = [ticker for ticker in tickers if ticker in cached.columns]
        missing = [ticker for ticker in tickers if ticker not in available]
        if not missing and len(available) >= max(8, int(len(tickers) * 0.65)):
            SOURCE_STATE["prices"] = "cache"
            return cached[available]

    stale = read_price_cache(PRICE_CACHE, max_age_hours=24 * 365)
    try:
        import yfinance as yf

        try:
            yf.set_tz_cache_location(str(CACHE_DIR / "yfinance_tz"))
        except Exception:
            pass
        raw = yf.download(
            tickers,
            period="10y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        closes = close_from_yfinance(raw, tickers)
        if not closes.empty:
            closes.to_csv(PRICE_CACHE)
            SOURCE_STATE["prices"] = "live"
            return closes
    except Exception:
        pass

    if stale is not None:
        available = [ticker for ticker in tickers if ticker in stale.columns]
        if len(available) >= max(8, int(len(tickers) * 0.65)):
            SOURCE_STATE["prices"] = "stale cache"
            return stale[available]

    SOURCE_STATE["prices"] = "synthetic fallback"
    return fallback_prices(tickers)


def fetch_us_cds_data() -> pd.DataFrame:
    ensure_cache()
    cached = read_price_cache(US_CDS_CACHE, max_age_hours=18)
    if cached is not None and "US_CDS_5Y" in cached.columns:
        SOURCE_STATE["cds"] = "cache"
        return cached.reset_index().rename(columns={cached.index.name or "index": "Date"})
    stale = read_price_cache(US_CDS_CACHE, max_age_hours=24 * 365)
    try:
        import requests
        import json

        session = requests.Session()
        page = session.get(US_CDS_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        page.raise_for_status()
        match = re.search(r"var\s+jsGlobalVars\s*=\s*(\{.*?\});\s*</script>", page.text, re.S)
        if not match:
            raise ValueError("Configuration CDS introuvable")
        config = json.loads(match.group(1))
        response = session.post(
            config["ENDPOINT"],
            json={"GLOBALVAR": config},
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Origin": "https://www.worldgovernmentbonds.com",
                "Referer": US_CDS_URL,
            },
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        quote = payload.get("result", {}).get("quote", {})
        rows = quote.values() if isinstance(quote, dict) else quote
        data = pd.DataFrame(rows)
        result = pd.DataFrame(
            {
                "Date": pd.to_datetime(data.get("DATA_VAL"), errors="coerce"),
                "US_CDS_5Y": pd.to_numeric(data.get("CLOSE_VAL"), errors="coerce"),
            }
        ).dropna()
        result = result.sort_values("Date").drop_duplicates(subset=["Date"])
        if not result.empty:
            result.set_index("Date").to_csv(US_CDS_CACHE)
            SOURCE_STATE["cds"] = "live"
            return result
    except Exception:
        if stale is not None and "US_CDS_5Y" in stale.columns:
            SOURCE_STATE["cds"] = "stale cache"
            return stale.reset_index().rename(columns={stale.index.name or "index": "Date"})
    SOURCE_STATE["cds"] = "unavailable"
    return pd.DataFrame(columns=["Date", "US_CDS_5Y"])


def fetch_put_call_data() -> pd.DataFrame:
    ensure_cache()
    cached = read_price_cache(PUT_CALL_CACHE, max_age_hours=18)
    if cached is not None and {"Ratio", "Puts", "Calls", "OpenInterest"}.issubset(cached.columns):
        SOURCE_STATE["put_call"] = "cache"
        return cached.reset_index().rename(columns={cached.index.name or "index": "Date"})

    stale = read_price_cache(PUT_CALL_CACHE, max_age_hours=24 * 365)
    try:
        import requests

        response = requests.get(
            SPX_OPEN_INTEREST_URL,
            params={"symbol": "SPX", "days": str(SPX_OPEN_INTEREST_DAYS)},
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
                "Referer": "https://www.optionsanalysissuite.com/index/spx/open-interest",
                "Origin": "https://www.optionsanalysissuite.com",
            },
            timeout=24,
        )
        response.raise_for_status()
        payload = response.json()
        table = pd.DataFrame(payload.get("data", []))
        table.columns = [str(col).strip() for col in table.columns]
        required = {"market_date", "call_oi", "put_oi"}
        if not required.issubset(set(table.columns)):
            raise ValueError("Colonnes SPX OI absentes")
        calls = pd.to_numeric(table["call_oi"], errors="coerce")
        puts = pd.to_numeric(table["put_oi"], errors="coerce")
        ratio = (puts / calls).replace([np.inf, -np.inf], np.nan)
        data = pd.DataFrame(
            {
                "Date": pd.to_datetime(table["market_date"], errors="coerce"),
                "Ratio": ratio,
                "Puts": puts,
                "Calls": calls,
                "OpenInterest": puts + calls,
                "Spot": pd.to_numeric(table.get("spot_price"), errors="coerce") if "spot_price" in table else np.nan,
            }
        ).dropna(subset=["Date", "Ratio"])
        data = data.sort_values("Date")
        if not data.empty:
            data.set_index("Date").to_csv(PUT_CALL_CACHE)
            SOURCE_STATE["put_call"] = "live"
            return data
    except Exception:
        pass

    if stale is not None and {"Ratio", "Puts", "Calls", "OpenInterest"}.issubset(stale.columns):
        SOURCE_STATE["put_call"] = "stale cache"
        return stale.reset_index().rename(columns={stale.index.name or "index": "Date"})
    SOURCE_STATE["put_call"] = "unavailable"
    return pd.DataFrame(columns=["Date", "Ratio", "Puts", "Calls", "OpenInterest", "Spot"])


def fallback_put_call_data(dates: pd.Index) -> pd.DataFrame:
    rng = np.random.default_rng(23)
    clean_dates = pd.to_datetime(pd.Series(dates), errors="coerce").dropna()
    if clean_dates.empty:
        clean_dates = pd.Series(pd.date_range(end=pd.Timestamp.today().normalize(), periods=252, freq="B"))
    clean_dates = clean_dates.tail(2520).reset_index(drop=True)
    ratio = np.clip(1.80 + np.cumsum(rng.normal(0, 0.012, len(clean_dates))), 1.15, 2.55)
    calls = np.clip(8_500_000 + rng.normal(0, 900_000, len(clean_dates)), 2_000_000, None)
    puts = calls * ratio
    return pd.DataFrame({"Date": clean_dates, "Ratio": ratio, "Puts": puts, "Calls": calls, "OpenInterest": puts + calls, "Spot": np.nan})


def put_call_context(pc_full: pd.DataFrame, pc_period: pd.DataFrame) -> dict[str, Any]:
    ratio = last(pc_full.get("Ratio"))
    puts = last(pc_full.get("Puts"))
    calls = last(pc_full.get("Calls"))
    open_interest = last(pc_full.get("OpenInterest"))
    ratio_series = pd.to_numeric(pc_full.get("Ratio", pd.Series(dtype=float)), errors="coerce").dropna()
    ma20 = float(ratio_series.tail(20).mean()) if not ratio_series.empty else float("nan")
    percentile = float((ratio_series <= ratio).mean() * 100) if len(ratio_series) and not pd.isna(ratio) else float("nan")

    if pd.isna(ratio):
        regime = "Signal indisponible"
        note = "Le ratio Put/Call basé sur l'open interest SPX n'a pas pu être calculé