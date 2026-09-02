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
        note = "Le ratio Put/Call basé sur l'open interest SPX n'a pas pu être calculé pour l'instant."
        tone = "muted"
    elif percentile >= 88 or ratio >= 2.15:
        regime = "Protection élevée"
        note = "L'open interest put domine nettement : le positionnement options reste défensif sur le S&P 500."
        tone = "red"
    elif percentile <= 15 or ratio <= 1.45:
        regime = "Complacence relative"
        note = "L'open interest call pèse davantage que d'habitude : le marché montre moins de demande structurelle de protection."
        tone = "gold"
    else:
        regime = "Équilibre SPX"
        note = "Le ratio OI reste dans une zone intermédiaire : le signal options ne montre pas d'excès clair."
        tone = "blue"

    return {
        "ratio": ratio,
        "puts": puts,
        "calls": calls,
        "open_interest": open_interest,
        "ma20": ma20,
        "percentile": percentile,
        "regime": regime,
        "note": note,
        "tone": tone,
    }


def fallback_prices(tickers: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=2520, freq="B")
    data: dict[str, np.ndarray] = {}
    for index, ticker in enumerate(tickers):
        if ticker == "^VIX":
            values = np.clip(18 + rng.normal(0, 1.2, len(dates)).cumsum() * 0.08, 11, 36)
        elif ticker == "^TNX":
            values = np.clip(42 + rng.normal(0, 0.55, len(dates)).cumsum() * 0.03, 32, 55)
        else:
            drift = 0.00022 + (index % 4) * 0.00005
            vol = 0.009 + (index % 6) * 0.0016
            values = 100 * np.exp(np.cumsum(rng.normal(drift, vol, len(dates))))
        data[ticker] = values
    return pd.DataFrame(data, index=dates)


def fallback_us_cds_data(dates: pd.Index) -> pd.DataFrame:
    rng = np.random.default_rng(41)
    clean_dates = pd.to_datetime(pd.Series(dates), errors="coerce").dropna()
    if clean_dates.empty:
        clean_dates = pd.Series(pd.date_range(end=pd.Timestamp.today().normalize(), periods=2520, freq="B"))
    clean_dates = clean_dates.tail(2520).reset_index(drop=True)
    values = np.clip(22 + np.cumsum(rng.normal(0, 0.12, len(clean_dates))), 8, 75)
    return pd.DataFrame({"Date": clean_dates, "US_CDS_5Y": values})


def slice_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    return df.tail(PERIODS.get(period, 63)).copy()


def fmt_pct(value: Any, digits: int = 1, signed: bool = True) -> str:
    try:
        if pd.isna(value):
            return "n.d."
        prefix = "+" if signed and float(value) > 0 else ""
        return f"{prefix}{float(value):.{digits}f}%"
    except Exception:
        return "n.d."


def fmt_num(value: Any, digits: int = 2) -> str:
    try:
        if pd.isna(value):
            return "n.d."
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n.d."


def fmt_int(value: Any) -> str:
    try:
        if pd.isna(value):
            return "n.d."
        return f"{float(value):,.0f}".replace(",", " ")
    except Exception:
        return "n.d."


def perf(series: Optional[pd.Series]) -> float:
    if series is None:
        return float("nan")
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 2 or clean.iloc[0] == 0:
        return float("nan")
    return float((clean.iloc[-1] / clean.iloc[0] - 1) * 100)


def last(series: Optional[pd.Series]) -> float:
    if series is None:
        return float("nan")
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.iloc[-1]) if not clean.empty else float("nan")


def performance_frame(prices: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    rows = []
    for ticker, label in labels.items():
        if ticker in prices:
            value = perf(prices[ticker])
            if not pd.isna(value):
                rows.append({"Ticker": ticker, "Nom": label, "Performance": value, "Valeur": last(prices[ticker])})
    return pd.DataFrame(rows)


def compute_breadth(full_prices: pd.DataFrame) -> pd.DataFrame:
    cols = [ticker for ticker in SECTORS if ticker in full_prices.columns]
    if not cols:
        return pd.DataFrame(columns=["Date", "Breadth", "RawAbove", "AvgDistance"])
    sector_prices = full_prices[cols].dropna(how="all")
    ma50 = sector_prices.rolling(50, min_periods=40).mean()
    distance = (sector_prices / ma50 - 1) * 100
    valid_count = distance.notna().sum(axis=1)
    valid_safe = valid_count.replace(0, np.nan)
    raw_above = sector_prices.gt(ma50).where(distance.notna()).sum(axis=1) / valid_safe * 100
    score = 100 / (1 + np.exp(-(distance / 3.2)))
    breadth = score.mean(axis=1).ewm(span=8, adjust=False, min_periods=3).mean()
    avg_distance = distance.mean(axis=1)
    data = pd.DataFrame(
        {
            "Date": sector_prices.index,
            "Breadth": breadth.clip(lower=0, upper=100).values,
            "RawAbove": raw_above.clip(lower=0, upper=100).values,
            "AvgDistance": avg_distance.values,
        }
    )
    valid_mask = valid_count.to_numpy() >= max(5, int(len(cols) * 0.6))
    return data.loc[valid_mask].dropna(subset=["Breadth"])


def sector_mm50_state(full_prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker, label in SECTORS.items():
        if ticker not in full_prices.columns:
            continue
        series = pd.to_numeric(full_prices[ticker], errors="coerce").dropna()
        if series.empty:
            continue
        ma50 = series.rolling(50, min_periods=18).mean()
        latest = float(series.iloc[-1])
        latest_ma = float(ma50.iloc[-1]) if not pd.isna(ma50.iloc[-1]) else float("nan")
        if pd.isna(latest_ma) or latest_ma == 0:
            continue
        distance = (latest / latest_ma - 1) * 100
        rows.append(
            {
                "Ticker": ticker,
                "Nom": label,
                "Close": latest,
                "MM50": latest_ma,
                "Distance": distance,
                "Above": latest >= latest_ma,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["Ticker", "Nom", "Close", "MM50", "Distance", "Above"])
    return pd.DataFrame(rows).sort_values("Distance", ascending=False)


def load_source_bundle(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load independent sources in parallel and reuse them during period changes."""
    global _RUNTIME_BUNDLE, _RUNTIME_BUNDLE_AT
    now = time.time()
    with _RUNTIME_BUNDLE_LOCK:
        if not force and _RUNTIME_BUNDLE is not None and now - _RUNTIME_BUNDLE_AT <= RUNTIME_BUNDLE_TTL:
            prices, put_call, cds = _RUNTIME_BUNDLE
            return prices.copy(), put_call.copy(), cds.copy()

    tickers = list(ASSETS.keys())
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="market-data") as pool:
        future_prices = pool.submit(download_prices, tickers)
        future_put_call = pool.submit(fetch_put_call_data)
        future_cds = pool.submit(fetch_us_cds_data)
        prices = future_prices.result()
        put_call = future_put_call.result()
        cds = future_cds.result()

    with _RUNTIME_BUNDLE_LOCK:
        _RUNTIME_BUNDLE = (prices.copy(), put_call.copy(), cds.copy())
        _RUNTIME_BUNDLE_AT = time.time()
    return prices, put_call, cds


def build_market_data(period: str, force_sources: bool = False) -> dict[str, Any]:
    full_prices, pc_full, cds_full = load_source_bundle(force=force_sources)
    prices = slice_period(full_prices, period)
    if pc_full.empty:
        SOURCE_STATE["put_call"] = "synthetic fallback"
        pc_full = fallback_put_call_data(full_prices.index)
    pc_full["Date"] = pd.to_datetime(pc_full["Date"], errors="coerce")
    pc_full = pc_full.dropna(subset=["Date"]).sort_values("Date")
    pc_chart = slice_period(pc_full.set_index("Date"), period).reset_index()
    pc_chart["MM20"] = pc_full.set_index("Date")["Ratio"].rolling(20, min_periods=8).mean().reindex(pc_chart["Date"]).values
    if "SPY" in full_prices:
        pc_chart["SPY"] = full_prices["SPY"].ffill().reindex(pc_chart["Date"], method="ffill").values
    pc_stats = put_call_context(pc_full, pc_chart)
    sector_perf = performance_frame(prices, SECTORS)
    cross_perf = performance_frame(prices, CROSS_ASSETS)
    labels = {ticker: name for ticker, (name, _cat) in ASSETS.items() if not ticker.startswith("^")}
    all_perf = performance_frame(prices, labels)
    breadth = slice_period(compute_breadth(full_prices).set_index("Date"), period).reset_index()
    if "SPY" in full_prices and not breadth.empty:
        bench = pd.to_numeric(full_prices["SPY"].ffill().reindex(breadth["Date"], method="ffill"), errors="coerce")
        if bench.notna().sum() >= 2 and not math.isclose(float(bench.max()), float(bench.min())):
            breadth["Benchmark"] = ((bench - bench.min()) / (bench.max() - bench.min()) * 100).values
    sector_state = sector_mm50_state(full_prices)

    if cds_full.empty:
        SOURCE_STATE["cds"] = "synthetic fallback"
        cds_full = fallback_us_cds_data(full_prices.index)
    cds_full["Date"] = pd.to_datetime(cds_full["Date"], errors="coerce")
    cds_full = cds_full.dropna(subset=["Date"]).sort_values("Date")
    cds_chart = slice_period(cds_full.set_index("Date"), period).reset_index()
    if "SPY" in full_prices and not cds_chart.empty:
        cds_chart["SPY"] = full_prices["SPY"].ffill().reindex(cds_chart["Date"], method="ffill").values
    cds_cols = [col for col in ["US_CDS_5Y", "SPY"] if col in cds_chart]
    if cds_cols:
        cds_chart = cds_chart.dropna(subset=cds_cols, how="all")

    spy = perf(prices.get("SPY"))
    qqq = perf(prices.get("QQQ"))
    iwm = perf(prices.get("IWM"))
    vix = last(full_prices.get("^VIX"))
    tnx = last(full_prices.get("^TNX"))
    if not pd.isna(tnx) and tnx > 15:
        tnx = tnx / 10

    participation = last(breadth.get("Breadth")) if not breadth.empty else float("nan")
    indexed = sector_perf.set_index("Ticker") if not sector_perf.empty else pd.DataFrame()
    cyc = indexed.reindex(CYCLICAL)["Performance"].mean() if not indexed.empty else float("nan")
    defensive = indexed.reindex(DEFENSIVE)["Performance"].mean() if not indexed.empty else float("nan")
    leadership = float(cyc - defensive) if not pd.isna(cyc) and not pd.isna(defensive) else float("nan")

    best_sector = "n.d."
    worst_sector = "n.d."
    if not sector_perf.empty:
        best = sector_perf.sort_values("Performance", ascending=False).iloc[0]
        worst = sector_perf.sort_values("Performance", ascending=True).iloc[0]
        best_sector = f"{best['Nom']} {fmt_pct(best['Performance'])}"
        worst_sector = f"{worst['Nom']} {fmt_pct(worst['Performance'])}"

    if not pd.isna(vix) and (vix >= 25 or (not pd.isna(spy) and spy <= -4)):
        regime = "Risque élevé"
        note = "La volatilité impose une lecture défensive. Les rebonds doivent être confirmés par la participation."
        tone = "red"
    elif not pd.isna(spy) and spy > 2 and not pd.isna(participation) and participation >= 55 and (pd.isna(vix) or vix < 20):
        regime = "Risk-on discipliné"
        note = "Le marché reste constructif : performance positive, volatilité contenue et participation correcte."
        tone = "green"
    elif not pd.isna(participation) and participation < 45:
        regime = "Marché sélectif"
        note = "La largeur manque de force. Les leaders comptent davantage que l'exposition générale."
        tone = "gold"
    elif not pd.isna(leadership) and leadership < -2:
        regime = "Rotation défensive"
        note = "Les secteurs défensifs reprennent du poids. L'appétit pour le risque est moins évident."
        tone = "gold"
    else:
        regime = "Équilibre à confirmer"
        note = "Les signaux sont mixtes. La volatilité et la participation doivent confirmer la direction."
        tone = "blue"

    corr_cols = [ticker for ticker in CROSS_ASSETS if ticker in prices.columns]
    returns = prices[corr_cols].pct_chang