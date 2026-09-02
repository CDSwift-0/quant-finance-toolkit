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
    returns = prices[corr_cols].pct_change(fill_method=None).dropna(how="all") if corr_cols else pd.DataFrame()
    corr = returns.corr() if len(returns) >= 5 else pd.DataFrame()

    price_chart = pd.DataFrame({"Date": prices.index})
    if "SPY" in prices:
        price_chart["SPY"] = prices["SPY"].ffill().values
        price_chart["MM20"] = full_prices["SPY"].rolling(20, min_periods=8).mean().reindex(prices.index).values
        price_chart["MM50"] = full_prices["SPY"].rolling(50, min_periods=18).mean().reindex(prices.index).values
    vix_chart = pd.DataFrame({"Date": prices.index})
    if "^VIX" in prices:
        vix_chart["VIX"] = prices["^VIX"].values
    if "SPY" in prices:
        vix_chart["SPY"] = prices["SPY"].ffill().values
    move_chart = pd.DataFrame({"Date": prices.index})
    if "^MOVE" in prices:
        move_chart["MOVE"] = prices["^MOVE"].values
        move_chart["MM20"] = full_prices["^MOVE"].rolling(20, min_periods=8).mean().reindex(prices.index).values
    if "SPY" in prices:
        move_chart["SPY"] = prices["SPY"].ffill().values

    style_chart = pd.DataFrame({"Date": prices.index})
    if {"XLY", "XLP"}.issubset(full_prices.columns):
        ratio = (full_prices["XLY"] / full_prices["XLP"]).replace([np.inf, -np.inf], np.nan)
        style_chart["XLY/XLP"] = ratio.reindex(prices.index).values
    if "SPY" in prices:
        style_chart["SPY"] = prices["SPY"].ffill().values
    if "XLY/XLP" in style_chart:
        style_chart = style_chart.dropna(subset=["XLY/XLP"])

    return {
        "source_state": dict(SOURCE_STATE),
        "updated_at": pd.Timestamp.now(),
        "all_perf": all_perf,
        "sector_perf": sector_perf,
        "cross_perf": cross_perf,
        "breadth": breadth,
        "sector_state": sector_state,
        "corr": corr,
        "price_chart": price_chart,
        "vix_chart": vix_chart,
        "move_chart": move_chart,
        "style_chart": style_chart,
        "cds_chart": cds_chart,
        "put_call_chart": pc_chart,
        "put_call": pc_stats,
        "kpis": {
            "S&P 500": fmt_pct(spy),
            "Nasdaq": fmt_pct(qqq),
            "Small caps": fmt_pct(iwm),
            "VIX": fmt_num(vix, 1),
            "SPX OI P/C": fmt_num(pc_stats["ratio"], 2),
            "Taux 10 ans": f"{fmt_num(tnx, 2)}%",
            "Participation": fmt_pct(participation, 0, signed=False),
        },
        "summary": {
            "regime": regime,
            "note": note,
            "tone": tone,
            "best_sector": best_sector,
            "worst_sector": worst_sector,
            "leadership": leadership,
            "period": period,
        },
    }


class MarketIntelligenceDashboard(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Market Intelligence")
        screen_w = max(self.winfo_screenwidth(), 1280)
        screen_h = max(self.winfo_screenheight(), 820)
        width = min(1920, max(1280, int(screen_w * 0.96)))
        height = min(1120, max(820, int(screen_h * 0.93)))
        self.geometry(f"{width}x{height}+{max(0, (screen_w-width)//2)}+{max(0, (screen_h-height)//2)}")
        self.minsize(1280, 820)
        self.colors = {
            "bg": "#f3f5f7", "panel": "#ffffff", "panel_alt": "#f8fafc", "panel_soft": "#f1f5f9",
            "ink": "#0f172a", "ink_soft": "#1e293b", "muted": "#64748b", "muted_light": "#94a3b8",
            "line": "#cbd5e1", "line_soft": "#e2e8f0", "grid": "#e7edf3", "grid_soft": "#eff3f7",
            "gold": "#b7791f", "gold_bg": "#f7ecd5", "green": "#0f9f6e", "green_bg": "#ddf3ea",
            "red": "#d64545", "red_bg": "#f8e3e3", "blue": "#2563eb", "blue_bg": "#e3ecff",
        }
        self.configure(bg=self.colors["bg"])
        self.queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.status = tk.StringVar(value="Prêt")
        self.period = tk.StringVar(value="1 an")
        self.spinner_job: Optional[str] = None
        self.spinner_angle = 0
        self.data: Optional[dict[str, Any]] = None
        self.series_visibility: dict[str, bool] = {}
        self._request_force_sources = False
        self.regular_panel_height = 500
        self.breadth_panel_height = 690
        self.range_text = tk.StringVar(value="Plage calculée après chargement")
        self.session_text = tk.StringVar(value="")
        self._scroll_job: Optional[str] = None
        self._scroll_target_y: Optional[float] = None
        self._styles()
        self._shell()
        self.after(180, self.refresh)

    def _styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Market.Treeview", background=self.colors["panel"], fieldbackground=self.colors["panel"], foreground=self.colors["ink"], borderwidth=0, rowheight=31, font=("Avenir Next", 10))
        style.configure("Market.Treeview.Heading", background=self.colors["panel_alt"], foreground=self.colors["muted"], borderwidth=0, font=("Avenir Next", 10, "bold"))

    def _shell(self) -> None:
        self.page = tk.Frame(self, bg=self.colors["bg"])
        self.page.pack(fill="both", expand=True)
        self._header()
        self._period_control(self.page).pack(fill="x", padx=20, pady=(0, 12))

        self.kpi_grid = tk.Frame(self.page, bg=self.colors["bg"])
        self.kpi_grid.pack(fill="x", padx=20, pady=(0, 11))
        for col in range(7):
            self.kpi_grid.grid_columnconfigure(col, weight=1, uniform="kpi")

        self.footer = tk.Label(self.page, text="", bg=self.colors["bg"], fg=self.colors["muted_light"], font=("Avenir Next", 8))
        self.footer.pack(side="bottom", fill="x", padx=20, pady=(4, 7))

        # Zone centrale défilable : l'en-tête, les KPI et les contrôles restent visibles.
        body_wrap = tk.Frame(self.page, bg=self.colors["bg"])
        body_wrap.pack(fill="both", expand=True, padx=(18, 10), pady=(0, 2))
        self.body_canvas = tk.Canvas(body_wrap, bg=self.colors["bg"], highlightthickness=0, borderwidth=0)
        self.body_scrollbar = tk.Scrollbar(body_wrap, orient="vertical", command=self.body_canvas.yview, width=9)
        self.body_canvas.configure(yscrollcommand=self.body_scrollbar.set)
        self.body_canvas.pack(side="left", fill="both", expand=True)
        self.body_scrollbar.pack(side="right", fill="y", padx=(5, 0))

        self.body = tk.Frame(self.body_canvas, bg=self.colors["bg"])
        self._body_window = self.body_canvas.create_window((0, 0), window=self.body, anchor="nw")
        for col in range(2):
            self.body.grid_columnconfigure(col, weight=1, uniform="main")

        self.body.bind("<Configure>", self._sync_scroll_region)
        self.body_canvas.bind("<Configure>", self._fit_body_width)
        self.bind_all("<MouseWheel>", self._on_mousewheel)
        self.bind_all("<Button-4>", self._on_mousewheel)
        self.bind_all("<Button-5>", self._on_mousewheel)

    def _sync_scroll_region(self, _event: Optional[tk.Event] = None) -> None:
        self.body_canvas.configure(scrollregion=self.body_canvas.bbox("all"))
        bbox = self.body_canvas.bbox("all")
        if bbox and self._scroll_target_y is not None:
            max_top = max(0.0, float(bbox[3] - bbox[1] - self.body_canvas.winfo_height()))
            self._scroll_target_y = min(max(self._scroll_target_y, 0.0), max_top)

    def _fit_body_width(self, event: tk.Event) -> None:
        self.body_canvas.itemconfigure(self._body_window, width=max(1, int(event.width)))

    def _on_mousewheel(self, event: tk.Event) -> str | None:
        if not hasattr(self, "body_canvas"):
            return None
        bbox = self.body_canvas.bbox("all")
        if not bbox:
            return None
        total_h = float(bbox[3] - bbox[1])
        view_h = float(self.body_canvas.winfo_height())
        max_top = max(0.0, total_h - view_h)
        if max_top <= 0:
            return None

        if getattr(event, "num", None) == 4:
            pixels = -86.0
        elif getattr(event, "num", None) == 5:
            pixels = 86.0
        else:
            delta = float(getattr(event, "delta", 0) or 0)
            if delta == 0:
                return None
            # Trackpad macOS : petits deltas continus. Souris classique : ±120.
            pixels = -(delta * 3.2 if abs(delta) < 120 else (delta / 120.0) * 96.0)

        current_top = float(self.body_canvas.yview()[0]) * total_h
        base = current_top if self._scroll_target_y is None else self._scroll_target_y
        self._scroll_target_y = min(max(base + pixels, 0.0), max_top)
        if self._scroll_job is None:
            self._animate_scroll()
        return "break"

    def _animate_scroll(self) -> None:
        bbox = self.body_canvas.bbox("all")
        if not bbox or self._scroll_target_y is None:
            self._scroll_job = None
            return
        total_h = float(bbox[3] - bbox[1])
        if total_h <= 0:
            self._scroll_job = None
            return
        current_top = float(self.body_canvas.yview()[0]) * total_h
        diff = self._scroll_target_y - current_top
        if abs(diff) < 0.8:
            self.body_canvas.yview_moveto(self._scroll_target_y / total_h)
            self._scroll_target_y = None
            self._scroll_job = None
            return
        next_top = current_top + diff * 0.24
        self.body_canvas.yview_moveto(next_top / total_h)
        self._scroll_job = self.after(16, self._animate_scroll)

    def _header(self) -> None:
        head = tk.Frame(self.page, bg=self.colors["bg"])
        head.pack(fill="x", padx=20, pady=(16, 10))
        left = tk.Frame(head, bg=self.colors["bg"])
        left.pack(side="left", fill="x", expand=True)
        tk.Label(left, text="MARKET INTELLIGENCE", bg=self.colors["bg"], fg=self.colors["blue"], font=("Avenir Next", 10, "bold")).pack(anchor="w")
        tk.Label(left, text="Tableau de marché", bg=self.colors["bg"], fg=self.colors["ink"], font=("Avenir Next", 27, "bold")).pack(anchor="w")
        tk.Label(left, text="Régime · volatilité · options · participation · stress macro", bg=self.colors["bg"], fg=self.colors["muted"], font=("Avenir Next", 11)).pack(anchor="w")

        right = tk.Frame(head, bg=self.colors["bg"])
        right.pack(side="right", anchor="ne", pady=(8, 0))
        self.spinner = tk.Canvas(right, width=26, height=26, bg=self.colors["bg"], highlightthickness=0)
        self.spinner.pack(side="left", padx=(0, 8), pady=(3, 0))
        tk.Label(right, textvariable=self.status, bg=self.colors["bg"], fg=self.colors["muted"], font=("Avenir Next", 10, "bold")).pack(side="left", padx=(0, 12), pady=(5, 0))
        tk.Button(right, text="Actualiser les données", command=lambda: self.refresh(force_sources=True), bg=self.colors["ink"], fg="white", activebackground=self.colors["ink_soft"], activeforeground="white", borderwidth=0, padx=18, pady=10, cursor="hand2", font=("Avenir Next", 10, "bold")).pack(side="left")

    def refresh(self, force_sources: bool = False) -> None:
        period_value = self.period.get()
        self.status.set("Mise à jour")
        self._start_spinner()
        self._clear(self.body)
        self._clear(self.kpi_grid)
        self._loading()
        threading.Thread(target=self._worker, args=(period_value, force_sources), daemon=True).start()
        self.after(100, self._poll)

    def _worker(self, period_value: str, force_sources: bool) -> None:
        try:
            self.queue.put(("ok", build_market_data(period_value, force_sources=force_sources)))
        except Exception as exc:
            self.queue.put(("error", exc))

    def _poll(self) -> None:
        try:
            state, payload = self.queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll)
            return
        self._stop_spinner()
        self._clear(self.body)
        self._clear(self.kpi_grid)
        if state == "error":
            self.status.set("Erreur")
            self._error(str(payload))
            return
        self.status.set("À jour")
        self.data = payload
        self._render()

    def _start_spinner(self) -> None:
        self.spinner_angle = 0
        def animate() -> None:
            self.spinner.delete("all")
            for i in range(8):
                color = self._blend(self.colors["line"], self.colors["blue"], i / 7)
                self.spinner.create_arc(4, 4, 20, 20, start=self.spinner_angle + i * 32, extent=18, style="arc", width=2, outline=color)
            self.spinner_angle = (self.spinner_angle + 20) % 360
            self.spinner_job = self.after(55, animate)
        animate()

    def _stop_spinner(self) -> None:
        if self.spinner_job is not None:
            self.after_cancel(self.spinner_job)
            self.spinner_job = None
        self.spinner.delete("all")

    def _blend(self, a: str, b: str, t: float) -> str:
        def rgb(value: str) -> tuple[int, int, int]:
            value = value.lstrip("#")
            return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))
        ar, ag, ab = rgb(a); br, bg, bb = rgb(b)
        return f"#{int(ar + (br-ar)*t):02x}{int(ag + (bg-ag)*t):02x}{int(ab + (bb-ab)*t):02x}"

    def _series_key(self, chart_key: str, label: str) -> str:
        return f"{chart_key}:{label}"

    def _series_is_visible(self, chart_key: str, label: str, default: bool = True) -> bool:
        return self.series_visibility.get(self._series_key(chart_key, label), default)

    def _toggle_series(self, chart_key: str, label: str, default: bool = True) -> None:
        key = self._series_key(chart_key, label)
        self.series_visibility[key] = not self.series_visibility.get(key, default)

    def _period_control(self, parent: tk.Widget) -> tk.Frame:
        outer = tk.Frame(parent, bg=self.colors["panel"], highlightbackground=self.colors["line_soft"], highlightthickness=1, height=94)
        outer.pack_propagate(False)

        left = tk.Frame(outer, bg=self.colors["panel"], width=220)
        left.pack(side="left", fill="y", padx=(18, 8), pady=14)
        left.pack_propagate(False)
        tk.Label(left, text="HORIZON D’ANALYSE", bg=self.colors["panel"], fg=self.colors["ink"], font=("Avenir Next", 11, "bold")).pack(anchor="w")
        tk.Label(left, text="Choisissez la profondeur historique", bg=self.colors["panel"], fg=self.colors["muted"], font=("Avenir Next", 9)).pack(anchor="w", pady=(4, 0))

        segment = tk.Frame(outer, bg=self.colors["panel_soft"], highlightbackground=self.colors["line_soft"], highlightthickness=1)
        segment.pack(side="left", fill="both", expand=True, padx=8, pady=15)
        self.period_buttons = {}
        for label in PERIODS:
            selected = self.period.get() == label
            button = tk.Button(
                segment, text=label, command=lambda value=label: self._set_period(value),
                bg=self.colors["ink"] if selected else self.colors["panel_soft"],
                fg="white" if selected else self.colors["ink_soft"],
                activebackground=self.colors["ink_soft"] if selected else self.colors["line_soft"],
                activeforeground="white" if selected else self.colors["ink"],
                borderwidth=0, padx=15, pady=10, cursor="hand2", font=("Avenir Next", 10, "bold")
            )
            button.pack(side="left", fill="both", expand=True, padx=2, pady=2)
            self.period_buttons[label] = button

        info = tk.Frame(outer, bg=self.colors["panel"], width=245)
        info.pack(side="right", fill="y", padx=(10, 18), pady=14)
        info.pack_propagate(False)
        tk.Label(info, text="PLAGE AFFICHÉE", bg=self.colors["panel"], fg=self.colors["muted_light"], font=("Avenir Next", 10, "bold")).pack(anchor="e")
        tk.Label(info, textvariable=self.range_text, bg=self.colors["panel"], fg=self.colors["ink"], font=("Avenir Next", 10, "bold")).pack(anchor="e", pady=(3, 0))
        tk.Label(info, textvariable=self.session_text, bg=self.colors["panel"], fg=self.colors["muted"], font=("Avenir Next", 8)).pack(anchor="e", pady=(2, 0))
        return outer

    def _set_period(self, value: str) -> None:
        if self.period.get() == value:
            return
        self.period.set(value)
        for label, button in getattr(self, "period_buttons", {}).items():
            selected = label == value
            button.configure(bg=self.colors["ink"] if selected else self.colors["panel_soft"], fg="white" if selected else self.colors["ink_soft"])
        self.refresh(force_sources=False)

    def _render(self) -> None:
        if not self.data:
            return
        for idx, (label, value) in enumerate(self.data["kpis"].items()):
            self._kpi_card(idx, label, value)

        # Tous les modules réguliers ont exactement la même hauteur ; le dernier seul est plus grand.
        for row in range(4):
            self.body.grid_rowconfigure(row, minsize=self.regular_panel_height + 12, weight=0)
        self.body.grid_rowconfigure(4, minsize=self.breadth_panel_height + 12, weight=0)

        date_source = self.data.get("price_chart", pd.DataFrame())
        if isinstance(date_source, pd.DataFrame) and not date_source.empty and "Date" in date_source:
            dates = pd.to_datetime(date_source["Date"], errors="coerce").dropna()
            if not dates.empty:
                self.range_text.set(f"{dates.iloc[0].strftime('%d.%m.%Y')}  →  {dates.iloc[-1].strftime('%d.%m.%Y')}")
                self.session_text.set(f"{len(dates):,} séances de marché".replace(",", " "))

        # Rangée 1 : synthèse du régime + carte de marché.
        self._summary_panel(0, 0, 1)
        map_canvas = self._chart_panel("Carte de marché", "Performance par bloc", 0, 1, 1)
        map_canvas.bind("<Configure>", lambda _e: self._draw_heatmap(map_canvas, self.data["all_perf"]))

        # Rangée 2 : confirmation cross-asset + volatilité actions.
        cross_canvas = self._chart_panel("Cross-asset", "Performance relative et corrélations au S&P 500", 1, 0, 1)
        cross_canvas.bind("<Configure>", lambda _e: self._draw_cross_asset_dashboard(cross_canvas, self.data["cross_perf"], self.data["corr"]))

        vix = self._chart_panel("VIX / S&P 500", "Volatilité actions", 1, 1, 1)
        vix.bind("<Configure>", lambda _e: self._draw_dual_axis_chart(vix, self.data["vix_chart"], [("VIX", self.colors["gold"])], "SPY", "S&P 500", zones=[(25, 35, self.colors["red_bg"]), (15, 18, self.colors["green_bg"])], hlines=[18, 25, 35], left_min=8, chart_key="vix"))

        # Rangée 3 : options + appétit cyclique/défensif.
        pc = self._chart_panel("SPX Put / Call OI", "Ratio, moyenne 20 jours et S&P 500", 2, 0, 1)
        pc.bind("<Configure>", lambda _e: self._draw_dual_axis_chart(pc, self.data["put_call_chart"], [("Ratio", self.colors["gold"]), ("MM20", self.colors["blue"])], "SPY", "S&P 500", hlines=[1.45, 1.80, 2.15], left_min=1.15, chart_key="put_call"))

        style = self._chart_panel("XLY / XLP", "Cycliques face aux défensifs", 2, 1, 1)
        style.bind("<Configure>", lambda _e: self._draw_dual_axis_chart(style, self.data["style_chart"], [("XLY/XLP", self.colors["gold"])], "SPY", "S&P 500", chart_key="style_ratio"))

        # Rangée 4 : stress obligataire et souverain.
        move = self._chart_panel("MOVE Index", "Volatilité obligataire", 3, 0, 1)
        move.bind("<Configure>", lambda _e: self._draw_dual_axis_chart(move, self.data["move_chart"], [("MOVE", self.colors["gold"]), ("MM20", self.colors["blue"])], "SPY", "S&P 500", hlines=[80, 100, 120], left_min=40, chart_key="move"))

        cds = self._chart_panel("CDS US 5 ans", "Stress de crédit souverain", 3, 1, 1)
        cds.bind("<Configure>", lambda _e: self._draw_dual_axis_chart(cds, self.data["cds_chart"], [("US_CDS_5Y", self.colors["gold"])], "SPY", "S&P 500", hlines=[20, 35, 50], left_min=0, chart_key="cds"))

        # Dernier module : volontairement plus grand et sur toute la largeur.
        breadth = self._chart_panel("Participation sectorielle", "Largeur du marché · MM50 · état des 11 secteurs", 4, 0, 2)
        breadth.bind("<Configure>", lambda _e: self._draw_breadth_chart(breadth, self.data["breadth"], self.data["sector_state"]))

        source_state = self.data.get("source_state", {})
        updated = self.data.get("updated_at")
        stamp = pd.to_datetime(updated).strftime("%Y-%m-%d %H:%M") if updated is not None else ""
        self.footer.configure(text=f"Mis à jour {stamp}  ·  Yahoo Finance : {source_state.get('prices','?')}  ·  Options SPX : {source_state.get('put_call','?')}  ·  CDS US : {source_state.get('cds','?')}  ·  Cache : {CACHE_DIR}")
        self.after_idle(self._sync_scroll_region)

    def _kpi_card(self, index: int, label: str, value: str) -> None:
        card = tk.Frame(self.kpi_grid, bg=self.colors["panel"], highlightbackground=self.colors["line_soft"], highlightthickness=1)
        card.grid(row=0, column=index, sticky="nsew", padx=3, pady=0)
        accent = self.colors["red"] if label == "VIX" else self.colors["gold"] if label == "SPX OI P/C" else self.colors["green"] if label in {"S&P 500", "Nasdaq", "Participation"} else self.colors["blue"]
        tk.Frame(card, bg=accent, width=3).pack(side="left", fill="y")
        inner = tk.Frame(card, bg=self.colors["panel"])
        inner.pack(fill="both", expand=True, padx=9, pady=7)
        tk.Label(inner, text=label.upper(), bg=self.colors["panel"], fg=self.colors["muted"], font=("Avenir Next", 10, "bold")).pack(anchor="w")
        tk.Label(inner, text=value, bg=self.colors["panel"], fg=self.colors["ink"], font=("Avenir Next", 17, "bold")).pack(anchor="w", pady=(2, 0))

    def _panel_frame(self, title: str, subtitle: str, row: int, col: int, colspan: int, rowspan: int = 1) -> tk.Frame:
        fixed_h = self.breadth_panel_height if row == 4 else self.regular_panel_height
        panel = tk.Frame(self.body, bg=self.colors["panel"], highlightbackground=self.colors["line_soft"], highlightthickness=1, height=fixed_h)
        panel.grid(row=row, column=col, columnspan=colspan, rowspan=rowspan, sticky="nsew", padx=6, pady=6)
        panel.grid_propagate(False)
        panel.pack_propagate(False)
        head = tk.Frame(panel, bg=self.colors["panel"], height=46)
        head.pack(fill="x", padx=14, pady=(9, 4))
        head.pack_propagate(False)
        tk.Label(head, text=title, bg=self.colors["panel"], fg=self.colors["ink"], font=("Avenir Next", 13, "bold")).pack(side="left", anchor="w")
        tk.Label(head, text=subtitle, bg=self.colors["panel"], fg=self.colors["muted"], font=("Avenir Next", 9)).pack(side="right", anchor="e")
        return panel

    def _chart_panel(self, title: str, subtitle: str, row: int, col: int, colspan: int, rowspan: int = 1) -> tk.Canvas:
        panel = self._panel_frame(title, subtitle, row, col, colspan, rowspan)
        canvas = tk.Canvas(panel, bg=self.colors["panel_alt"], highlightthickness=0)
        canvas.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        return canvas

    def _summary_panel(self, row: int, col: int, colspan: int) -> None:
        panel = self._panel_frame("Lecture de marché", "Synthèse multi-signal", row, col, colspan)
        summary = self.data["summary"]
        stats = self.data["put_call"]
        tone = self.colors.get(summary["tone"], self.colors["blue"])

        price = self.data.get("price_chart", pd.DataFrame()).copy()
        spy = mm20 = mm50 = float("nan")
        if not price.empty:
            for key in ["SPY", "MM20", "MM50"]:
                if key in price:
                    clean = pd.to_numeric(price[key], errors="coerce").dropna()
                    if not clean.empty:
                        if key == "SPY": spy = float(clean.iloc[-1])
                        elif key == "MM20": mm20 = float(clean.iloc[-1])
                        else: mm50 = float(clean.iloc[-1])

        breadth_df = self.data.get("breadth", pd.DataFrame())
        participation = last(breadth_df.get("Breadth")) if isinstance(breadth_df, pd.DataFrame) and not breadth_df.empty else float("nan")
        vix_df = self.data.get("vix_chart", pd.DataFrame())
        vix = last(vix_df.get("VIX")) if isinstance(vix_df, pd.DataFrame) and not vix_df.empty else float("nan")

        if not any(pd.isna(v) for v in [spy, mm20, mm50]) and spy > mm20 > mm50:
            trend_label, trend_detail, trend_tone = "Haussière", "SPY > MM20 > MM50", self.colors["green"]
        elif not any(pd.isna(v) for v in [spy, mm50]) and spy > mm50:
            trend_label, trend_detail, trend_tone = "Positive", "SPY au-dessus de MM50", self.colors["green"]
        elif not any(pd.isna(v) for v in [spy, mm20, mm50]) and spy < mm20 < mm50:
            trend_label, trend_detail, trend_tone = "Baissière", "SPY < MM20 < MM50", self.colors["red"]
        else:
            trend_label, trend_detail, trend_tone = "Mixte", "Moyennes non alignées", self.colors["gold"]

        if pd.isna(participation):
            breadth_label, breadth_tone = "n.d.", self.colors["muted"]
        elif participation >= 65:
            breadth_label, breadth_tone = "Large", self.colors["green"]
        elif participation <= 40:
            breadth_label, breadth_tone = "Fragile", self.colors["red"]
        else:
            breadth_label, breadth_tone = "Intermédiaire", self.colors["gold"]

        if pd.isna(vix):
            vol_label, vol_tone = "n.d.", self.colors["muted"]
        elif vix < 18:
            vol_label, vol_tone = "Contenue", self.colors["green"]
        elif vix >= 25:
            vol_label, vol_tone = "Élevée", self.colors["red"]
        else:
            vol_label, vol_tone = "Normale", self.colors["gold"]

        hero_notes = {
            "Risque élevé": "Volatilité élevée : privilégier les confirmations par la participation.",
            "Risk-on discipliné": "Performance, volatilité et participation restent constructives.",
            "Marché sélectif": "La hausse manque de largeur : les leaders restent déterminants.",
            "Rotation défensive": "Les secteurs défensifs reprennent du poids face aux cycliques.",
            "Équilibre à confirmer": "Signaux mixtes : attendre confirmation par volatilité et participation.",
        }
        hero_note = hero_notes.get(summary["regime"], summary["note"])
        options_regime = str(stats.get("regime", "n.d."))
        options_short = {
            "Protection élevée": "Protection ↑",
            "Complacence relative": "Complacence",
            "Équilibre SPX": "Équilibre",
            "Signal indisponible": "n.d.",
        }.get(options_regime, options_regime)

        hero = tk.Frame(panel, bg=self._blend(self.colors["panel"], self.colors.get(f"{summary['tone']}_bg", self.colors["blue_bg"]), 0.55), highlightbackground=self.colors["line_soft"], highlightthickness=1)
        hero.pack(fill="x", padx=12, pady=(4, 8))
        tk.Frame(hero, bg=tone, width=5).pack(side="left", fill="y")
        hero_inner = tk.Frame(hero, bg=hero.cget("bg"))
        hero_inner.pack(fill="both", expand=True, padx=13, pady=10)
        badge = tk.Label(hero_inner, text="RÉGIME ACTUEL", bg=tone, fg="white", font=("Avenir Next", 8, "bold"), padx=9, pady=4)
        badge.pack(anchor="w")
        tk.Label(hero_inner, text=summary["regime"], bg=hero_inner.cget("bg"), fg=self.colors["ink"], font=("Avenir Next", 20, "bold")).pack(anchor="w", pady=(6, 1))
        tk.Label(hero_inner, text=hero_note, bg=hero_inner.cget("bg"), fg=self.colors["muted"], font=("Avenir Next", 9), justify="left", wraplength=500).pack(anchor="w")

        signals = tk.Frame(panel, bg=self.colors["panel"])
        signals.pack(fill="x", padx=10, pady=(0, 7))
        for c in range(4):
            signals.grid_columnconfigure(c, weight=1, uniform="signals")

        signal_specs = [
            ("TENDANCE", trend_label, trend_detail, trend_tone),
            ("PARTICIPATION", breadth_label, fmt_pct(participation, 0, signed=False), breadth_tone),
            ("VOLATILITÉ", vol_label, f"VIX {fmt_num(vix, 1)}", vol_tone),
            ("OPTIONS", options_short, f"P/C {fmt_num(stats.get('ratio'), 2)} · P{fmt_num(stats.get('percentile'), 0)}", self.colors.get(stats.get("tone", "blue"), self.colors["blue"])),
        ]
        for idx, (label, value, detail, accent) in enumerate(signal_specs):
            card = tk.Frame(signals, bg=self.colors["panel_alt"], highlightbackground=self.colors["line_soft"], highlightthickness=1)
            card.grid(row=0, column=idx, sticky="nsew", padx=3)
            tk.Frame(card, bg=accent, height=3).pack(fill="x")
            tk.Label(card, text=label, bg=self.colors["panel_alt"], fg=self.colors["muted_light"], font=("Avenir Next", 10, "bold")).pack(anchor="w", padx=8, pady=(6, 1))
            tk.Label(card, text=value, bg=self.colors["panel_alt"], fg=accent, font=("Avenir Next", 10, "bold"), wraplength=118, justify="left").pack(anchor="w", padx=8)
            tk.Label(card, text=detail, bg=self.colors["panel_alt"], fg=self.colors["muted"], font=("Avenir Next", 8), wraplength=118, justify="left").pack(anchor="w", padx=8, pady=(1, 6))

        lower = tk.Frame(panel, bg=self.colors["panel"])
        lower.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        lower.grid_columnconfigure(0, weight=5)
        lower.grid_columnconfigure(1, weight=3)
        lower.grid_rowconfigure(0, weight=1)

        trend_box = tk.Frame(lower, bg=self.colors["panel_alt"], highlightbackground=self.colors["line_soft"], highlightthickness=1)
        trend_box.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        tk.Label(trend_box, text="TENDANCE S&P 500", bg=self.colors["panel_alt"], fg=self.colors["muted"], font=("Avenir Next", 10, "bold")).pack(anchor="w", padx=9, pady=(6, 0))
        spy_canvas = tk.Canvas(trend_box, bg=self.colors["panel_alt"], height=110, highlightthickness=0)
        spy_canvas.pack(fill="both", expand=True, padx=3, pady=(0, 3))
        spy_canvas.bind("<Configure>", lambda _e: self._draw_line_chart(spy_canvas, self.data["price_chart"], [("SPY", self.colors["blue"]), ("MM20", self.colors["gold"]), ("MM50", self.colors["green"])], False))

        leadership_box = tk.Frame(lower, bg=self.colors["panel_alt"], highlightbackground=self.colors["line_soft"], highlightthickness=1)
        leadership_box.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        tk.Label(leadership_box, text="LEADERSHIP SECTORIEL", bg=self.colors["panel_alt"], fg=self.colors["muted"], font=("Avenir Next", 10, "bold")).pack(anchor="w", padx=10, pady=(8, 5))
        tk.Label(leadership_box, text="Leader", bg=self.colors["panel_alt"], fg=self.colors["muted_light"], font=("Avenir Next", 10, "bold")).pack(anchor="w", padx=10)
        best_text = str(summary["best_sector"]).replace("Consumer Discretionary", "Cons. Discretionary").replace("Consumer Staples", "Cons. Staples").replace("Communication", "Comm.")
        tk.Label(leadership_box, text=best_text, bg=self.colors["panel_alt"], fg=self.colors["green"], font=("Avenir Next", 10, "bold"), wraplength=155, justify="left").pack(anchor="w", padx=10, pady=(1, 5))
        tk.Label(leadership_box, text="Sous pression", bg=self.colors["panel_alt"], fg=self.colors["muted_light"], font=("Avenir Next", 10, "bold")).pack(anchor="w", padx=10)
        worst_text = str(summary["worst_sector"]).replace("Consumer Discretionary", "Cons. Discretionary").replace("Consumer Staples", "Cons. Staples").replace("Communication", "Comm.")
        tk.Label(leadership_box, text=worst_text, bg=self.colors["panel_alt"], fg=self.colors["red"], font=("Avenir Next", 10, "bold"), wraplength=155, justify="left").pack(anchor="w", padx=10, pady=(1, 7))
        leadership = summary.get("leadership", float("nan"))
        leadership_tone = self.colors["green"] if not pd.isna(leadership) and leadership >= 0 else self.colors["red"]
        tk.Label(leadership_box, text=f"Cyc. – déf.  {fmt_pct(leadership)}", bg=self.colors["panel_alt"], fg=leadership_tone, font=("Avenir Next", 9, "bold")).pack(anchor="w", padx=10, pady=(0, 8))

    def _draw_dual_axis_chart(
        self,
        canvas: tk.Canvas,
        df: pd.DataFrame,
        left_series: list[tuple[str, str]],
        right_col: str,
        right_label: str,
        zones: Optional[list[tuple[float, float, str]]] = None,
        hlines: Optional[list[float]] = None,
        left_min: Optional[float] = None,
        left_max: Optional[float] = None,
        chart_key: str = "chart",
    ) -> None:
        canvas.delete("all")
        width = max(canvas.winfo_width(), 340)
        height = max(canvas.winfo_height(), 155)
        pad_left, pad_right, pad_top, pad_bottom = (46, 54, 26, 28) if width < 560 else (54, 64, 28, 32)
        if df.empty or "Date" not in df:
            return

        plot_df = df.copy()
        plot_df["Date"] = pd.to_datetime(plot_df["Date"], errors="coerce")
        plot_df = plot_df.dropna(subset=["Date"])
        if plot_df.empty:
            return

        visible_left = [(label, color) for label, color in left_series if self._series_is_visible(chart_key, label)]
        right_visible = self._series_is_visible(chart_key, right_label)

        left_values: list[float] = []
        for col, _color in visible_left:
            if col in plot_df:
                left_values.extend(pd.to_numeric(plot_df[col], errors="coerce").dropna().tolist())
        if not left_values:
            for col, _color in left_series:
                if col in plot_df:
                    left_values.extend(pd.to_numeric(plot_df[col], errors="coerce").dropna().tolist())
        if not left_values:
            left_values = [0.0, 1.0]
        left_low = min(left_values) if left_min is None else left_min
        left_high = max(left_values) if left_max is None else left_max
        if zones:
            for z0, z1, _color in zones:
                left_low = min(left_low, z0, z1)
                left_high = max(left_high, z0, z1)
        if hlines:
            left_low = min(left_low, *hlines)
            left_high = max(left_high, *hlines)
        if math.isclose(left_low, left_high):
            left_low -= 1
            left_high += 1
        pad = (left_high - left_low) * 0.08
        left_low -= 0 if left_min is not None else pad
        left_high += 0 if left_max is not None else pad

        right_values = pd.to_numeric(plot_df.get(right_col, pd.Series(dtype=float)), errors="coerce").dropna()
        has_right = not right_values.empty and right_visible
        if has_right:
            right_low = float(right_values.min())
            right_high = float(right_values.max())
            if math.isclose(right_low, right_high):
                right_low -= 1
                right_high += 1
            right_pad = (right_high - right_low) * 0.08
            right_low -= right_pad
            right_high += right_pad
        else:
            right_low, right_high = 0.0, 1.0

        plot_w = width - pad_left - pad_right
        plot_h = height - pad_top - pad_bottom

        def x_for(idx: int, count: int) -> float:
            return pad_left + idx * plot_w / max(count - 1, 1)

        def y_left(value: float) -> float:
            return height - pad_bottom - ((value - left_low) / (left_high - left_low)) * plot_h

        def y_right(value: float) -> float:
            return height - pad_bottom - ((value - right_low) / (right_high - right_low)) * plot_h

        canvas.create_rectangle(pad_left, pad_top, width - pad_right, height - pad_bottom, fill=self.colors["panel_alt"], outline=self.colors["line_soft"])
        if zones:
            for z0, z1, color in zones:
                y0, y1 = y_left(z0), y_left(z1)
                canvas.create_rectangle(pad_left, min(y0, y1), width - pad_right, max(y0, y1), fill=color, outline="")
        for i in range(5):
            y = pad_top + i * plot_h / 4
            canvas.create_line(pad_left, y, width - pad_right, y, fill=self.colors["grid"], width=1)

        dates = plot_df["Date"].dropna().reset_index(drop=True)
        if not dates.empty:
            date_values = plot_df["Date"].reset_index(drop=True)
            year_positions: list[tuple[int, int]] = []
            first_year = int(dates.iloc[0].year)
            last_year = int(dates.iloc[-1].year)
            for year in range(first_year, last_year + 1):
                candidates = date_values[date_values >= pd.Timestamp(year=year, month=1, day=1)]
                if candidates.empty:
                    continue
                idx = int(candidates.index[0])
                if idx < len(plot_df):
                    year_positions.append((idx, year))
            if first_year == last_year:
                year_positions = [(0, first_year)]
            elif len(year_positions) <= 1:
                year_positions = [(0, first_year), (len(plot_df) - 1, last_year)]
            last_x = -999.0
            for idx, year in year_positions:
                x = x_for(idx, len(plot_df))
                if x - last_x < 54 and idx != len(plot_df) - 1:
                    continue
                canvas.create_line(x, pad_top, x, height - pad_bottom, fill=self.colors["grid_soft"], width=1)
                canvas.create_text(x, height - 10, text=str(year), anchor="center", fill=self.colors["muted"], font=("Avenir Next", 10, "bold"))
                last_x = x

        if hlines:
            for line in hlines:
                y = y_left(line)
                canvas.create_line(pad_left, y, width - pad_right, y, fill=self.colors["line"], width=1, dash=(5, 5))
                axis_digits = 2 if left_high - left_low <= 5 else 1
                canvas.create_text(pad_left + 6, y - 8, text=fmt_num(line, axis_digits), anchor="w", fill=self.colors["muted"], font=("Avenir Next", 10, "bold"))

        axis_digits = 2 if left_high - left_low <= 5 else 1
        for value in np.linspace(left_low, left_high, 5):
            y = y_left(float(value))
            canvas.create_text(pad_left - 12, y, text=fmt_num(value, axis_digits), anchor="e", fill=self.colors["muted_light"], font=("Avenir Next", 10, "bold"))
        if has_right:
            for value in np.linspace(right_low, right_high, 5):
                y = y_right(float(value))
                canvas.create_text(width - pad_right + 12, y, text=fmt_num(value, 0), anchor="w", fill=self.colors["green"], font=("Avenir Next", 10, "bold"))

        legend_x = pad_left
        for label, color in left_series:
            if label not in plot_df:
                continue
            visible = self._series_is_visible(chart_key, label)
            tag = f"legend_{chart_key}_{label}".replace(" ", "_").replace("/", "_")
            active_color = color if visible else self.colors["muted_light"]
            text_color = self.colors["muted"] if visible else self.colors["muted_light"]
            canvas.create_rectangle(legend_x - 5, 3, legend_x + 78, 25, fill=self.colors["panel_alt"], outline="", tags=(tag,))
            canvas.create_line(legend_x, 13, legend_x + 18, 13, fill=active_color, width=3, dash=() if visible else (3, 3), tags=(tag,))
            canvas.create_text(legend_x + 24, 13, text=label, anchor="w", fill=text_color, font=("Avenir Next", 9, "bold"), tags=(tag,))
            canvas.tag_bind(tag, "<Button-1>", lambda _e, series_label=label: self._redraw_dual_after_toggle(canvas, chart_key, series_label, df, left_series, right_col, right_label, zones, hlines, left_min, left_max))
            legend_x += 92
        right_tag = f"legend_{chart_key}_{right_label}".replace(" ", "_").replace("/", "_")
        right_text_color = self.colors["muted"] if right_visible else self.colors["muted_light"]
        right_line_color = self.colors["green"] if right_visible else self.colors["muted_light"]
        canvas.create_rectangle(legend_x - 5, 3, legend_x + 112, 25, fill=self.colors["panel_alt"], outline="", tags=(right_tag,))
        canvas.create_line(legend_x, 13, legend_x + 18, 13, fill=right_line_color, width=3, dash=() if right_visible else (3, 3), tags=(right_tag,))
        canvas.create_text(legend_x + 24, 13, text=right_label, anchor="w", fill=right_text_color, font=("Avenir Next", 9, "bold"), tags=(right_tag,))
        canvas.tag_bind(right_tag, "<Button-1>", lambda _e: self._redraw_dual_after_toggle(canvas, chart_key, right_label, df, left_series, right_col, right_label, zones, hlines, left_min, left_max))

        n = len(plot_df)
        for col, color in visible_left:
            if col not in plot_df:
                continue
            clean = pd.to_numeric(plot_df[col], errors="coerce")
            points: list[float] = []
            for idx, value in enumerate(clean):
                if pd.isna(value):
                    continue
                points.extend([x_for(idx, n), y_left(float(value))])
            if len(points) >= 4:
                canvas.create_line(points, fill=color, width=2 if "MM" in col else 3, smooth=False)
            latest = clean.dropna()
            if not latest.empty and len(points) >= 2:
                canvas.create_oval(points[-2] - 3, points[-1] - 3, points[-2] + 3, points[-1] + 3, fill=color, outline=color)
                canvas.create_text(points[-2] + 8, points[-1], text=fmt_num(float(latest.iloc[-1]), 2), anchor="w", fill=color, font=("Avenir Next", 10, "bold"))

        if has_right:
            clean = pd.to_numeric(plot_df[right_col], errors="coerce")
            points = []
            for idx, value in enumerate(clean):
                if pd.isna(value):
                    continue
                points.extend([x_for(idx, n), y_right(float(value))])
            if len(points) >= 4:
                canvas.create_line(points, fill=self.colors["green"], width=2, smooth=False)
            latest = clean.dropna()
            if not latest.empty and len(points) >= 2:
                canvas.create_oval(points[-2] - 3, points[-1] - 3, points[-2] + 3, points[-1] + 3, fill=self.colors["green"], outline=self.colors["green"])
                canvas.create_text(width - pad_right + 8, points[-1], text=fmt_num(float(latest.iloc[-1]), 0), anchor="w", fill=self.colors["green"], font=("Avenir Next", 10, "bold"))

        def show_crosshair(event: tk.Event) -> None:
            canvas.delete("crosshair")
            if event.x < pad_left or event.x > width - pad_right or event.y < pad_top or event.y > height - pad_bottom:
                return
            idx = int(round((event.x - pad_left) / max(plot_w, 1) * max(len(plot_df) - 1, 1)))
            idx = max(0, min(len(plot_df) - 1, idx))
            row = plot_df.iloc[idx]
            cross_color = "#8c867c"
            canvas.create_line(event.x, pad_top, event.x, height - pad_bottom, fill=cross_color, width=1, tags="crosshair")
            canvas.create_line(pad_left, event.y, width - pad_right, event.y, fill=cross_color, width=1, tags="crosshair")
            items = [pd.to_datetime(row["Date"]).strftime("%Y-%m-%d")]
            for label, color in visible_left:
                if label in row and not pd.isna(row[label]):
                    items.append(f"{label} {fmt_num(row[label], 2 if label in {'Ratio', 'MM20'} else 1)}")
            if has_right and right_col in row and not pd.isna(row[right_col]):
                items.append(f"{right_label} {fmt_num(row[right_col], 0)}")
            text = "   ".join(items)
            tw = min(width - 2 * pad_left, max(210, len(text) * 6.3))
            tx = min(max(pad_left + 8, event.x + 12), width - pad_right - tw - 8)
            ty = pad_top + 10
            canvas.create_rectangle(tx, ty, tx + tw, ty + 30, fill=self.colors["panel"], outline=self.colors["line"], tags="crosshair")
            canvas.create_text(tx + 10, ty + 14, text=text, anchor="w", fill=self.colors["ink"], font=("Avenir Next", 9, "bold"), tags="crosshair")

        canvas.bind("<Motion>", show_crosshair)
        canvas.bind("<Leave>", lambda _e: canvas.delete("crosshair"))

    def _redraw_dual_after_toggle(
        self,
        canvas: tk.Canvas,
        chart_key: str,
        label: str,
        df: pd.DataFrame,
        left_series: list[tuple[str, str]],
        right_col: str,
        right_label: str,
        zones: Optional[list[tuple[float, float, str]]],
        hlines: Optional[list[float]],
        left_min: Optional[float],
        left_max: Optional[float],
    ) -> None:
        self._toggle_series(chart_key, label)
        self._draw_dual_axis_chart(canvas, df, left_series, right_col, right_label, zones, hlines, left_min, left_max, chart_key=chart_key)

    def _draw_breadth_chart(self, canvas: tk.Canvas, df: pd.DataFrame, states: pd.DataFrame) -> None:
        canvas.delete("all")
        width = max(canvas.winfo_width(), 760)
        height = max(canvas.winfo_height(), 560)
        canvas.create_rectangle(0, 0, width, height, fill=self.colors["panel_alt"], outline="")
        if df.empty or "Date" not in df or "Breadth" not in df:
            return

        plot_df = df.copy()
        plot_df["Date"] = pd.to_datetime(plot_df["Date"], errors="coerce")
        for col in ["Breadth", "RawAbove", "AvgDistance", "Benchmark"]:
            if col in plot_df:
                plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
        plot_df = plot_df.dropna(subset=["Date", "Breadth"])
        plot_df["Breadth"] = plot_df["Breadth"].clip(0, 100)
        if "RawAbove" in plot_df:
            plot_df["RawAbove"] = plot_df["RawAbove"].clip(0, 100)
        if plot_df.empty:
            return

        n = len(plot_df)
        dense_mode = n > 800
        very_dense_mode = n > 1700
        latest = float(plot_df["Breadth"].iloc[-1])
        latest_raw = last(plot_df.get("RawAbove"))
        avg_distance = last(plot_df.get("AvgDistance"))
        change20 = float(latest - plot_df["Breadth"].iloc[-21]) if n >= 21 else float("nan")
        above_count = int(pd.to_numeric(states.get("Above"), errors="coerce").fillna(0).sum()) if states is not None and not states.empty else 0
        total_count = int(len(states)) if states is not None and not states.empty else 0

        if latest >= 70:
            state_label, state_tone = "Participation large", self.colors["green"]
        elif latest <= 35:
            state_label, state_tone = "Participation fragile", self.colors["red"]
        else:
            state_label, state_tone = "Participation intermédiaire", self.colors["gold"]
        direction = "élargissement" if not pd.isna(change20) and change20 > 2 else "rétrécissement" if not pd.isna(change20) and change20 < -2 else "stable" if not pd.isna(change20) else "tendance n.d."

        canvas.create_text(20, 14, text=state_label, anchor="nw", fill=state_tone, font=("Avenir Next", 18, "bold"))
        canvas.create_text(20, 43, text=f"{direction} sur 20 séances", anchor="nw", fill=self.colors["muted"], font=("Avenir Next", 10, "bold"))

        metric_specs = [
            ("SCORE MM50", fmt_pct(latest, 0, signed=False), state_tone),
            ("> MM50", f"{above_count}/{total_count}", self.colors["green"] if total_count and above_count / total_count >= 0.55 else self.colors["red"]),
            ("BRUT", fmt_pct(latest_raw, 0, signed=False), self.colors["blue"]),
            ("DIST. MOY.", fmt_pct(avg_distance, 1), self.colors["green"] if not pd.isna(avg_distance) and avg_distance >= 0 else self.colors["red"]),
            ("Δ 20 J", fmt_pct(change20, 1), self.colors["green"] if not pd.isna(change20) and change20 >= 0 else self.colors["red"]),
        ]
        metrics_x0 = max(330, width * 0.36)
        metrics_w = width - metrics_x0 - 20
        gap = 8
        cell_w = (metrics_w - gap * 4) / 5
        for idx, (label, value, accent) in enumerate(metric_specs):
            x = metrics_x0 + idx * (cell_w + gap)
            canvas.create_rectangle(x, 10, x + cell_w, 66, fill=self.colors["panel"], outline=self.colors["line_soft"])
            canvas.create_rectangle(x, 10, x + cell_w, 14, fill=accent, outline=accent)
            canvas.create_text(x + 10, 25, text=label, anchor="nw", fill=self.colors["muted"], font=("Avenir Next", 8, "bold"))
            canvas.create_text(x + 10, 43, text=value, anchor="nw", fill=accent, font=("Avenir Next", 13, "bold"))

        pad_left, pad_right = 64, 34
        chart_top = 112
        sector_area_h = 170
        chart_bottom = height - sector_area_h - 48
        chart_h = max(210, chart_bottom - chart_top)
        plot_w = width - pad_left - pad_right

        def y_for(value: float) -> float:
            return chart_bottom - (min(100.0, max(0.0, value)) / 100.0) * chart_h

        canvas.create_rectangle(pad_left, chart_top, width - pad_right, chart_bottom, fill=self.colors["panel"], outline=self.colors["line_soft"])
        canvas.create_rectangle(pad_left, y_for(100), width - pad_right, y_for(75), fill=self._blend(self.colors["panel"], self.colors["green_bg"], 0.55), outline="")
        canvas.create_rectangle(pad_left, y_for(25), width - pad_right, y_for(0), fill=self._blend(self.colors["panel"], self.colors["red_bg"], 0.55), outline="")
        for value in [0, 25, 50, 75, 100]:
            y = y_for(value)
            strong = value in {25, 50, 75}
            canvas.create_line(pad_left, y, width - pad_right, y, fill=self.colors["line"] if strong else self.colors["grid"], width=1, dash=(5, 5) if strong else ())
            canvas.create_text(pad_left - 12, y, text=f"{value}%", anchor="e", fill=self.colors["muted"], font=("Avenir Next", 9, "bold"))

        dates = plot_df["Date"].reset_index(drop=True)
        if n > 1:
            if dates.iloc[0].year == dates.iloc[-1].year:
                mark_count = 5
                mark_idx = np.linspace(0, n - 1, mark_count, dtype=int)
                marks = [(int(i), dates.iloc[int(i)].strftime("%b")) for i in mark_idx]
            else:
                mark_count = 6 if very_dense_mode else 7 if dense_mode else min(8, max(4, int(plot_w // 160)))
                mark_idx = np.unique(np.linspace(0, n - 1, mark_count, dtype=int))
                marks = [(int(i), dates.iloc[int(i)].strftime("%Y")) for i in mark_idx]
                # Évite deux libellés identiques sur les horizons longs.
                dedup = []
                seen = set()
                for item in marks:
                    if item[1] not in seen or item[0] == n - 1:
                        dedup.append(item); seen.add(item[1])
                marks = dedup
            for idx, label in marks:
                x = pad_left + idx * plot_w / max(n - 1, 1)
                canvas.create_line(x, chart_top, x, chart_bottom, fill=self.colors["grid_soft"], width=1)
                canvas.create_text(x, chart_bottom + 18, text=str(label), anchor="center", fill=self.colors["muted"], font=("Avenir Next", 9, "bold"))

        series_specs = [
            ("Breadth", "Score MM50", self.colors["blue"], 3, (), True),
            ("RawAbove", "% secteurs > MM50", self.colors["gold"], 2, (5, 4), not dense_mode),
            ("Benchmark", "S&P 500 normalisé", self.colors["green"], 2, (3, 5), not very_dense_mode),
        ]
        legend_x = pad_left
        for col, label, color, _line_w, dash, default_visible in series_specs:
            if col not in plot_df:
                continue
            visible = self._series_is_visible("breadth", label, default_visible)
            tag = f"breadth_legend_{col}"
            active = color if visible else self.colors["muted_light"]
            legend_w = 150 if label != "% secteurs > MM50" else 165
            canvas.create_rectangle(legend_x - 5, 76, legend_x + legend_w, 103, fill=self.colors["panel_alt"], outline="", tags=(tag,))
            canvas.create_line(legend_x, 89, legend_x + 22, 89, fill=active, width=3 if visible else 2, dash=dash if visible else (3, 3), tags=(tag,))
            canvas.create_text(legend_x + 29, 89, text=label, anchor="w", fill=self.colors["muted"] if visible else self.colors["muted_light"], font=("Avenir Next", 9, "bold"), tags=(tag,))
            canvas.tag_bind(tag, "<Button-1>", lambda _e, series_label=label, d=default_visible: self._redraw_breadth_after_toggle(canvas, series_label, df, states, d))
            legend_x += legend_w + 18

        # Sur 5–10 ans, on réduit le nombre de points réellement dessinés à la résolution utile du canevas.
        max_render_points = max(360, int(plot_w * 0.72))
        render_step = max(1, math.ceil(n / max_render_points))
        render_indices = list(range(0, n, render_step))
        if render_indices[-1] != n - 1:
            render_indices.append(n - 1)

        for col, label, color, line_w, dash, default_visible in series_specs:
            if col not in plot_df or not self._series_is_visible("breadth", label, default_visible):
                continue
            clean = pd.to_numeric(plot_df[col], errors="coerce")
            points: list[float] = []
            for idx in render_indices:
                value = clean.iloc[idx]
                if pd.isna(value):
                    continue
                x = pad_left + idx * plot_w / max(n - 1, 1)
                points.extend([x, y_for(float(value))])
            if len(points) >= 4:
                canvas.create_line(points, fill=color, width=line_w, smooth=False, dash=dash)
            if len(points) >= 2:
                canvas.create_oval(points[-2] - 4, points[-1] - 4, points[-2] + 4, points[-1] + 4, fill=color, outline=self.colors["panel"], width=1)

        section_top = chart_bottom + 45
        canvas.create_text(pad_left, section_top, text="ÉTAT ACTUEL DES SECTEURS", anchor="nw", fill=self.colors["ink"], font=("Avenir Next", 10, "bold"))
        canvas.create_text(width - pad_right, section_top, text="écart à la MM50", anchor="ne", fill=self.colors["muted"], font=("Avenir Next", 9))
        if states is not None and not states.empty:
            state_rows = states.copy()
            state_rows["Distance"] = pd.to_numeric(state_rows["Distance"], errors="coerce")
            state_rows = state_rows.dropna(subset=["Distance"]).sort_values("Distance", ascending=False).reset_index(drop=True)
            cols, rows = 6, 2
            card_gap_x, card_gap_y = 8, 8
            cards_top = section_top + 27
            card_w = (plot_w - card_gap_x * (cols - 1)) / cols
            card_h = 52
            max_abs_dist = max(1.0, float(state_rows["Distance"].abs().max()))
            for idx, state in state_rows.head(cols * rows).iterrows():
                r, c = divmod(idx, cols)
                x = pad_left + c * (card_w + card_gap_x)
                y = cards_top + r * (card_h + card_gap_y)
                dist = float(state["Distance"])
                above = bool(state.get("Above"))
                accent = self.colors["green"] if above else self.colors["red"]
                fill = self._blend(self.colors["panel"], self.colors["green_bg"] if above else self.colors["red_bg"], 0.34)
                canvas.create_rectangle(x, y, x + card_w, y + card_h, fill=fill, outline=self.colors["line_soft"])
                canvas.create_rectangle(x, y, x + 4, y + card_h, fill=accent, outline=accent)
                canvas.create_text(x + 10, y + 10, text=str(state.get("Ticker", "")), anchor="nw", fill=self.colors["ink"], font=("Avenir Next", 10, "bold"))
                name = str(state.get("Nom", ""))
                if len(name) > 15:
                    name = name[:14] + "…"
                canvas.create_text(x + 10, y + 30, text=name, anchor="nw", fill=self.colors["muted"], font=("Avenir Next", 8))
                canvas.create_text(x + card_w - 9, y + 10, text=fmt_pct(dist, 1), anchor="ne", fill=accent, font=("Avenir Next", 10, "bold"))
                bar_x0 = x + card_w * 0.58
                bar_x1 = x + card_w - 9
                bar_y = y + 35
                canvas.create_line(bar_x0, bar_y, bar_x1, bar_y, fill=self.colors["line"], width=3)
                bar_len = (bar_x1 - bar_x0) * min(abs(dist) / max_abs_dist, 1)
                canvas.create_line(bar_x0, bar_y, bar_x0 + bar_len, bar_y, fill=accent, width=3)

        def show_crosshair(event: tk.Event) -> None:
            canvas.delete("crosshair")
            if event.x < pad_left or event.x > width - pad_right or event.y < chart_top or event.y > chart_bottom:
                return
            idx = int(round((event.x - pad_left) / max(plot_w, 1) * max(n - 1, 1)))
            idx = max(0, min(n - 1, idx))
            row = plot_df.iloc[idx]
            x = pad_left + idx * plot_w / max(n - 1, 1)
            canvas.create_line(x, chart_top, x, chart_bottom, fill=self.colors["muted_light"], width=1, tags="crosshair")
            items = [pd.to_datetime(row["Date"]).strftime("%d.%m.%Y"), f"score {fmt_pct(row['Breadth'], 1, signed=False)}"]
            if "RawAbove" in row and not pd.isna(row["RawAbove"]):
                items.append(f"> MM50 {fmt_pct(row['RawAbove'], 0, signed=False)}")
            if "AvgDistance" in row and not pd.isna(row["AvgDistance"]):
                items.append(f"écart {fmt_pct(row['AvgDistance'], 1)}")
            tooltip = "  ·  ".join(items)
            tw = min(width - 2 * pad_left, max(315, len(tooltip) * 6.0))
            tx = min(max(pad_left + 8, x + 12), width - pad_right - tw - 8)
            ty = chart_top + 10
            canvas.create_rectangle(tx, ty, tx + tw, ty + 34, fill=self.colors["panel"], outline=self.colors["line"], tags="crosshair")
            canvas.create_text(tx + 10, ty + 17, text=tooltip, anchor="w", fill=self.colors["ink"], font=("Avenir Next", 9, "bold"), tags="crosshair")

        canvas.bind("<Motion>", show_crosshair)
        canvas.bind("<Leave>", lambda _e: canvas.delete("crosshair"))

    def _redraw_breadth_after_toggle(self, canvas: tk.Canvas, label: str, df: pd.DataFrame, states: pd.DataFrame, default_visible: bool = True) -> None:
        self._toggle_series("breadth", label, default_visible)
        self._draw_breadth_chart(canvas, df, states)

    def _draw_cross_asset_dashboard(self, canvas: tk.Canvas, df: pd.DataFrame, corr: pd.DataFrame) -> None:
        canvas.delete("all")
        width = max(canvas.winfo_width(), 420)
        height = max(canvas.winfo_height(), 220)
        canvas.create_rectangle(0, 0, width, height, fill=self.colors["panel_alt"], outline="")
        if df.empty or "Ticker" not in df or "Performance" not in df:
            return

        work = df.copy()
        work["Performance"] = pd.to_numeric(work["Performance"], errors="coerce")
        work = work.dropna(subset=["Performance"]).sort_values("Performance", ascending=False)
        values = work.set_index("Ticker")["Performance"].to_dict()
        spy, tlt = values.get("SPY", float("nan")), values.get("TLT", float("nan"))
        qqq, iwm = values.get("QQQ", float("nan")), values.get("IWM", float("nan"))
        gld, uup = values.get("GLD", float("nan")), values.get("UUP", float("nan"))
        uso = values.get("USO", float("nan"))

        if not pd.isna(spy) and not pd.isna(tlt) and spy - tlt > 5:
            regime, note, tone = "Risk-on confirmé", "Actions > duration", self.colors["green"]
        elif not pd.isna(gld) and not pd.isna(spy) and gld - spy > 3:
            regime, note, tone = "Recherche de protection", "Or > actions", self.colors["gold"]
        elif not pd.isna(uup) and uup > 2 and not pd.isna(spy) and spy < 0:
            regime, note, tone = "Stress dollar", "USD ferme / actions faibles", self.colors["red"]
        else:
            regime, note, tone = "Confirmation mixte", "Signaux cross-asset partagés", self.colors["blue"]

        canvas.create_text(14, 10, text=regime, anchor="nw", fill=tone, font=("Avenir Next", 15, "bold"))
        canvas.create_text(14, 31, text=note, anchor="nw", fill=self.colors["muted"], font=("Avenir Next", 10, "bold"))

        left_x, left_y = 14, 56
        left_w = width * 0.46
        left_h = height - 68
        max_abs = max(1.0, float(work["Performance"].abs().max()))
        zero_x = left_x + left_w * 0.58
        row_h = max(18, min(27, left_h / max(len(work), 1)))
        canvas.create_line(zero_x, left_y, zero_x, min(height - 12, left_y + row_h * len(work)), fill=self.colors["line"], width=1)
        for idx, (_, row) in enumerate(work.iterrows()):
            y = left_y + idx * row_h + row_h / 2
            if y > height - 10:
                break
            value = float(row["Performance"])
            label = CROSS_ASSETS.get(str(row["Ticker"]), str(row["Ticker"]))
            color = self.colors["green"] if value >= 0 else self.colors["red"]
            bar_w = abs(value) / max_abs * (left_w * 0.34)
            x0, x1 = (zero_x, zero_x + bar_w) if value >= 0 else (zero_x - bar_w, zero_x)
            canvas.create_text(left_x, y, text=label[:14], anchor="w", fill=self.colors["ink"], font=("Avenir Next", 10, "bold"))
            canvas.create_rectangle(x0, y - 4, x1, y + 4, fill=color, outline="")
            canvas.create_text(left_x + left_w - 4, y, text=fmt_pct(value), anchor="e", fill=color, font=("Avenir Next", 10, "bold"))

        right_x = left_x + left_w + 12
        right_w = width - right_x - 12
        metrics = [
            ("SPY-TLT", spy - tlt if not pd.isna(spy) and not pd.isna(tlt) else float("nan")),
            ("QQQ-IWM", qqq - iwm if not pd.isna(qqq) and not pd.isna(iwm) else float("nan")),
            ("GLD-UUP", gld - uup if not pd.isna(gld) and not pd.isna(uup) else float("nan")),
            ("USO", uso), ("UUP", uup), ("TLT", tlt),
        ]
        gap = 5
        cell_w = (right_w - gap) / 2
        cell_h = 35
        for idx, (label, value) in enumerate(metrics):
            row, col = divmod(idx, 2)
            x = right_x + col * (cell_w + gap)
            y = 52 + row * (cell_h + gap)
            accent = self.colors["green"] if not pd.isna(value) and value >= 0 else self.colors["red"]
            canvas.create_rectangle(x, y, x + cell_w, y + cell_h, fill=self.colors["panel"], outline=self.colors["line_soft"])
            canvas.create_text(x + 7, y + 8, text=label, anchor="w", fill=self.colors["muted"], font=("Avenir Next", 7, "bold"))
            canvas.create_text(x + cell_w - 7, y + 18, text=fmt_pct(value), anchor="e", fill=accent, font=("Avenir Next", 10, "bold"))

        corr_assets = [t for t in ["QQQ", "IWM", "TLT", "GLD", "UUP", "USO"] if t in corr.columns and "SPY" in corr.index]
        corr_y = 52 + 3 * (cell_h + gap) + 8
        if corr_assets and corr_y + 42 <= height:
            canvas.create_text(right_x, corr_y, text="CORR. AU S&P 500", anchor="nw", fill=self.colors["muted"], font=("Avenir Next", 7, "bold"))
            cgap = 4
            cw = (right_w - cgap * (len(corr_assets) - 1)) / len(corr_assets)
            for idx, ticker in enumerate(corr_assets):
                value = float(corr.loc["SPY", ticker])
                x = right_x + idx * (cw + cgap)
                y = corr_y + 15
                canvas.create_rectangle(x, y, x + cw, y + 26, fill=self._corr_bg(value), outline=self.colors["line_soft"])
                canvas.create_text(x + cw / 2, y + 8, text=ticker, anchor="center", fill=self.colors["ink"], font=("Avenir Next", 7, "bold"))
                canvas.create_text(x + cw / 2, y + 19, text=f"{value:.2f}", anchor="center", fill=self.colors["muted"], font=("Avenir Next", 7, "bold"))

    def _draw_line_chart(self, canvas: tk.Canvas, df: pd.DataFrame, series: list[tuple[str, str]], percent: bool, zones: Optional[list[tuple[float, Optional[float], str]]] = None, hlines: Optional[list[float]] = None, y_min: Optional[float] = None, y_max: Optional[float] = None) -> None:
        canvas.delete("all")
        width = max(canvas.winfo_width(), 280)
        height = max(canvas.winfo_height(), 90)
        pad_x, pad_y = 34, 20
        if df.empty or "Date" not in df:
            return
        plot_df = df.copy()
        plot_df["Date"] = pd.to_datetime(plot_df["Date"], errors="coerce")
        plot_df = plot_df.dropna(subset=["Date"])
        values = []
        for col, _color in series:
            if col in plot_df:
                values.extend(pd.to_numeric(plot_df[col], errors="coerce").dropna().tolist())
        if not values:
            return
        low = min(values) if y_min is None else y_min
        high = max(values) if y_max is None else y_max
        if zones:
            for z0, z1, _color in zones:
                zone_high = z1 if z1 is not None else z0 + 0.35
                low = min(low, z0, zone_high)
                high = max(high, z0, zone_high)
        if math.isclose(low, high):
            low -= 1
            high += 1
        pad = (high - low) * 0.06
        low -= 0 if y_min is not None else pad
        high += 0 if y_max is not None else pad

        def y_for(value: float) -> float:
            return height - pad_y - ((value - low) / (high - low)) * (height - 2 * pad_y)

        if zones:
            for z0, z1, color in zones:
                top = y_for(high if z1 is None else z1)
                bottom = y_for(z0)
                canvas.create_rectangle(pad_x, top, width - pad_x, bottom, fill=color, outline="")
        for i in range(5):
            y = pad_y + i * (height - 2 * pad_y) / 4
            canvas.create_line(pad_x, y, width - pad_x, y, fill=self.colors["grid"], width=1)
        if hlines:
            for line in hlines:
                y = y_for(line)
                canvas.create_line(pad_x, y, width - pad_x, y, fill=self.colors["line"], width=1, dash=(5, 5))
                label = fmt_pct(line, 0, signed=False) if percent else fmt_num(line, 0)
                canvas.create_text(width - pad_x, y - 8, text=label, anchor="e", fill=self.colors["muted"], font=("Avenir Next", 10, "bold"))
        legend_x = pad_x
        for label, color in series:
            if label not in plot_df:
                continue
            canvas.create_line(legend_x, 12, legend_x + 18, 12, fill=color, width=3)
            canvas.create_text(legend_x + 24, 12, text=label, anchor="w", fill=self.colors["muted"], font=("Avenir Next", 10, "bold"))
            legend_x += 92
        n = len(plot_df)
        for series_index, (col, color) in enumerate(series):
            if col not in plot_df:
                continue
            clean = pd.to_numeric(plot_df[col], errors="coerce")
            points: list[float] = []
            for idx, value in enumerate(clean):
                if pd.isna(value):
                    continue
                x = pad_x + idx * (width - 2 * pad_x) / max(n - 1, 1)
                y = y_for(float(value))
                points.extend([x, y])
            if len(points) >= 4:
                dash = (5, 5) if "MM50" in col else None
                canvas.create_line(points, fill=color, width=2 if "MM" in col else 3, smooth=True, dash=dash)
            if len(points) >= 2:
                canvas.create_oval(points[-2] - 3, points[-1] - 3, points[-2] + 3, points[-1] + 3, fill=color, outline=color)
                latest = clean.dropna()
                if not latest.empty:
                    last_value = float(latest.iloc[-1])
                    label = fmt_pct(last_value, 1, signed=False) if percent else fmt_num(last_value, 2)
                    offset = (series_index - (len(series) - 1) / 2) * 10
                    canvas.create_text(points[-2] + 6, points[-1] + offset, text=label, anchor="w", fill=color, font=("Avenir Next", 10, "bold"))
        dates = plot_df["Date"].dropna()
        if not dates.empty:
            canvas.create_text(pad_x, height - 8, text=dates.iloc[0].strftime("%Y-%m"), anchor="w", fill=self.colors["muted"], font=("Avenir Next", 10, "bold"))
            canvas.create_text(width - pad_x, height - 8, text=dates.iloc[-1].strftime("%Y-%m"), anchor="e", fill=self.colors["muted"], font=("Avenir Next", 10, "bold"))

    def _draw_bar_chart(self, canvas: tk.Canvas, df: pd.DataFrame, label_col: str, value_col: str) -> None:
        canvas.delete("all")
        width = max(canvas.winfo_width(), 300)
        height = max(canvas.winfo_height(), 180)
        if df.empty or label_col not in df or value_col not in df:
            return
        show = df.sort_values(value_col, ascending=True).tail(12)
        values = pd.to_numeric(show[value_col], errors="coerce").fillna(0)
        max_abs = max(1.0, float(values.abs().max()))
        pad_x, pad_y = 170, 24
        zero_x = pad_x + (width - pad_x - 52) / 2
        canvas.create_line(zero_x, pad_y, zero_x, height - pad_y, fill=self.colors["line"], width=1)
        row_h = (height - 2 * pad_y) / max(len(show), 1)
        for i, (_, row) in enumerate(show.iterrows()):
            value = float(row[value_col])
            y = pad_y + i * row_h + row_h / 2
            bar_w = abs(value) / max_abs * ((width - pad_x - 68) / 2)
            color = self.colors["green"] if value >= 0 else self.colors["red"]
            x0 = zero_x if value >= 0 else zero_x - bar_w
            x1 = zero_x + bar_w if value >= 0 else zero_x
            canvas.create_text(12, y, text=str(row[label_col]), anchor="w", fill=self.colors["ink"], font=("Avenir Next", 10, "bold"))
            canvas.create_rectangle(x0, y - 7, x1, y + 7, fill=color, outline="")
            canvas.create_text(width - 12, y, text=fmt_pct(value), anchor="e", fill=color, font=("Avenir Next", 10, "bold"))

    def _draw_heatmap(self, canvas: tk.Canvas, df: pd.DataFrame) -> None:
        canvas.delete("all")
        width = max(canvas.winfo_width(), 470)
        height = max(canvas.winfo_height(), 210)
        if df.empty:
            return
        work = df.copy()
        work["Category"] = work["Ticker"].map(lambda t: ASSETS.get(t, ("", "Autres"))[1])
        work = work[work["Category"] != "Volatilité"].sort_values(["Category", "Performance"], ascending=[True, False])
        categories = [cat for cat in ["Indices", "Secteurs", "Macro"] if cat in set(work["Category"])]
        margin, gap = 14, 8
        usable = width - 2 * margin - gap * max(len(categories) - 1, 0)
        weights = {"Indices": 0.27, "Secteurs": 0.43, "Macro": 0.30}
        widths = {cat: usable * weights.get(cat, 1 / max(len(categories), 1)) for cat in categories}

        canvas.create_rectangle(0, 0, width, height, fill=self.colors["panel_alt"], outline="")
        cursor_x = margin
        for c_idx, category in enumerate(categories):
            col_w = widths[category]
            x = cursor_x
            cursor_x += col_w + gap
            rows = work[work["Category"] == category].copy()
            avg = rows["Performance"].mean()
            panel_color = self._perf_bg(float(avg), subtle=True)
            canvas.create_rectangle(x, 14, x + col_w, height - 16, fill=self.colors["panel"], outline=self.colors["line_soft"])
            canvas.create_rectangle(x, 14, x + col_w, 60, fill=panel_color, outline=self.colors["line_soft"])
            canvas.create_text(x + 12, 28, text=category.upper(), anchor="w", fill=self.colors["ink"], font=("Avenir Next", 9, "bold"))
            canvas.create_text(x + 12, 47, text=fmt_pct(avg), anchor="w", fill=self.colors["green"] if avg >= 0 else self.colors["red"], font=("Avenir Next", 10, "bold"))

            top = 72
            available_h = height - top - 30
            if category == "Secteurs":
                cols = 2
            else:
                cols = 1
            tile_gap = 7
            tile_w = (col_w - 24 - tile_gap * (cols - 1)) / cols
            tile_rows = math.ceil(len(rows) / cols)
            tile_h = min(40, max(18, (available_h - tile_gap * max(tile_rows - 1, 0)) / max(tile_rows, 1)))

            for idx, (_, row) in enumerate(rows.iterrows()):
                r = idx // cols
                c = idx % cols
                tx = x + 12 + c * (tile_w + tile_gap)
                ty = top + r * (tile_h + tile_gap)
                value = float(row["Performance"])
                fill = self._perf_bg(value)
                accent = self.colors["green"] if value >= 0 else self.colors["red"]
                canvas.create_rectangle(tx, ty, tx + tile_w, ty + tile_h, fill=fill, outline=self.colors["line_soft"])
                canvas.create_rectangle(tx, ty, tx + 4, ty + tile_h, fill=accent, outline=accent)
                if category == "Secteurs" and tile_w < 125:
                    canvas.create_text(tx + 10, ty + tile_h / 2, text=str(row["Ticker"]), anchor="w", fill=self.colors["ink"], font=("Avenir Next", 10, "bold"))
                    canvas.create_text(tx + tile_w - 7, ty + tile_h / 2, text=fmt_pct(value), anchor="e", fill=accent, font=("Avenir Next", 10, "bold"))
                else:
                    canvas.create_text(tx + 12, ty + tile_h / 2 - 7, text=str(row["Ticker"]), anchor="w", fill=self.colors["ink"], font=("Avenir Next", 10, "bold"))
                    name = str(row["Nom"])
                    max_chars = 18
                    if len(name) > max_chars:
                        name = name[:max_chars - 1] + "…"
                    detail = name
                    if category in {"Indices", "Macro"} and "Valeur" in row and not pd.isna(row["Valeur"]):
                        detail = f"{name} · {fmt_num(row['Valeur'], 2)}"
                    canvas.create_text(tx + 12, ty + tile_h / 2 + 8, text=detail, anchor="w", fill=self.colors["muted"], font=("Avenir Next", 8))
                    canvas.create_text(tx + tile_w - 9, ty + tile_h / 2, text=fmt_pct(value), anchor="e", fill=accent, font=("Avenir Next", 10, "bold"))

    def _perf_bg(self, value: float, subtle: bool = False) -> str:
        scale = 12 if subtle else 8
        if value >= 0:
            return self._blend(self.colors["panel_alt"], self.colors["green_bg"], min(abs(value) / scale, 1))
        return self._blend(self.colors["panel_alt"], self.colors["red_bg"], min(abs(value) / scale, 1))

    def _draw_corr(self, canvas: tk.Canvas, corr: pd.DataFrame) -> None:
        canvas.delete("all")
        width = max(canvas.winfo_width(), 420)
        height = max(canvas.winfo_height(), 260)
        if corr.empty:
            return
        labels = [CROSS_ASSETS.get(col, col) for col in corr.columns]
        n = len(labels)
        pad_left, pad_top = 126, 34
        size = min((width - pad_left - 18) / max(n, 1), (height - pad_top - 18) / max(n, 1))
        for i, label in enumerate(labels):
            canvas.create_text(pad_left - 10, pad_top + i * size + size / 2, text=label, anchor="e", fill=self.colors["muted"], font=("Avenir Next", 10, "bold"))
            canvas.create_text(pad_left + i * size + size / 2, pad_top - 10, text=label[:8], anchor="s", fill=self.colors["muted"], font=("Avenir Next", 10, "bold"))
        for r in range(n):
            for c in range(n):
                value = float(corr.iloc[r, c])
                color = self._corr_bg(value)
                x = pad_left + c * size
                y = pad_top + r * size
                canvas.create_rectangle(x, y, x + size - 2, y + size - 2, fill=color, outline=self.colors["panel_alt"])
                canvas.create_text(x + size / 2, y + size / 2, text=f"{value:.2f}", fill=self.colors["ink"], font=("Avenir Next", 10, "bold"))

    def _corr_bg(self, value: float) -> str:
        if pd.isna(value):
            return self.colors["panel"]
        if value >= 0:
            return self._blend(self.colors["panel_alt"], self.colors["gold_bg"], min(abs(value), 1))
        return self._blend(self.colors["panel_alt"], self.colors["blue_bg"], min(abs(value), 1))

    def _tree(self, parent: tk.Frame, df: pd.DataFrame, columns: list[str]) -> None:
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=max(8, min(len(df), 12)), style="Market.Treeview")
        tree.pack(fill="both", expand=True, padx=12, pady=12)
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=130 if col != "Nom" else 260, anchor="e" if col == "Performance" else "w", stretch=True)
        tree.tag_configure("even", background=self.colors["panel"])
        tree.tag_configure("odd", background=self.colors["panel_alt"])
        tree.tag_configure("positive", foreground=self.colors["green"])
        tree.tag_configure("negative", foreground=self.colors["red"])
        for index, (_, row) in enumerate(df.iterrows()):
            values = [fmt_pct(row[col]) if col == "Performance" else str(row[col]) for col in columns]
            tags = ["even" if index % 2 == 0 else "odd"]
            try:
                tags.append("positive" if float(row.get("Performance", 0)) >= 0 else "negative")
            except Exception:
                pass
            tree.insert("", "end", values=values, tags=tuple(tags))

    def _loading(self) -> None:
        panel = tk.Frame(self.body, bg=self.colors["panel"], highlightbackground=self.colors["line_soft"], highlightthickness=1)
        panel.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=4, pady=4)
        tk.Label(panel, text="Chargement des données de marché", bg=self.colors["panel"], fg=self.colors["ink"], font=("Avenir Next", 18, "bold")).pack(expand=True)
        tk.Label(panel, text="Prix, positionnement options et stress macro sont chargés en parallèle lorsque possible.", bg=self.colors["panel"], fg=self.colors["muted"], font=("Avenir Next", 10)).pack(pady=(0, 14))

    def _error(self, message: str) -> None:
        panel = tk.Frame(self.body, bg=self.colors["panel"], highlightbackground=self.colors["line_soft"], highlightthickness=1)
        panel.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=4, pady=4)
        tk.Label(panel, text="Impossible de construire le tableau", bg=self.colors["panel"], fg=self.colors["red"], font=("Avenir Next", 18, "bold")).pack(expand=True)
        tk.Label(panel, text=message, bg=self.colors["panel"], fg=self.colors["muted"], wraplength=1000, justify="center", font=("Avenir Next", 10)).pack(pady=(0, 14))

    def _clear(self, widget: tk.Widget) -> None:
        for child in widget.winfo_children():
            child.destroy()



def main() -> None:
    app = MarketIntelligenceDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
