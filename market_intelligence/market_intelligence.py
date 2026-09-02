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
    