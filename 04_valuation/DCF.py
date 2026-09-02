#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Valorisation sectorielle du S&P 500 — version optimisée.

Sortie unique :
    sp500_sector_dcf_chart.PNG

Le fichier est créé dans le même dossier que ce script et remplacé à chaque
exécution. Aucun dossier output, CSV, Excel, SVG, LaTeX ou fichier temporaire
persistant n'est créé.

Dépendances :
    python -m pip install pandas numpy requests yfinance matplotlib openpyxl python-dotenv

Exécution :
    python DCF_optimized.py
    python DCF_optimized.py --top-n 5 --workers 12
"""

from __future__ import annotations

import argparse
import io
import html as html_lib
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import requests
import yfinance as yf

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "sp500_sector_dcf_chart.PNG"

GICS_SECTOR_ETFS: Dict[str, str] = {
    "Information Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}

STATE_STREET_HOLDINGS_URL_TEMPLATE = (
    "https://www.ssga.com/us/en/intermediary/etfs/library-content/products/"
    "fund-data/etfs/us/holdings-daily-us-en-{ticker}.xlsx"
)

DEFAULT_TOP_N = 5
DEFAULT_WORKERS = 12
DEFAULT_REQUEST_TIMEOUT = 12

CLASSIFICATION_THRESHOLDS = {
    "undervalued": 0.10,
    "overvalued": -0.10,
}

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
)

_thread_local = threading.local()


# =============================================================================
# Structures de données
# =============================================================================

@dataclass(frozen=True)
class DataSourceInfo:
    name: str
    url: str
    retrieved_at_utc: str
    as_of_date: Optional[str] = None
    notes: str = ""


@dataclass(frozen=True)
class Holding:
    sector: str
    etf: str
    ticker: str
    name: str
    weight_in_etf: float


@dataclass(frozen=True)
class ExternalValuation:
    intrinsic_value_per_share: Optional[float]
    source: str
    url: str
    retrieved_at_utc: str
    as_of_date: Optional[str] = None
    dcf_value: Optional[float] = None
    relative_value: Optional[float] = None
    notes: str = ""


@dataclass(frozen=True)
class ValuationResult:
    ticker: str
    sector: str
    etf: str
    weight_in_etf: float
    price: Optional[float]
    intrinsic_value_per_share: Optional[float]
    upside_downside_pct: Optional[float]
    margin_of_safety: Optional[float]
    classification: str
    reliability_score: float
    source: str


# =============================================================================
# Utilitaires
# =============================================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(str(value).strip().replace(",", "").replace("%", ""))
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def normalize_ticker(ticker: Any) -> str:
    if ticker is None:
        return ""
    return str(ticker).strip().upper().replace(" ", "").replace("/", ".")


def yfinance_symbol(ticker: str) -> str:
    return ticker.replace(".", "-")


def classify_margin(margin: Optional[float]) -> str:
    if margin is None or not np.isfinite(margin):
        return "insufficient data"
    if margin >= CLASSIFICATION_THRESHOLDS["undervalued"]:
        return "undervalued"
    if margin <= CLASSIFICATION_THRESHOLDS["overvalued"]:
        return "overvalued"
    return "fairly valued"


def weighted_average(values: Sequence[Optional[float]], weights: Sequence[float]) -> Optional[float]:
    pairs = [
        (float(v), float(w))
        for v, w in zip(values, weights)
        if v is not None and np.isfinite(v) and w is not None and np.isfinite(w) and w > 0
    ]
    if not pairs:
        return None
    vals = np.fromiter((v for v, _ in pairs), dtype=float)
    wts = np.fromiter((w for _, w in pairs), dtype=float)
    total = wts.sum()
    if total <= 0:
        return None
    return float(np.dot(vals, wts) / total)


def parse_date_from_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    for pattern in (
        r"(\d{1,2}/\d{1,2}/\d{4})",
        r"(\d{4}-\d{2}-\d{2})",
        r"([A-Z][a-z]+ \d{1,2}, \d{4})",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def get_http_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=1)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _thread_local.session = session
    return session


# =============================================================================
# Holdings et poids sectoriels — téléchargements parallélisés
# =============================================================================

def download_excel(url: str, timeout: int) -> pd.DataFrame:
    response = get_http_session().get(url, timeout=timeout)
    response.raise_for_status()
    return pd.read_excel(io.BytesIO(response.content), header=None)


def detect_header_row(raw_df: pd.DataFrame, required_terms: Sequence[str]) -> int:
    required = tuple(term.lower() for term in required_terms)
    for idx, row in raw_df.iterrows():
        text = " | ".join(str(cell).strip().lower() for cell in row.tolist())
        if all(term in text for term in required):
            return int(idx)
    raise ValueError(f"En-tête introuvable: {required_terms}")


def find_first_row_text(df: pd.DataFrame, patterns: Iterable[str]) -> Optional[str]:
    patterns_lower = tuple(p.lower() for p in patterns)
    for _, row in df.iterrows():
        text = " | ".join(str(x) for x in row.dropna().tolist()).strip()
        if any(pattern in text.lower() for pattern in patterns_lower):
            return text
    return None


def normalize_holdings_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[Any, str] = {}
    for col in df.columns:
        name = re.sub(r"\s+", " ", str(col).strip().lower())
        if name in {"ticker", "symbol"} or "ticker" in name:
            rename_map[col] = "ticker"
        elif name in {"name", "company name", "holding name"} or "name" in name:
            rename_map[col] = "name"
        elif "weight" in name:
            rename_map[col] = "weight"
        elif "sector" in name:
            rename_map[col] = "sector"
    return df.rename(columns=rename_map).copy()


def parse_state_street_holdings(
    raw_df: pd.DataFrame,
    ticker: str,
    expected_sector: Optional[str],
) -> Tuple[pd.DataFrame, DataSourceInfo]:
    as_of_text = find_first_row_text(raw_df, ["as of", "holdings as of"])
    as_of_date = parse_date_from_text(as_of_text)

    header_row = detect_header_row(raw_df, ["ticker", "weight"])
    df = raw_df.iloc[header_row + 1 :].copy()
    df.columns = [str(c).strip() for c in raw_df.iloc[header_row].tolist()]
    df = normalize_holdings_columns(df)

    if "ticker" not in df.columns or "weight" not in df.columns:
        raise ValueError(f"Colonnes ticker/weight absentes pour {ticker}")
    if "name" not in df.columns:
        df["name"] = df["ticker"]
    if "sector" not in df.columns:
        df["sector"] = expected_sector

    df["ticker"] = df["ticker"].map(normalize_ticker)
    df["name"] = df["name"].astype(str).str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df[df["ticker"].ne("") & df["weight"].notna() & (df["weight"] > 0)].copy()

    if float(df["weight"].sum()) > 1.5:
        df["weight"] = df["weight"] / 100.0

    cash_like = {"CASH_USD", "USD", "CASH", "US DOLLAR", "FUTURES", "SWAP"}
    df = df[~df["ticker"].isin(cash_like)].copy()

    if expected_sector is not None:
        df["sector"] = expected_sector
    else:
        df["sector"] = df["sector"].replace({"-": np.nan, "": np.nan, "nan": np.nan})

    source = DataSourceInfo(
        name=f"State Street Global Advisors holdings daily ({ticker.upper()})",
        url=STATE_STREET_HOLDINGS_URL_TEMPLATE.format(ticker=ticker.lower()),
        retrieved_at_utc=utc_now_iso(),
        as_of_date=as_of_date,
        notes="Public daily holdings Excel file.",
    )
    return df.reset_index(drop=True), source


def fetch_state_street_holdings(
    ticker: str,
    expected_sector: Optional[str],
    timeout: int,
) -> Tuple[pd.DataFrame, DataSourceInfo]:
    url = STATE_STREET_HOLDINGS_URL_TEMPLATE.format(ticker=ticker.lower())
    return parse_state_street_holdings(download_excel(url, timeout), ticker, expected_sector)


def fetch_sp500_constituent_sectors(timeout: int) -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = get_http_session().get(url, timeout=timeout)
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text))
    if not tables:
        raise ValueError("Table S&P 500 introuvable sur Wikipedia")
    table = tables[0]
    if "Symbol" not in table.columns or "GICS Sector" not in table.columns:
        raise ValueError("Colonnes Symbol/GICS Sector absentes")
    out = table[["Symbol", "GICS Sector"]].rename(columns={"Symbol": "ticker", "GICS Sector": "sector"})
    out["ticker"] = out["ticker"].map(normalize_ticker)
    out["sector"] = out["sector"].astype(str).str.strip()
    return out


def fetch_holdings_bundle(
    top_n: int,
    workers: int,
    timeout: int,
) -> Tuple[pd.DataFrame, DataSourceInfo, List[Holding]]:
    requests_to_make: List[Tuple[str, Optional[str]]] = [("SPY", None)]
    requests_to_make += [(etf, sector) for sector, etf in GICS_SECTOR_ETFS.items()]

    downloaded: Dict[str, Tuple[pd.DataFrame, DataSourceInfo]] = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(requests_to_make))) as executor:
        futures = {
            executor.submit(fetch_state_street_holdings, ticker, sector, timeout): ticker
            for ticker, sector in requests_to_make
        }
        for future in as_completed(futures):
            ticker = futures[future]
            downloaded[ticker] = future.result()

    spy, spy_source = downloaded["SPY"]
    valid_spy_sectors = set(spy["sector"].dropna()) if "sector" in spy.columns else set()
    if not set(GICS_SECTOR_ETFS).issubset(valid_spy_sectors):
        sector_map = fetch_sp500_constituent_sectors(timeout).rename(columns={"sector": "sector_wiki"})
        spy = spy.merge(sector_map, on="ticker", how="left")
        if "sector" not in spy.columns:
            spy["sector"] = spy["sector_wiki"]
        else:
            spy["sector"] = spy["sector"].fillna(spy["sector_wiki"])
        spy = spy.drop(columns=["sector_wiki"], errors="ignore")

    sector_weights = (
        spy.dropna(subset=["sector"])
        .groupby("sector", as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "sp500_sector_weight"})
    )
    sector_weights = sector_weights[sector_weights["sector"].isin(GICS_SECTOR_ETFS)].copy()
    total = float(sector_weights["sp500_sector_weight"].sum())
    if total <= 0:
        raise ValueError("Poids sectoriels SPY indisponibles")
    sector_weights["sp500_sector_weight"] /= total

    missing = set(GICS_SECTOR_ETFS) - set(sector_weights["sector"])
    if missing:
        raise ValueError(f"Poids sectoriels manquants: {sorted(missing)}")

    holdings: List[Holding] = []
    for sector, etf in GICS_SECTOR_ETFS.items():
        etf_df, _ = downloaded[etf]
        selected = etf_df.nlargest(top_n, "weight")
        for row in selected.itertuples(index=False):
            holdings.append(
                Holding(
                    sector=sector,
                    etf=etf,
                    ticker=normalize_ticker(row.ticker),
                    name=str(row.name),
                    weight_in_etf=float(row.weight),
                )
            )

    return sector_weights, spy_source, holdings


# =============================================================================
# Prix de marché — un téléchargement Yahoo groupé
# =============================================================================

def fetch_market_prices(tickers: Sequence[str]) -> Dict[str, Optional[float]]:
    unique = list(dict.fromkeys(tickers))
    symbols = [yfinance_symbol(t) for t in unique]
    if not symbols:
        return {}

    prices: Dict[str, Optional[float]] = {ticker: None for ticker in unique}
    try:
        data = yf.download(
            tickers=symbols,
            period="5d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="column",
        )
        if data.empty:
            return prices

        if len(symbols) == 1:
            close = data["Close"].dropna()
            prices[unique[0]] = safe_float(close.iloc[-1]) if not close.empty else None
            return prices

        close_block = data["Close"] if "Close" in data.columns.get_level_values(0) else pd.DataFrame()
        for ticker, symbol in zip(unique, symbols):
            if symbol in close_block.columns:
                series = close_block[symbol].dropna()
                if not series.empty:
                    prices[ticker] = safe_float(series.iloc[-1])
    except Exception as exc:
        logging.warning("Téléchargement groupé des prix Yahoo impossible: %s", exc)

    return prices


def fetch_yahoo_target_price(ticker: str) -> Optional[float]:
    try:
        info = yf.Ticker(yfinance_symbol(ticker)).get_info()
        return safe_float(info.get("targetMeanPrice")) or safe_float(info.get("targetMedianPrice"))
    except Exception:
        return None


# =============================================================================
# Valeurs intrinsèques externes
# =============================================================================

def fetch_fmp_external_valuation(ticker: str, timeout: int) -> Optional[ExternalValuation]:
    api_key = os.getenv("FMP_API_KEY") or os.getenv("FINANCIAL_MODELING_PREP_API_KEY")
    if not api_key:
        return None
    url = f"https://financialmodelingprep.com/api/v3/discounted-cash-flow/{ticker}?apikey={api_key}"
    try:
        response = get_http_session().get(url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or not payload:
            return None
        item = payload[0]
        intrinsic = safe_float(item.get("dcf")) or safe_float(item.get("DCF"))
        if intrinsic is None:
            return None
        return ExternalValuation(
            intrinsic_value_per_share=intrinsic,
            source="Financial Modeling Prep Discounted Cash Flow API",
            url=url.split("?apikey=")[0],
            retrieved_at_utc=utc_now_iso(),
            as_of_date=str(item.get("date")) if item.get("date") else None,
            dcf_value=intrinsic,
            notes="External DCF value from FMP.",
        )
    except Exception:
        return None


def parse_alpha_spread_html(text: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[str]]:
    base_match = re.search(
        r"baseCaseData&quot;:\[\{&quot;value&quot;:([0-9.eE+-]+),&quot;dcfValue&quot;:([0-9.eE+-]+),&quot;relativeValue&quot;:([0-9.eE+-]+).*?updatedAt&quot;:\[&quot;([^&]+)&quot;",
        text,
        re.S,
    )
    if base_match:
        return (
            safe_float(base_match.group(1)),
            safe_float(base_match.group(2)),
            safe_float(base_match.group(3)),
            base_match.group(4),
        )

    plain = re.sub(r"<[^>]+>", " ", text)
    faq_match = re.search(
        r"intrinsic value\s+for.*?under the\s+Base Case\s+is\s+([0-9,.]+)\s+USD",
        plain,
        re.I | re.S,
    )
    if faq_match:
        return safe_float(faq_match.group(1).replace(",", "")), None, None, None
    return None, None, None, None


def fetch_alpha_spread_external_valuation(ticker: str, timeout: int) -> Optional[ExternalValuation]:
    raw = ticker.lower()
    variants = list(dict.fromkeys([raw, raw.replace(".", "-"), raw.replace("-", ".")]))
    for clean in variants:
        for exchange in ("nasdaq", "nyse", "amex"):
            url = f"https://www.alphaspread.com/security/{exchange}/{clean}/summary"
            try:
                response = get_http_session().get(url, timeout=timeout)
                if response.status_code != 200:
                    continue
                intrinsic, dcf_value, relative_value, as_of = parse_alpha_spread_html(response.text)
                if intrinsic is not None:
                    return ExternalValuation(
                        intrinsic_value_per_share=intrinsic,
                        source="Alpha Spread intrinsic value base case",
                        url=url,
                        retrieved_at_utc=utc_now_iso(),
                        as_of_date=as_of,
                        dcf_value=dcf_value,
                        relative_value=relative_value,
                        notes="External intrinsic value parsed from Alpha Spread.",
                    )
            except Exception:
                continue
    return None


def parse_valueinvesting_html(text: str, ticker: str) -> Tuple[Optional[float], Optional[str]]:
    plain = html_lib.unescape(re.sub(r"<[^>]+>", " ", text))
    plain = re.sub(r"\s+", " ", plain)
    ticker_pattern = re.escape(ticker.upper())
    main_match = re.search(
        rf"As of\s+([0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}),\s+the\s+Intrinsic\s+Value\s+of\s+.*?\({ticker_pattern}\)\s+is\s+([0-9,.]+)\s+USD",
        plain,
        re.I,
    )
    if main_match:
        return safe_float(main_match.group(2).replace(",", "")), main_match.group(1)
    generic_match = re.search(
        r"Intrinsic\s+Value\s+of\s+.+?\([A-Z0-9.\-]+\)\s+is\s+([0-9,.]+)\s+USD",
        plain,
        re.I,
    )
    if generic_match and ticker.upper() in plain[: plain.find(generic_match.group(0)) + len(generic_match.group(0))].upper():
        return safe_float(generic_match.group(1).replace(",", "")), None
    return None, None


def fetch_valueinvesting_external_valuation(ticker: str, timeout: int) -> Optional[ExternalValuation]:
    raw = ticker.upper()
    variants = list(dict.fromkeys([raw, raw.replace(".", "-"), raw.replace("-", "."), raw.replace(".", "")]))
    for candidate in variants:
        url = f"https://valueinvesting.io/{candidate}/valuation/intrinsic-value"
        try:
            response = get_http_session().get(url, timeout=timeout, allow_redirects=True)
            if response.status_code != 200 or "/valuation/intrinsic-value" not in response.url:
                continue
            intrinsic, as_of = parse_valueinvesting_html(response.text, candidate)
            if intrinsic is None:
                intrinsic, as_of = parse_valueinvesting_html(response.text, ticker)
            if intrinsic is not None:
                return ExternalValuation(
                    intrinsic_value_per_share=intrinsic,
                    source="ValueInvesting.io intrinsic value",
                    url=url,
                    retrieved_at_utc=utc_now_iso(),
                    as_of_date=as_of,
                    dcf_value=intrinsic,
                    notes="External intrinsic value parsed from ValueInvesting.io.",
                )
        except Exception:
            continue
    return None


def is_usable_external_value(value: Optional[float], price: Optional[float]) -> bool:
    if value is None or not np.isfinite(value) or value <= 0:
        return False
    if price is None or not np.isfinite(price) or price <= 0:
        return True
    return price * 0.01 <= value <= price * 25.0


def combine_external_valuations(
    ticker: str,
    valuations: Sequence[ExternalValuation],
    price: Optional[float],
) -> ExternalValuation:
    usable = [v for v in valuations if is_usable_external_value(v.intrinsic_value_per_share, price)]
    if not usable:
        return ExternalValuation(None, "No external intrinsic value available", "", utc_now_iso())
    if len(usable) == 1:
        return usable[0]

    values = np.array([float(v.intrinsic_value_per_share) for v in usable if v.intrinsic_value_per_share is not None])
    median = float(np.median(values))
    return ExternalValuation(
        intrinsic_value_per_share=median,
        source=f"Median external intrinsic value ({len(usable)} sources)",
        url=next((v.url for v in usable if v.url), ""),
        retrieved_at_utc=utc_now_iso(),
        as_of_date=max((v.as_of_date for v in usable if v.as_of_date), default=None),
        dcf_value=median,
        relative_value=float(np.std(values) / median) if median else None,
        notes=f"{ticker}: median of {len(usable)} usable external values.",
    )


def fetch_external_intrinsic_value(
    ticker: str,
    price: Optional[float],
    timeout: int,
) -> ExternalValuation:
    valuations: List[ExternalValuation] = []

    fmp = fetch_fmp_external_valuation(ticker, timeout)
    if fmp is not None:
        valuations.append(fmp)

    alpha = fetch_alpha_spread_external_valuation(ticker, timeout)
    if alpha is not None:
        valuations.append(alpha)

    valueinvesting = fetch_valueinvesting_external_valuation(ticker, timeout)
    if valueinvesting is not None:
        valuations.append(valueinvesting)

    usable = [v for v in valuations if is_usable_external_value(v.intrinsic_value_per_share, price)]
    if usable:
        return combine_external_valuations(ticker, usable, price)

    target = fetch_yahoo_target_price(ticker)
    if is_usable_external_value(target, price):
        return ExternalValuation(
            intrinsic_value_per_share=target,
            source="Yahoo Finance analyst target price proxy",
            url=f"https://finance.yahoo.com/quote/{yfinance_symbol(ticker)}",
            retrieved_at_utc=utc_now_iso(),
            notes="Fallback only: consensus analyst target price, not a DCF value.",
        )

    return ExternalValuation(
        intrinsic_value_per_share=None,
        source="No external intrinsic value available",
        url="",
        retrieved_at_utc=utc_now_iso(),
        notes="No usable external value or Yahoo target-price proxy.",
    )


def provider_score(source: str) -> float:
    if source.startswith("Median external"):
        return 0.82
    if source.startswith("Financial Modeling Prep"):
        return 0.80
    if source.startswith("ValueInvesting.io"):
        return 0.74
    if source.startswith("Alpha Spread"):
        return 0.70
    if source.startswith("Yahoo Finance analyst target"):
        return 0.45
    return 0.0


def value_holding(holding: Holding, price: Optional[float], timeout: int) -> ValuationResult:
    external = fetch_external_intrinsic_value(holding.ticker, price, timeout)
    intrinsic = external.intrinsic_value_per_share
    margin = (intrinsic - price) / price if intrinsic is not None and price is not None and price > 0 else None
    return ValuationResult(
        ticker=holding.ticker,
        sector=holding.sector,
        etf=holding.etf,
        weight_in_etf=holding.weight_in_etf,
        price=price,
        intrinsic_value_per_share=intrinsic,
        upside_downside_pct=margin,
        margin_of_safety=margin,
        classification=classify_margin(margin),
        reliability_score=provider_score(external.source) if margin is not None else 0.0,
        source=external.source,
    )


def value_holdings_parallel(
    holdings: Sequence[Holding],
    prices: Dict[str, Optional[float]],
    workers: int,
    timeout: int,
) -> List[ValuationResult]:
    results: List[ValuationResult] = []
    total = len(holdings)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(value_holding, holding, prices.get(holding.ticker), timeout): holding
            for holding in holdings
        }
        for index, future in enumerate(as_completed(futures), start=1):
            holding = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                logging.warning("%s: valorisation impossible: %s", holding.ticker, exc)
                result = ValuationResult(
                    ticker=holding.ticker,
                    sector=holding.sector,
                    etf=holding.etf,
                    weight_in_etf=holding.weight_in_etf,
                    price=prices.get(holding.ticker),
                    intrinsic_value_per_share=None,
                    upside_downside_pct=None,
                    margin_of_safety=None,
                    classification="insufficient data",
                    reliability_score=0.0,
                    source="error",
                )
            results.append(result)
            logging.info("Valorisations: %d/%d", index, total)

    order = {(h.sector, h.ticker): i for i, h in enumerate(holdings)}
    results.sort(key=lambda r: order.get((r.sector, r.ticker), 10**9))
    return results


# =============================================================================
# Agrégation
# =============================================================================

def aggregate_sectors(
    results: Sequence[ValuationResult],
    sector_weights: pd.DataFrame,
) -> pd.DataFrame:
    action_df = pd.DataFrame([r.__dict__ for r in results])
    rows: List[Dict[str, Any]] = []

    for sector in GICS_SECTOR_ETFS:
        group = action_df[action_df["sector"] == sector].copy()
        if group.empty:
            continue

        selected_weight = float(group["weight_in_etf"].sum())
        normalized = (
            (group["weight_in_etf"] / selected_weight).tolist()
            if selected_weight > 0
            else group["weight_in_etf"].tolist()
        )

        weighted_upside = weighted_average(group["upside_downside_pct"].tolist(), normalized)
        weighted_margin = weighted_average(group["margin_of_safety"].tolist(), normalized)
        weighted_reliability = weighted_average(group["reliability_score"].tolist(), normalized)

        if weighted_reliability is not None:
            coverage_cap = min(max(selected_weight / 0.75, 0.0), 1.0)
            weighted_reliability = min(weighted_reliability, coverage_cap, 0.85)

        weight_row = sector_weights.loc[sector_weights["sector"] == sector, "sp500_sector_weight"]
        sp500_weight = float(weight_row.iloc[0]) if not weight_row.empty else np.nan

        rows.append(
            {
                "sector": sector,
                "etf": GICS_SECTOR_ETFS[sector],
                "sp500_sector_weight": sp500_weight,
                "weighted_upside_downside_pct": weighted_upside,
                "weighted_margin_of_safety": weighted_margin,
                "sector_classification": classify_margin(weighted_margin),
                "sector_confidence_score": weighted_reliability,
            }
        )

    return pd.DataFrame(rows)


def aggregate_sp500(sector_df: pd.DataFrame) -> Optional[float]:
    covered = sector_df.dropna(subset=["weighted_upside_downside_pct", "sp500_sector_weight"])
    if covered.empty:
        return None
    return weighted_average(
        covered["weighted_upside_downside_pct"].tolist(),
        covered["sp500_sector_weight"].tolist(),
    )


# =============================================================================
# Graphique — unique sortie persistante
# =============================================================================

def color_for_classification(classification: str) -> str:
    return {
        "undervalued": "#1f9d55",
        "fairly valued": "#607d8b",
        "overvalued": "#c0392b",
        "insufficient data": "#9e9e9e",
    }.get(classification, "#607d8b")


def make_sector_chart(
    sector_df: pd.DataFrame,
    aggregate: Optional[float],
    sector_source: DataSourceInfo,
    output_path: Path,
) -> Path:
    plot_df = sector_df.copy()
    plot_df["weighted_upside_downside_pct"] = pd.to_numeric(
        plot_df["weighted_upside_downside_pct"], errors="coerce"
    )
    plot_df = plot_df.sort_values("weighted_upside_downside_pct", ascending=True, na_position="first")

    fig, ax = plt.subplots(figsize=(15, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.axvspan(-1.0, -0.10, color="#f8d7da", alpha=0.45)
    ax.axvspan(-0.10, 0.10, color="#eceff1", alpha=0.70)
    ax.axvspan(0.10, 1.0, color="#d4edda", alpha=0.45)
    ax.axvline(0.0, color="#263238", linewidth=1.2)

    if aggregate is not None and np.isfinite(aggregate):
        ax.axvline(aggregate, color="#0d47a1", linewidth=2.0, linestyle="--")

    y_positions = np.arange(len(plot_df))
    values = plot_df["weighted_upside_downside_pct"].fillna(0.0).to_numpy(dtype=float)
    colors = [color_for_classification(c) for c in plot_df["sector_classification"]]
    bars = ax.barh(
        y_positions,
        values,
        color=colors,
        edgecolor="#263238",
        linewidth=0.7,
        alpha=0.92,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{row.sector} ({row.etf})" for row in plot_df.itertuples()], fontsize=10)

    finite_values = np.abs(plot_df["weighted_upside_downside_pct"].dropna().to_numpy(dtype=float))
    max_abs = max(0.25, (float(finite_values.max()) + 0.08) if finite_values.size else 0.25)
    if aggregate is not None and np.isfinite(aggregate):
        max_abs = max(max_abs, abs(float(aggregate)) + 0.08)
    max_abs = min(max_abs, 1.0)
    ax.set_xlim(-max_abs, max_abs)

    ax.xaxis.set_major_formatter(lambda x, _: f"{x * 100:.0f}%")
    ax.set_xlabel("Upside / downside intrinsèque estimé", fontsize=11)
    ax.set_title(
        "Valorisation intrinsèque estimée des 11 secteurs GICS du S&P 500",
        fontsize=17,
        fontweight="bold",
        loc="left",
        pad=20,
    )
    ax.text(
        0.0,
        1.015,
        f"Valeurs intrinsèques externes pondérées par poids ETF ; poids sectoriels récupérés le {sector_source.retrieved_at_utc}",
        transform=ax.transAxes,
        fontsize=10,
        color="#455a64",
        ha="left",
    )

    for bar, row in zip(bars, plot_df.itertuples()):
        value = row.weighted_upside_downside_pct
        if value is None or not np.isfinite(value):
            ax.text(0.01, bar.get_y() + bar.get_height() / 2, "données insuffisantes", va="center", ha="left", fontsize=8.5, color="#616161")
            continue

        x = value + (0.012 if value >= 0 else -0.012)
        ha = "left" if value >= 0 else "right"
        if row.weighted_margin_of_safety is not None and np.isfinite(row.weighted_margin_of_safety):
            annotation = f"{value * 100:+.1f}% | w {row.sp500_sector_weight * 100:.1f}% | MOS {row.weighted_margin_of_safety * 100:+.1f}%"
        else:
            annotation = f"{value * 100:+.1f}% | w {row.sp500_sector_weight * 100:.1f}%"
        ax.text(x, bar.get_y() + bar.get_height() / 2, annotation, va="center", ha=ha, fontsize=9)

    legend_handles = [
        Patch(facecolor="#1f9d55", edgecolor="#263238", label="Undervalued"),
        Patch(facecolor="#607d8b", edgecolor="#263238", label="Fairly valued"),
        Patch(facecolor="#c0392b", edgecolor="#263238", label="Overvalued"),
    ]
    if aggregate is not None and np.isfinite(aggregate):
        legend_handles.append(Patch(facecolor="#0d47a1", edgecolor="#0d47a1", label=f"S&P 500 aggregate: {aggregate * 100:+.1f}%"))

    ax.legend(handles=legend_handles, loc="lower right", frameon=True, fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.text(
        0.01,
        0.015,
        "Sources: State Street SPY/Select Sector SPDR, FMP si clé API disponible, Alpha Spread, ValueInvesting.io, puis Yahoo target-price en dernier recours.",
        fontsize=8.5,
        color="#455a64",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", format="png")
    plt.close(fig)
    return output_path


# =============================================================================
# Exécution
# =============================================================================

def run(top_n: int, workers: int, timeout: int, output_path: Path) -> Path:
    if load_dotenv is not None:
        load_dotenv()

    logging.info("1/4 Téléchargement simultané des holdings ETF et des poids sectoriels...")
    sector_weights, sector_source, holdings = fetch_holdings_bundle(top_n, workers, timeout)

    logging.info("2/4 Téléchargement groupé des prix de marché...")
    prices = fetch_market_prices([h.ticker for h in holdings])

    logging.info("3/4 Valorisation parallèle de %d positions...", len(holdings))
    results = value_holdings_parallel(holdings, prices, workers, timeout)

    logging.info("4/4 Agrégation sectorielle et génération du PNG...")
    sector_df = aggregate_sectors(results, sector_weights)
    aggregate = aggregate_sp500(sector_df)
    return make_sector_chart(sector_df, aggregate, sector_source, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valorisation sectorielle S&P 500 — sortie PNG unique")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Nombre de principales positions analysées par secteur (défaut: 5).")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Nombre maximal de tâches réseau parallèles (défaut: 12).")
    parser.add_argument("--timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT, help="Timeout HTTP en secondes (défaut: 12).")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE, help="Chemin du PNG final. Par défaut: même dossier que le script.")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    if args.top_n <= 0:
        raise SystemExit("--top-n doit être > 0")
    if args.workers <= 0:
        raise SystemExit("--workers doit être > 0")
    if args.timeout <= 0:
        raise SystemExit("--timeout doit être > 0")

    output = run(args.top_n, args.workers, args.timeout, args.output)
    print(f"\nFichier généré : {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())