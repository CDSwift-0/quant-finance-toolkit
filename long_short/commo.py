# -*- coding: utf-8 -*-

"""
Performance 3 mois et 1 an pour un panier de commodités (futures Yahoo Finance),
puis génération d'un signal binaire:
  +1 si Perf_1an > 0 et Perf_3mois > 0
  -1 si Perf_1an < 0 et Perf_3mois < 0
   0 sinon (on ne fait rien)

Le script affiche:
  1) Un tableau récapitulatif des perfs (présentation soignée).
  2) La liste des actifs retenus (LONG = +1, SHORT = -1).
  3) Une sélection 'tendance' (ratio SMA200/SMA21): 2 meilleurs LONGs (ratio le plus bas)
     et 2 meilleurs SHORTS (ratio le plus haut).
  4) La volatilité réalisée sur 3 mois (annualisée) pour chaque ligne.
  5) Un CSV "positions_finales_YYYY-MM-DD.csv" avec les 2 longs et 2 shorts retenus.
"""

import sys
import subprocess

def _ensure_deps():
    try:
        import pandas as pd  # noqa
        import numpy as np   # noqa
        import yfinance as yf  # noqa
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy", "yfinance"])
_ensure_deps()

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from pathlib import Path

# ——————————————————————————————————————————————————————————————————————————
# Paramètres
# ————————————————————————————————————————————————————————————————————————
COMMODITIES = [
    ("Or",            "GC=F"),
    ("Argent",        "SI=F"),
    ("Cuivre",        "HG=F"),
    ("Aluminium",     "ALI=F"),
    ("Pétrole brut",  "CL=F"),   # WTI
    ("Gaz naturel",   "NG=F"),
    ("Café",          "KC=F"),
    ("Sucre",         "SB=F"),
    ("Cacao",         "CC=F"),
    ("Blé",           "ZW=F"),   # CBOT Wheat
    ("Maïs",          "ZC=F"),
    ("Soja",          "ZS=F"),
]

DOWNLOAD_PERIODS = ("5y", "3y")
TRADING_DAYS_PER_YEAR = 252

# ————————————————————————————————————————————————————————————————————————
# Utilitaires robustes
# ————————————————————————————————————————————————————————————————————————
def _as_float(x) -> float:
    """Convertit proprement en float sans FutureWarning si x est une Series de taille 1."""
    if isinstance(x, pd.Series):
        if x.size == 0:
            return np.nan
        return float(x.iloc[0])
    try:
        return float(x)
    except Exception:
        try:
            arr = np.asarray(x).reshape(-1)
            return float(arr[0]) if arr.size else np.nan
        except Exception:
            return np.nan

# ————————————————————————————————————————————————————————————————————————
# Téléchargement & calculs élémentaires
# ———————————————————————————————————————————————————————————————————————————————
def _download_close_series(ticker: str) -> pd.Series:
    for period in DOWNLOAD_PERIODS:
        df = yf.download(
            tickers=ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
        )
        if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns:
            s = df["Close"].copy()
            if getattr(s.index, "tz", None) is not None:
                s.index = s.index.tz_localize(None)
            s = s.sort_index().dropna()
            if not s.empty:
                return s
    return pd.Series(dtype=float)

def _last_price_on_or_before(series: pd.Series, when: pd.Timestamp) -> float:
    if series.empty:
        return np.nan
    if when > series.index[-1]:
        v = series.iloc[-1]
        return _as_float(v)
    idx = series.index.get_indexer([pd.to_datetime(when)], method="pad")
    if idx.size == 0 or idx[0] == -1:
        return np.nan
    v = series.iloc[idx[0]]
    return _as_float(v)

def _sma_ratio(series: pd.Series, win_fast: int = 21, win_slow: int = 200):
    """
    Retourne (sma_fast_last, sma_slow_last, ratio_slow_over_fast) au dernier point disponible.
    Si insuffisant, renvoie (nan, nan, nan).
    """
    if series.empty:
        return (np.nan, np.nan, np.nan)
    sma_fast = series.rolling(win_fast, min_periods=win_fast).mean()
    sma_slow = series.rolling(win_slow, min_periods=win_slow).mean()
    valid = pd.concat([sma_fast, sma_slow], axis=1).dropna()
    if valid.empty:
        return (np.nan, np.nan, np.nan)
    sma_f = _as_float(valid.iloc[-1, 0])
    sma_s = _as_float(valid.iloc[-1, 1])
    ratio = (sma_s / sma_f) if sma_f != 0 and np.isfinite(sma_f) and np.isfinite(sma_s) else np.nan
    return (sma_f, sma_s, ratio)

def _realized_vol_3m(series: pd.Series, today_local: pd.Timestamp) -> float:
    """
    Volatilité réalisée sur ~3 mois (annualisée), en %.
    Méthode: écart-type des rendements log quotidiens depuis (today-3mois),
    multiplié par sqrt(252) puis *100.
    """
    if series.empty:
        return np.nan
    start = today_local - pd.DateOffset(months=3)
    s = series.loc[series.index >= start]
    if s.shape[0] < 21:
        return np.nan
    rets = np.log(s / s.shift(1)).dropna()
    if rets.empty:
        return np.nan
    std_val = rets.std(ddof=1)          # peut être un scalaire ou une Series(1)
    std_scalar = _as_float(std_val)     # conversion sûre, sans FutureWarning
    vol_ann = std_scalar * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0
    return vol_ann

# ————————————————————————————————————————————————————————————————————————
# Calcul principal
# ————————————————————————————————————————————————————————————————————————
def compute_table():
    today_local = pd.Timestamp(datetime.today().date())
    d_3m = today_local - pd.DateOffset(months=3)
    d_1y = today_local - pd.DateOffset(years=1)

    rows = []
    for name, ticker in COMMODITIES:
        s = _download_close_series(ticker)
        last_date = s.index[-1] if not s.empty else pd.NaT

        if not s.empty:
            v_last = s.iloc[-1]
            p_now = _as_float(v_last)
        else:
            p_now = np.nan

        p_3m = _last_price_on_or_before(s, d_3m)
        p_1y = _last_price_on_or_before(s, d_1y)

        perf_3m = (p_now / p_3m - 1.0) if (np.isfinite(p_now) and np.isfinite(p_3m) and p_3m != 0) else np.nan
        perf_1y = (p_now / p_1y - 1.0) if (np.isfinite(p_now) and np.isfinite(p_1y) and p_1y != 0) else np.nan

        if np.isfinite(perf_3m) and np.isfinite(perf_1y):
            if perf_1y > 0 and perf_3m > 0:
                signal = 1
            elif perf_1y < 0 and perf_3m < 0:
                signal = -1
            else:
                signal = 0
        else:
            signal = 0

        sma21, sma200, ratio = _sma_ratio(s, 21, 200)
        vol3m = _realized_vol_3m(s, today_local)

        rows.append({
            "Actif": name,
            "Ticker": ticker,
            "Date dernière": "" if pd.isna(last_date) else last_date.strftime("%Y-%m-%d"),
            "Cours récent": p_now,
            "Perf 3 mois (%)": (perf_3m * 100.0) if np.isfinite(perf_3m) else np.nan,
            "Perf 1 an (%)":   (perf_1y * 100.0) if np.isfinite(perf_1y) else np.nan,
            "SMA21": sma21,
            "SMA200": sma200,
            "Ratio SMA200/SMA21": ratio,
            "Vol réalisé 3m (%)": vol3m,
            "Signal": signal,
        })

    df = pd.DataFrame(rows, columns=[
        "Actif", "Ticker", "Date dernière", "Cours récent",
        "Perf 3 mois (%)", "Perf 1 an (%)",
        "SMA21", "SMA200", "Ratio SMA200/SMA21",
        "Vol réalisé 3m (%)",
        "Signal"
    ])

    # Arrondis pour lisibilité
    for col in ["Cours récent", "SMA21", "SMA200"]:
        df[col] = df[col].round(4)
    for col in ["Perf 3 mois (%)", "Perf 1 an (%)", "Ratio SMA200/SMA21", "Vol réalisé 3m (%)"]:
        df[col] = df[col].round(2)

    return today_local, df

# ————————————————————————————————————————————————————————————————————————
# Présentation & export
# ————————————————————————————————————————————————————————————————————————
def _format_table_for_terminal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formate les colonnes présentes pour un rendu propre dans le terminal.
    Tolérant: ne touche qu'aux colonnes réellement existantes.
    """
    df = df.copy()

    def fmt_pct(v, n=2):
        return "" if pd.isna(v) else f"{v:.{n}f}%"

    def fmt_num(v, n=4):
        return "" if pd.isna(v) else f"{v:.{n}f}"

    if "Perf 3 mois (%)" in df.columns:
        df["Perf 3 mois (%)"] = df["Perf 3 mois (%)"].apply(lambda v: fmt_pct(v, 2))
    if "Perf 1 an (%)" in df.columns:
        df["Perf 1 an (%)"]   = df["Perf 1 an (%)"].apply(lambda v: fmt_pct(v, 2))
    if "Vol réalisé 3m (%)" in df.columns:
        df["Vol réalisé 3m (%)"] = df["Vol réalisé 3m (%)"].apply(lambda v: fmt_pct(v, 2))
    if "Ratio SMA200/SMA21" in df.columns:
        df["Ratio SMA200/SMA21"] = df["Ratio SMA200/SMA21"].apply(lambda v: fmt_num(v, 4))
    if "Cours récent" in df.columns:
        df["Cours récent"] = df["Cours récent"].apply(lambda v: fmt_num(v, 4))
    if "SMA21" in df.columns:
        df["SMA21"] = df["SMA21"].apply(lambda v: fmt_num(v, 4))
    if "SMA200" in df.columns:
        df["SMA200"] = df["SMA200"].apply(lambda v: fmt_num(v, 4))

    return df

def print_outputs():
    today, df = compute_table()

    title = f"Performance 3 mois / 1 an + SMA & Vol 3m – As-of {today.strftime('%Y-%m-%d')}"
    bar = "═" * len(title)
    print(bar)
    print(title)
    print(bar)

    with pd.option_context("display.width", 200, "display.max_columns", None):
        df_display = _format_table_for_terminal(df[[
            "Actif","Ticker","Date dernière","Cours récent",
            "Perf 3 mois (%)","Perf 1 an (%)",
            "SMA21","SMA200","Ratio SMA200/SMA21",
            "Vol réalisé 3m (%)","Signal"
        ]])
        print(df_display.to_string(index=False))

    longs = df[df["Signal"] == 1].reset_index(drop=True)
    shorts = df[df["Signal"] == -1].reset_index(drop=True)

    print("\n" + "—"*80)
    print("Sélection selon la règle (Signal ≠ 0)")
    print("—"*80)

    if not longs.empty:
        print("\nLONG (+1) :")
        with pd.option_context("display.width", 200):
            print(_format_table_for_terminal(longs[[
                "Actif","Ticker","Perf 3 mois (%)","Perf 1 an (%)",
                "Ratio SMA200/SMA21","Vol réalisé 3m (%)"
            ]]).to_string(index=False))
    else:
        print("\nLONG (+1) : aucun actif retenu.")

    if not shorts.empty:
        print("\nSHORT (-1) :")
        with pd.option_context("display.width", 200):
            print(_format_table_for_terminal(shorts[[
                "Actif","Ticker","Perf 3 mois (%)","Perf 1 an (%)",
                "Ratio SMA200/SMA21","Vol réalisé 3m (%)"
            ]]).to_string(index=False))
    else:
        print("\nSHORT (-1) : aucun actif retenu.")

    print("\n" + "—"*80)
    print("Sélection 'tendance' (ratio SMA200/SMA21)")
    print("—"*80)

    longs_trend  = longs.dropna(subset=["Ratio SMA200/SMA21"]).sort_values("Ratio SMA200/SMA21", ascending=True)
    shorts_trend = shorts.dropna(subset=["Ratio SMA200/SMA21"]).sort_values("Ratio SMA200/SMA21", ascending=False)

    top2_longs  = longs_trend.head(2)  if not longs_trend.empty  else longs_trend
    top2_shorts = shorts_trend.head(2) if not shorts_trend.empty else shorts_trend

    if not top2_longs.empty:
        print("\nTOP LONGS (ratio le plus bas) :")
        with pd.option_context("display.width", 200):
            print(_format_table_for_terminal(top2_longs[[
                "Actif","Ticker","Ratio SMA200/SMA21","SMA21","SMA200",
                "Perf 3 mois (%)","Perf 1 an (%)","Vol réalisé 3m (%)"
            ]]).to_string(index=False))
    else:
        print("\nTOP LONGS : aucun actif éligible.")

    if not top2_shorts.empty:
        print("\nTOP SHORTS (ratio le plus haut) :")
        with pd.option_context("display.width", 200):
            print(_format_table_for_terminal(top2_shorts[[
                "Actif","Ticker","Ratio SMA200/SMA21","SMA21","SMA200",
                "Perf 3 mois (%)","Perf 1 an (%)","Vol réalisé 3m (%)"
            ]]).to_string(index=False))
    else:
        print("\nTOP SHORTS : aucun actif éligible.")

    # ————————————————————————————————————
    # Export CSV des positions finales
    # ————————————————————————————————————
    final_rows = []
    if not top2_longs.empty:
        tmp = top2_longs.copy()
        tmp.insert(0, "Position", "LONG")
        final_rows.append(tmp)
    if not top2_shorts.empty:
        tmp = top2_shorts.copy()
        tmp.insert(0, "Position", "SHORT")
        final_rows.append(tmp)

    if final_rows:
        final_df = pd.concat(final_rows, ignore_index=True)
        final_df = final_df[[
            "Position","Actif","Ticker","Date dernière","Cours récent",
            "Perf 3 mois (%)","Perf 1 an (%)",
            "SMA21","SMA200","Ratio SMA200/SMA21","Vol réalisé 3m (%)"
        ]].copy()
        csv_path = Path.cwd() / f"positions_finales_{today.strftime('%Y-%m-%d')}.csv"
        final_df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"\nFichier CSV écrit : {csv_path}")
    else:
        print("\nAucune position finale à écrire (listes vides).")

if __name__ == "__main__":
    print_outputs()
