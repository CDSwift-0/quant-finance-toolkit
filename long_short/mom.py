#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stratégie Long / Short SMA200/SMA21 avec confirmation secondaire de pente.

La stratégie principale choisit d'abord les ratios SMA200/SMA21 les plus bas
et les plus hauts, puis retient les meilleures ou les pires performances à
trois mois. La pente du spread ne classe aucune action : elle confirme ou
refuse seulement les candidats déjà choisis par la stratégie principale. En
cas de refus, le candidat suivant dans l'ordre de base est examiné.

Lancement :
    python3 mom.py
    python3 mom.py --refresh
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf


SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_FILE = SCRIPT_DIR / ".prix_spread_pente_10y.pkl"
OUTPUT_DIR = SCRIPT_DIR / "Resultats"


@dataclass(frozen=True)
class StrategyConfig:
    sma_short: int = 21
    sma_long: int = 200
    slope_window: int = 20
    confirmation_window: int = 5
    min_slope: float = 0.0
    performance_months: int = 3
    sigma_5y_bars: int = 252 * 5
    sigma_3y_bars: int = 252 * 3
    extreme_pool_size: int = 10
    positions_per_side: int = 3
    long_share: float = 0.50
    short_share: float = 0.50
    batch_size: int = 75


UNIVERSE_TEXT = """
NVDA MSFT AAPL AMZN META AVGO GOOGL GOOG TSLA BRK.B WMT ORCL JPM LLY V NFLX MA XOM JNJ COST
PLTR ABBV HD BAC AMD PG UNH GE CVX KO WFC CSCO IBM MS TMUS CAT PM GS AXP CRM ABT MCD MU LIN
RTX MRK PEP APP DIS TMO UBER NOW T BLK C INTU LRCX ANET AMAT INTC NEE QCOM SCHW VZ GEV BKNG
BA TJX AMGN TXN ISRG APH ACN DHR ETN SPGI GILD BSX KLAC SYK PFE ADBE PANW COF LOW UNP PGR HON
BX CEG CRWD MDT DE HOOD DASH ADI LMT ADP WELL PLD KKR CB SO MO CMCSA COP VRTX DELL CVS NEM NKE
DUK MMC HCA MCK CME TT SBUX PH ICE GD AMT BMY CDNS NOC ORLY COIN MCO WM RCL SHW MMM SNPS EQIX
MDLZ CI ELV WMB ECL HWM AON AJG BK ABNB CTAS GLW MSI EMR APO MAR USB ITW JCI PNC UPS VST TDG
RSG CSX AZO MNST FI TEL PWR URI NSC PYPL ADSK FTNT AEP ZTS CL HLT WDAY COR KMI REGN TRV SRE FCX
DLR EOG AFL SPG CMI APD CMG MPC TFC FDX DDOG GM NXPI MET O ROP LHX BDX PSA ALL MMM AOS AES AFL
A APD ABNB AKAM ALB ARE ALGN ALLE LNT ALL MO AMCR AEE AEP AIG AMT AWK AMP AME APH AON APA APO
AMAT APTV ACGL ADM ANET AJG AIZ ATO ADSK ADP AZO AVB AVY AXON BKR BALL BAX BDX BBY TECH BIIB
BX BK BA BKNG BSX BMY BR BRO BF.B BLDR BG BXP CHRW CDNS CPT CPB COF CAH KMX CCL CARR CAT CBOE
CBRE CDW COR CNC CNP CF CRL SCHW CHTR CVX CMG CB CHD CI CINF CTAS CSCO C CFG CLX CME CMS CTS KO
CTSH CL CMCSA CAG COP ED STZ CEG COO CPRT CPAY CTVA CSGP COST CTRA CRWD CCI CSX CMI CVS DHR DRI
DVA DAY DECK DE DELL DAL DVN DXCM FANG DLR DG DLTR D DPZ DOV DOW DHI DTE DUK DD EMN ETN EBAY
ECL EIX EW EA ELV EME EMR ETR EOG EPAM EQT EFX EQIX EQR ERIE ESS EL EG EVRG ES EXC EXPE EXPD
EXR XOM FFIV FDS FICO FAST FRT FDX FIS FITB FSLR FE F FTNT FTV FOXA FOX BEN FCX GRMN IT GE GEHC
GEN GNRC GD GIS GM GPC GPN GL GDDY GS HAL HIG HAS HCA DOC HSIC HSY HPE HLT HOLX HD HON HRL HST
HWM HPQ HUBB HUM HBAN HII IBM IEX IDXX ITW INCY IR PODD IBKR ICE IFF IP IPG INTU ISRG IVZ INVH
IQV IRM JBHT JBL JKHY J JNJ JCI JPM K KVUE KDP KEY KEYS KMB KIM KMI KKR KLAC KHC KR LHX LH LW
LVS LDOS LEN LII LLY LIN LYV LKQ LMT L LOW LULU LYB MTB MAR MMC MLM MAS MA MTCH MKC MCD MCK MDT
MRK MET MTD MGM MCHP MSFT MAA MRNA MHK MOH TAP MDLZ MPWR MNST MCO MS MOS MSI MSCI NDAQ NTAP NFLX
NWSA NWS NEE NI NDSN NTRS NOC NCLH NRG NUE NVDA NVR ORLY OXY ODFL OMC ON OKE ORCL OTIS PCAR PKG
PANW PSKY PH PAYX PAYC PNR PEP PFE PCG PM PSX PNW PNC POOL PPG PPL PFG PG PLD PRU PEG PTC PSA
PHM PWR QCOM DGX RL RJF RTX O REG REGN RF RSG RMD RVTY ROK ROL ROP ROST SPGI CRM SBAC SLB STX SRE
NOW SPG SWKS SJM SW SNA SOLV SO LUV SWK SBUX STT STLD STE SYK SMCI SYF SNPS SYY TMUS TROW TTWO
TPR TRGP TGT TEL TDY TER TSLA TXN TPL TXT TMO TJX TKO TTD TSCO TT TDG TRV TRMB TFC TYL TSN USB
UDR ULTA UNP UAL UPS URI UNH UHS VLO VTR VLTO VRSN VRSK VZ VRTX VTRS VICI V VST VMC WRB GWW WAB
WMT DIS WBD WM WAT WEC WFC WELL WST WDC WY WSM WMB WTW WDAY WYNN XEL XYL YUM ZBRA ZBH ZTS
"""

UNIVERSE = list(dict.fromkeys(UNIVERSE_TEXT.split()))


def yahoo_ticker(ticker: str) -> str:
    return ticker.replace(".", "-")


YAHOO_TO_OFFICIAL = {yahoo_ticker(ticker): ticker for ticker in UNIVERSE}


def configure_yfinance_cache() -> None:
    cache_dir = SCRIPT_DIR / ".yfinance-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        if hasattr(yf, "cache") and hasattr(yf.cache, "set_cache_location"):
            yf.cache.set_cache_location(str(cache_dir))
        elif hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(str(cache_dir))
    except Exception:
        pass


def _extract_close(raw: pd.DataFrame, requested: list[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        for level in range(raw.columns.nlevels):
            values = raw.columns.get_level_values(level)
            if "Close" in values:
                close = raw.xs("Close", axis=1, level=level, drop_level=True)
                break
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
    return close.apply(pd.to_numeric, errors="coerce")


def download_closes(tickers: Iterable[str], batch_size: int) -> pd.DataFrame:
    yahoo_symbols = [yahoo_ticker(ticker) for ticker in tickers]
    frames: list[pd.DataFrame] = []

    for start in range(0, len(yahoo_symbols), batch_size):
        batch = yahoo_symbols[start : start + batch_size]
        print(f"Téléchargement {start + 1}-{min(start + len(batch), len(yahoo_symbols))}/{len(yahoo_symbols)}...")
        try:
            raw = yf.download(
                batch,
                period="10y",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="column",
            )
        except Exception as exc:
            print(f"Paquet ignoré après une erreur de téléchargement : {exc}")
            continue

        close = _extract_close(raw, batch)
        if not close.empty:
            frames.append(close)

    if not frames:
        raise RuntimeError("Aucun cours n'a pu être téléchargé.")

    closes = pd.concat(frames, axis=1)
    closes = closes.loc[:, ~closes.columns.duplicated()]

    missing = [
        symbol
        for symbol in yahoo_symbols
        if symbol not in closes.columns or not closes[symbol].notna().any()
    ]
    if missing:
        print(f"Nouvelle tentative séquentielle pour {len(missing)} ticker(s) manquant(s)...")
        retry_frames: list[pd.DataFrame] = []
        for symbol in missing:
            try:
                raw = yf.download(
                    symbol,
                    period="10y",
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                    group_by="column",
                )
            except Exception:
                continue
            close = _extract_close(raw, [symbol])
            if not close.empty and close[symbol].notna().any():
                retry_frames.append(close)
        if retry_frames:
            closes = closes.drop(columns=missing, errors="ignore")
            closes = pd.concat([closes, *retry_frames], axis=1)
            closes = closes.loc[:, ~closes.columns.duplicated(keep="last")]

    closes = closes.rename(columns=YAHOO_TO_OFFICIAL)
    selected = [ticker for ticker in UNIVERSE if ticker in closes.columns]
    return closes[selected].sort_index().dropna(how="all")


def load_closes(refresh: bool, config: StrategyConfig) -> pd.DataFrame:
    if not refresh and CACHE_FILE.exists():
        cache_date = pd.Timestamp.fromtimestamp(CACHE_FILE.stat().st_mtime).date()
        if cache_date == pd.Timestamp.today().date():
            try:
                cached = pd.read_pickle(CACHE_FILE)
                if isinstance(cached, pd.DataFrame) and not cached.empty:
                    print("Utilisation du cache de cours créé aujourd'hui.")
                    return cached
            except Exception:
                pass

    closes = download_closes(UNIVERSE, config.batch_size)
    try:
        closes.to_pickle(CACHE_FILE)
    except Exception as exc:
        print(f"Le cache n'a pas pu être enregistré : {exc}")
    return closes


def linear_slope(series: pd.Series, window: int) -> float:
    values = series.dropna().tail(window).to_numpy(dtype=float)
    if len(values) < window or not np.isfinite(values).all():
        return math.nan
    x = np.arange(window, dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def performance_over_months(series: pd.Series, months: int) -> float:
    prices = series.dropna().sort_index()
    if prices.empty:
        return math.nan
    cutoff = prices.index[-1] - pd.DateOffset(months=months)
    start_prices = prices.loc[:cutoff]
    if start_prices.empty:
        return math.nan
    start_price = float(start_prices.iloc[-1])
    end_price = float(prices.iloc[-1])
    if start_price == 0 or not math.isfinite(start_price) or not math.isfinite(end_price):
        return math.nan
    return (end_price / start_price - 1.0) * 100.0


def sigma_reference(spread: pd.Series, config: StrategyConfig) -> tuple[float, str]:
    valid = spread.dropna()
    if len(valid) >= config.sigma_5y_bars:
        return float(valid.tail(config.sigma_5y_bars).std(ddof=0)), "5 ans"
    if len(valid) >= config.sigma_3y_bars:
        return float(valid.tail(config.sigma_3y_bars).std(ddof=0)), "3 ans"
    return math.nan, "insuffisant"


def extension_label(spread_value: float, sigma: float) -> str:
    if not math.isfinite(sigma) or sigma <= 0:
        return "Sigma indisponible"
    multiple = spread_value / sigma
    if multiple >= 2.0:
        return "Haussière > +2σ"
    if multiple <= -2.0:
        return "Baissière < -2σ"
    return "Dans ±2σ"


def compute_snapshot(closes: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for ticker in UNIVERSE:
        if ticker not in closes.columns:
            continue
        prices = closes[ticker].dropna().sort_index()
        if prices.empty:
            continue

        sma21 = prices.rolling(config.sma_short, min_periods=config.sma_short).mean()
        sma200 = prices.rolling(config.sma_long, min_periods=config.sma_long).mean()
        spread = ((sma21 / sma200) - 1.0) * 100.0
        valid_spread = spread.dropna()
        if valid_spread.empty:
            continue

        as_of = valid_spread.index[-1]
        spread_value = float(valid_spread.iloc[-1])
        sigma, sigma_period = sigma_reference(valid_spread, config)
        sigma_multiple = spread_value / sigma if math.isfinite(sigma) and sigma > 0 else math.nan

        rows.append(
            {
                "Ticker": ticker,
                "Date": as_of.date().isoformat(),
                "Cours": float(prices.loc[:as_of].iloc[-1]),
                "SMA21": float(sma21.loc[as_of]),
                "SMA200": float(sma200.loc[as_of]),
                "Spread_%": spread_value,
                "Ratio_200_21": float(sma200.loc[as_of] / sma21.loc[as_of]),
                "Pente_confirmation": linear_slope(valid_spread, config.confirmation_window),
                "Pente_principale": linear_slope(valid_spread, config.slope_window),
                "Perf_3M_%": performance_over_months(prices.loc[:as_of], config.performance_months),
                "Sigma_ref": sigma,
                "Periode_sigma": sigma_period,
                "Multiple_sigma": sigma_multiple,
                "Extension": extension_label(spread_value, sigma),
            }
        )

    snapshot = pd.DataFrame(rows)
    if snapshot.empty:
        raise RuntimeError("Aucun indicateur n'a pu être calculé.")
    return snapshot


def confirm_candidate(row: pd.Series, position: str, config: StrategyConfig) -> tuple[str, str]:
    if pd.isna(row["Pente_confirmation"]) or pd.isna(row["Pente_principale"]):
        return "NON CONFIRMÉ", "Historique de pente insuffisant"

    recent_slope = float(row["Pente_confirmation"])
    main_slope = float(row["Pente_principale"])
    threshold = config.min_slope

    if position == "LONG":
        if recent_slope <= threshold:
            return "NON CONFIRMÉ", "Le spread haussier se retourne récemment"
        if main_slope <= threshold:
            return "NON CONFIRMÉ", "La pente de fond du spread n'est pas positive"
        return "CONFIRMÉ", "Les pentes récente et principale sont positives"

    if recent_slope >= -threshold:
        return "NON CONFIRMÉ", "Le spread baissier remonte récemment"
    if main_slope >= -threshold:
        return "NON CONFIRMÉ", "La pente de fond du spread n'est pas négative"
    return "CONFIRMÉ", "Les pentes récente et principale sont négatives"


def build_strategy_candidates(
    snapshot: pd.DataFrame, config: StrategyConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = snapshot.copy()
    result["Groupe_extreme"] = "HORS TOP 10"
    result["Rang_extreme"] = pd.NA
    result["Selection_strategie"] = ""
    result["Confirmation_indicateur"] = ""
    result["Motif_confirmation"] = ""
    result["Statut_final"] = ""

    valid = result.dropna(subset=["Ratio_200_21", "Perf_3M_%"])
    bullish_pool = (
        valid[valid["Ratio_200_21"] < 1.0]
        .nsmallest(config.extreme_pool_size, "Ratio_200_21")
        .copy()
    )
    bearish_pool = (
        valid[valid["Ratio_200_21"] > 1.0]
        .nlargest(config.extreme_pool_size, "Ratio_200_21")
        .copy()
    )

    bullish_pool["Rang_extreme"] = range(1, len(bullish_pool) + 1)
    bearish_pool["Rang_extreme"] = range(1, len(bearish_pool) + 1)

    result.loc[bullish_pool.index, "Groupe_extreme"] = "10 PLUS HAUSSIÈRES"
    result.loc[bearish_pool.index, "Groupe_extreme"] = "10 PLUS BAISSIÈRES"
    result.loc[bullish_pool.index, "Rang_extreme"] = bullish_pool["Rang_extreme"]
    result.loc[bearish_pool.index, "Rang_extreme"] = bearish_pool["Rang_extreme"]

    def scan_until_confirmed(pool: pd.DataFrame, position: str, ascending: bool) -> pd.DataFrame:
        ordered = pool.sort_values("Perf_3M_%", ascending=ascending).copy()
        ordered["Position"] = position
        ordered["Rang_performance"] = range(1, len(ordered) + 1)
        examined: list[pd.Series] = []
        confirmed_count = 0

        for index, row in ordered.iterrows():
            status, reason = confirm_candidate(row, position, config)
            candidate = row.copy()
            candidate["Groupe_extreme"] = (
                "10 PLUS HAUSSIÈRES" if position == "LONG" else "10 PLUS BAISSIÈRES"
            )
            candidate["Selection_strategie"] = position
            candidate["Confirmation_indicateur"] = status
            candidate["Motif_confirmation"] = reason
            candidate["Statut_final"] = "RETENU" if status == "CONFIRMÉ" else "ÉCARTÉ ET REMPLACÉ"
            examined.append(candidate)

            result.at[index, "Selection_strategie"] = position
            result.at[index, "Confirmation_indicateur"] = status
            result.at[index, "Motif_confirmation"] = reason
            result.at[index, "Statut_final"] = candidate["Statut_final"]

            if status == "CONFIRMÉ":
                confirmed_count += 1
                if confirmed_count >= config.positions_per_side:
                    break

        return pd.DataFrame(examined)

    long_candidates = scan_until_confirmed(bullish_pool, "LONG", ascending=False)
    short_candidates = scan_until_confirmed(bearish_pool, "SHORT", ascending=True)
    candidates = pd.concat([long_candidates, short_candidates], ignore_index=False)
    if candidates.empty:
        return result, candidates

    return result, candidates.reset_index(drop=True)


def select_positions(candidates: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()

    positions = candidates[candidates["Confirmation_indicateur"] == "CONFIRMÉ"].copy()
    longs = positions[positions["Position"] == "LONG"]
    shorts = positions[positions["Position"] == "SHORT"]
    if not longs.empty:
        positions.loc[longs.index, "Allocation_%"] = config.long_share * 100.0 / len(longs)
    if not shorts.empty:
        positions.loc[shorts.index, "Allocation_%"] = -config.short_share * 100.0 / len(shorts)

    columns = [
        "Position",
        "Ticker",
        "Date",
        "Allocation_%",
        "Rang_extreme",
        "Rang_performance",
        "Cours",
        "Ratio_200_21",
        "Spread_%",
        "Pente_confirmation",
        "Pente_principale",
        "Perf_3M_%",
        "Confirmation_indicateur",
        "Statut_final",
        "Motif_confirmation",
        "Multiple_sigma",
        "Extension",
    ]
    return positions[columns] if not positions.empty else pd.DataFrame(columns=columns)


def export_results(
    snapshot: pd.DataFrame, candidates: pd.DataFrame, positions: pd.DataFrame
) -> tuple[Path, Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    as_of = snapshot["Date"].dropna().max()
    universe_path = OUTPUT_DIR / f"univers_analyse_{as_of}.csv"
    candidates_path = OUTPUT_DIR / f"candidats_strategie_{as_of}.csv"
    positions_path = OUTPUT_DIR / f"positions_{as_of}.csv"
    current_candidates_path = OUTPUT_DIR / "candidats_actuels.csv"
    latest_path = OUTPUT_DIR / "positions_actuelles.csv"

    snapshot_export = snapshot.sort_values(["Groupe_extreme", "Rang_extreme", "Ticker"]).copy()
    candidates_export = candidates.copy()
    positions_export = positions.copy()

    for frame in (snapshot_export, candidates_export, positions_export):
        numeric_columns = frame.select_dtypes(include="number").columns
        frame[numeric_columns] = frame[numeric_columns].round(4)

    snapshot_export.to_csv(universe_path, index=False, encoding="utf-8-sig")
    candidates_export.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    candidates_export.to_csv(current_candidates_path, index=False, encoding="utf-8-sig")
    positions_export.to_csv(positions_path, index=False, encoding="utf-8-sig")
    positions_export.to_csv(latest_path, index=False, encoding="utf-8-sig")
    return universe_path, candidates_path, positions_path


def print_results(
    snapshot: pd.DataFrame, candidates: pd.DataFrame, positions: pd.DataFrame, config: StrategyConfig
) -> None:
    confirmed = int((candidates["Confirmation_indicateur"] == "CONFIRMÉ").sum()) if not candidates.empty else 0
    rejected = len(candidates) - confirmed
    confirmed_word = "confirmé" if confirmed == 1 else "confirmés"
    rejected_word = "non confirmé" if rejected == 1 else "non confirmés"

    print("\nSTRATÉGIE SMA200/SMA21 + CONFIRMATION DE PENTE")
    print(
        f"{len(snapshot)} actions analysées | top {config.extreme_pool_size} haussier et baissier | "
        f"{len(candidates)} candidats examinés dans l'ordre de base | {confirmed} {confirmed_word} | "
        f"{rejected} {rejected_word}"
    )
    print(
        "Décision principale : ratio SMA200/SMA21 extrême, puis performance 3 mois. "
        f"Confirmation seulement : pente sur {config.confirmation_window} et {config.slope_window} séances."
    )

    if candidates.empty:
        print("\nAucun candidat n'a pu être sélectionné par la stratégie principale.")
        return

    candidate_columns = [
        "Position",
        "Ticker",
        "Rang_extreme",
        "Rang_performance",
        "Ratio_200_21",
        "Perf_3M_%",
        "Pente_confirmation",
        "Pente_principale",
        "Confirmation_indicateur",
        "Statut_final",
        "Motif_confirmation",
    ]
    candidate_display = candidates[candidate_columns].copy()
    numeric_columns = [
        "Ratio_200_21",
        "Pente_confirmation",
        "Pente_principale",
        "Perf_3M_%",
    ]
    candidate_display[numeric_columns] = candidate_display[numeric_columns].round(3)
    print("\nPARCOURS DES CANDIDATS DE LA STRATÉGIE PRINCIPALE")
    print(candidate_display.to_string(index=False))

    if positions.empty:
        print("\nAucun des candidats n'est confirmé par la pente aujourd'hui.")
    else:
        position_display = positions.copy()
        position_numeric = position_display.select_dtypes(include="number").columns
        position_display[position_numeric] = position_display[position_numeric].round(2)
        print("\nPOSITIONS CONFIRMÉES")
        print(position_display.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stratégie Long/Short SMA200/SMA21 avec confirmation secondaire de pente."
    )
    parser.add_argument("--refresh", action="store_true", help="Ignore le cache du jour et retélécharge les cours.")
    parser.add_argument("--top", type=int, default=3, help="Nombre de positions par côté, 3 par défaut.")
    parser.add_argument(
        "--extremes",
        type=int,
        default=10,
        help="Nombre de ratios SMA200/SMA21 extrêmes étudiés de chaque côté, 10 par défaut.",
    )
    parser.add_argument("--pente", type=int, default=20, help="Fenêtre de pente principale en séances, 20 par défaut.")
    parser.add_argument("--confirmation", type=int, default=5, help="Fenêtre de confirmation récente, 5 par défaut.")
    parser.add_argument(
        "--pente-min",
        type=float,
        default=0.0,
        help="Pente absolue minimale en point de pourcentage par séance.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.top < 1
        or args.extremes < args.top
        or args.pente < 2
        or args.confirmation < 2
        or args.pente_min < 0
    ):
        raise SystemExit(
            "Les paramètres doivent être positifs et --extremes doit être supérieur ou égal à --top."
        )

    config = StrategyConfig(
        extreme_pool_size=args.extremes,
        positions_per_side=args.top,
        slope_window=args.pente,
        confirmation_window=args.confirmation,
        min_slope=args.pente_min,
    )
    configure_yfinance_cache()
    closes = load_closes(args.refresh, config)
    snapshot = compute_snapshot(closes, config)
    snapshot, candidates = build_strategy_candidates(snapshot, config)
    positions = select_positions(candidates, config)
    universe_path, candidates_path, positions_path = export_results(snapshot, candidates, positions)
    print_results(snapshot, candidates, positions, config)
    print(f"\nAnalyse complète : {universe_path}")
    print(f"Candidats de la stratégie : {candidates_path}")
    print(f"Positions retenues : {positions_path}")


if __name__ == "__main__":
    main()
