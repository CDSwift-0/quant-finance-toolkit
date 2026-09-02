import time
from io import StringIO
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import requests


# ============================================================
# CFTC COT — TFF Futures Only — E-mini S&P 500
# Sortie unique : institutional_net_pct_open_interest.png
# ============================================================

DATASET_ID = "gpe5-46if"
BASE_URL = f"https://publicreporting.cftc.gov/resource/{DATASET_ID}.csv"

# Marché principal étudié
TARGET_MARKET = "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE"
START_DATE = "2006-01-01"

# Réseau
TIMEOUT = 45
MAX_RETRIES = 5

# Lissage / contexte statistique
EMA_WEEKS = 26
ZSCORE_WEEKS = 52
PERCENTILE_WEEKS = 156  # environ 3 ans

# Sortie : directement à côté du script, sans dossier output
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = SCRIPT_DIR / "institutional_net_pct_open_interest.png"

# Colonnes CFTC utilisées uniquement
MARKET_COL = "market_and_exchange_names"
DATE_COL = "report_date_as_yyyy_mm_dd"
LONG_COL = "asset_mgr_positions_long"
SHORT_COL = "asset_mgr_positions_short"
SPREAD_COL = "asset_mgr_positions_spread"
OI_COL = "open_interest_all"

SELECT_COLUMNS = [
    MARKET_COL,
    DATE_COL,
    LONG_COL,
    SHORT_COL,
    SPREAD_COL,
    OI_COL,
]


# ------------------------------------------------------------
# Téléchargement
# ------------------------------------------------------------

def request_csv(session: requests.Session, params: dict) -> str:
    """Effectue une requête CFTC robuste avec backoff exponentiel."""
    for attempt in range(MAX_RETRIES):
        response = session.get(BASE_URL, params=params, timeout=TIMEOUT)

        if response.status_code in {429, 500, 502, 503, 504}:
            if attempt == MAX_RETRIES - 1:
                response.raise_for_status()
            wait_seconds = 2 ** attempt
            print(
                f"Serveur CFTC temporairement indisponible "
                f"(HTTP {response.status_code}) — nouvelle tentative dans {wait_seconds}s"
            )
            time.sleep(wait_seconds)
            continue

        response.raise_for_status()
        return response.text

    raise RuntimeError("Impossible de récupérer les données CFTC.")


def download_market_data(session: requests.Session) -> pd.DataFrame:
    """
    Télécharge en une requête uniquement les colonnes nécessaires pour
    l'E-mini S&P 500. Une requête hebdomadaire depuis 2006 reste très légère.
    """
    escaped_market = TARGET_MARKET.replace("'", "''")

    params = {
        "$select": ",".join(SELECT_COLUMNS),
        "$where": (
            f"{MARKET_COL} = '{escaped_market}' "
            f"AND {DATE_COL} >= '{START_DATE}'"
        ),
        "$order": f"{DATE_COL} ASC",
        "$limit": 50000,
    }

    csv_text = request_csv(session, params)
    data = pd.read_csv(StringIO(csv_text))

    if data.empty:
        raise RuntimeError(
            "Aucune donnée CFTC retournée pour le marché sélectionné. "
            "Vérifiez TARGET_MARKET ou START_DATE."
        )

    missing = [column for column in SELECT_COLUMNS if column not in data.columns]
    if missing:
        raise RuntimeError(f"Colonnes CFTC manquantes : {missing}")

    return data


# ------------------------------------------------------------
# Préparation des données
# ------------------------------------------------------------

def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les données et calcule les indicateurs utilisés dans l'image."""
    df = data.copy()

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    numeric_columns = [LONG_COL, SHORT_COL, SPREAD_COL, OI_COL]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.dropna(subset=[DATE_COL, LONG_COL, SHORT_COL, OI_COL])
        .sort_values(DATE_COL)
        .drop_duplicates(subset=[DATE_COL], keep="last")
        .reset_index(drop=True)
    )

    if df.empty:
        raise RuntimeError("Aucune ligne exploitable après nettoyage des données.")

    df = df[df[OI_COL] > 0].copy()

    df["institutional_net"] = df[LONG_COL] - df[SHORT_COL]
    df["institutional_net_pct_oi"] = 100.0 * df["institutional_net"] / df[OI_COL]

    # Un lissage modéré pour lire la tendance sans masquer la série brute
    df["net_pct_ema"] = (
        df["institutional_net_pct_oi"]
        .ewm(span=EMA_WEEKS, adjust=False, min_periods=8)
        .mean()
    )

    # Contexte statistique glissant
    rolling_mean = df["institutional_net_pct_oi"].rolling(
        ZSCORE_WEEKS, min_periods=26
    ).mean()
    rolling_std = df["institutional_net_pct_oi"].rolling(
        ZSCORE_WEEKS, min_periods=26
    ).std(ddof=0)

    df["zscore_52w"] = (
        (df["institutional_net_pct_oi"] - rolling_mean) / rolling_std.replace(0, pd.NA)
    )

    return df


# ------------------------------------------------------------
# Statistiques du dernier rapport
# ------------------------------------------------------------

def safe_change(series: pd.Series, periods: int) -> float:
    if len(series) <= periods:
        return float("nan")
    return float(series.iloc[-1] - series.iloc[-1 - periods])


def latest_percentile(series: pd.Series, window: int) -> float:
    values = series.dropna().tail(window)
    if len(values) < 2:
        return float("nan")
    return float(values.rank(pct=True, method="average").iloc[-1] * 100.0)


def build_summary(df: pd.DataFrame) -> dict:
    latest = df.iloc[-1]
    net_pct = df["institutional_net_pct_oi"]

    return {
        "date": latest[DATE_COL],
        "long": float(latest[LONG_COL]),
        "short": float(latest[SHORT_COL]),
        "spread": float(latest[SPREAD_COL]) if pd.notna(latest[SPREAD_COL]) else float("nan"),
        "net": float(latest["institutional_net"]),
        "open_interest": float(latest[OI_COL]),
        "net_pct": float(latest["institutional_net_pct_oi"]),
        "change_1w": safe_change(net_pct, 1),
        "change_4w": safe_change(net_pct, 4),
        "zscore_52w": float(latest["zscore_52w"]) if pd.notna(latest["zscore_52w"]) else float("nan"),
        "percentile_3y": latest_percentile(net_pct, PERCENTILE_WEEKS),
        "historical_median": float(net_pct.median()),
    }


# ------------------------------------------------------------
# Formatage
# ------------------------------------------------------------

def fmt_number(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:,.0f}".replace(",", " ")


def fmt_pct(value: float, decimals: int = 1, signed: bool = False) -> str:
    if pd.isna(value):
        return "—"
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{value:.{decimals}f}%"


def fmt_pp(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:+.2f} pp"


def fmt_z(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:+.2f} σ"


# ------------------------------------------------------------
# Graphique final
# ------------------------------------------------------------

def draw_summary_panel(ax, summary: dict) -> None:
    """Dessine un tableau de synthèse propre à droite du graphique."""
    ax.set_axis_off()

    ax.text(
        0.0,
        0.98,
        "Dernier rapport",
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="#18212f",
        transform=ax.transAxes,
    )

    ax.text(
        0.0,
        0.92,
        summary["date"].strftime("%d %b %Y"),
        ha="left",
        va="top",
        fontsize=10.5,
        color="#667085",
        transform=ax.transAxes,
    )

    # KPI principal
    net_color = "#087a55" if summary["net_pct"] >= 0 else "#b42318"
    ax.text(
        0.0,
        0.82,
        "Net Asset Manager / Open Interest",
        ha="left",
        va="top",
        fontsize=10,
        color="#667085",
        transform=ax.transAxes,
    )
    ax.text(
        0.0,
        0.765,
        fmt_pct(summary["net_pct"], 2, signed=True),
        ha="left",
        va="top",
        fontsize=24,
        fontweight="bold",
        color=net_color,
        transform=ax.transAxes,
    )

    rows = [
        ("Long", fmt_number(summary["long"])),
        ("Short", fmt_number(summary["short"])),
        ("Spreading", fmt_number(summary["spread"])),
        ("Net contracts", fmt_number(summary["net"])),
        ("Open interest", fmt_number(summary["open_interest"])),
        ("Variation 1 semaine", fmt_pp(summary["change_1w"])),
        ("Variation 4 semaines", fmt_pp(summary["change_4w"])),
        ("Z-score 52 semaines", fmt_z(summary["zscore_52w"])),
        ("Percentile 3 ans", fmt_pct(summary["percentile_3y"], 0)),
    ]

    y = 0.64
    row_height = 0.061

    for i, (label, value) in enumerate(rows):
        if i in {5, 7}:
            ax.plot(
                [0.0, 1.0],
                [y + 0.028, y + 0.028],
                transform=ax.transAxes,
                color="#e4e7ec",
                linewidth=0.8,
                clip_on=False,
            )

        ax.text(
            0.0,
            y,
            label,
            ha="left",
            va="center",
            fontsize=10,
            color="#667085",
            transform=ax.transAxes,
        )
        ax.text(
            1.0,
            y,
            value,
            ha="right",
            va="center",
            fontsize=10.5,
            fontweight="semibold",
            color="#18212f",
            transform=ax.transAxes,
        )
        y -= row_height

    ax.text(
        0.0,
        0.035,
        "Lecture : net = positions longues − positions courtes.\n"
        "Le spreading est affiché séparément et n'entre pas dans le net.",
        ha="left",
        va="bottom",
        fontsize=8.6,
        color="#98a2b3",
        linespacing=1.4,
        transform=ax.transAxes,
    )


def save_dashboard(df: pd.DataFrame, summary: dict) -> None:
    """Crée l'unique PNG produit par le script."""
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    grid = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[4.4, 1.35],
        left=0.065,
        right=0.965,
        top=0.82,
        bottom=0.11,
        wspace=0.075,
    )

    ax = fig.add_subplot(grid[0, 0])
    ax_summary = fig.add_subplot(grid[0, 1])

    dates = df[DATE_COL]
    net_pct = df["institutional_net_pct_oi"]
    ema = df["net_pct_ema"]

    # Zone positive / négative très discrète
    ax.fill_between(
        dates,
        net_pct,
        0,
        where=(net_pct >= 0),
        interpolate=True,
        color="#12b76a",
        alpha=0.07,
        linewidth=0,
    )
    ax.fill_between(
        dates,
        net_pct,
        0,
        where=(net_pct < 0),
        interpolate=True,
        color="#f04438",
        alpha=0.07,
        linewidth=0,
    )

    # Série principale et tendance
    ax.plot(
        dates,
        net_pct,
        color="#1d2939",
        linewidth=1.45,
        label="Net / Open Interest",
        zorder=3,
    )
    ax.plot(
        dates,
        ema,
        color="#2e90fa",
        linewidth=1.8,
        alpha=0.95,
        label=f"EMA {EMA_WEEKS} semaines",
        zorder=4,
    )

    # Repères simples
    ax.axhline(0, color="#98a2b3", linewidth=0.9, linestyle=(0, (2, 3)), zorder=1)
    ax.axhline(
        summary["historical_median"],
        color="#b7bcc5",
        linewidth=0.9,
        linestyle=(0, (6, 4)),
        zorder=1,
        label="Médiane historique",
    )

    # Dernière observation
    latest_date = summary["date"]
    latest_value = summary["net_pct"]
    latest_color = "#087a55" if latest_value >= 0 else "#b42318"

    ax.scatter(
        [latest_date],
        [latest_value],
        s=42,
        color=latest_color,
        edgecolor="white",
        linewidth=0.8,
        zorder=6,
    )
    ax.annotate(
        fmt_pct(latest_value, 1, signed=True),
        xy=(latest_date, latest_value),
        xytext=(-8, 13),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color=latest_color,
    )

    # Mise en forme axes
    ax.set_ylabel("Position nette des Asset Managers (% de l'Open Interest)", fontsize=10.5, color="#344054")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))

    span_years = max((dates.iloc[-1] - dates.iloc[0]).days / 365.25, 1)
    year_interval = 2 if span_years >= 12 else 1
    ax.xaxis.set_major_locator(mdates.YearLocator(base=year_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.grid(axis="y", color="#eaecf0", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.margins(x=0.008)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#d0d5dd")

    ax.tick_params(axis="x", colors="#667085", labelsize=9)
    ax.tick_params(axis="y", colors="#667085", labelsize=9, length=0)

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=9,
        ncols=3,
        handlelength=2.4,
        columnspacing=1.5,
    )

    draw_summary_panel(ax_summary, summary)

    # Titres au niveau figure pour garder de l'air autour du graphique
    fig.text(
        0.065,
        0.935,
        "CFTC positioning — E-mini S&P 500",
        ha="left",
        va="top",
        fontsize=23,
        fontweight="bold",
        color="#101828",
    )
    fig.text(
        0.065,
        0.892,
        "Asset Manager / Institutional — TFF Futures Only",
        ha="left",
        va="top",
        fontsize=11.5,
        color="#667085",
    )
    fig.text(
        0.965,
        0.058,
        "Source : U.S. Commodity Futures Trading Commission (CFTC)",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#98a2b3",
    )

    fig.savefig(
        OUTPUT_FILE,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


# ------------------------------------------------------------
# Programme principal
# ------------------------------------------------------------

def main() -> None:
    print("Téléchargement CFTC — E-mini S&P 500 — Asset Managers")

    with requests.Session() as session:
        session.headers.update(
            {
                "User-Agent": "CFTC-SP500-Positioning/2.0",
                "Accept": "text/csv",
            }
        )
        raw = download_market_data(session)

    df = prepare_data(raw)
    summary = build_summary(df)
    save_dashboard(df, summary)

    print(f"Marché : {TARGET_MARKET}")
    print(f"Dernier rapport : {summary['date'].date()}")
    print(f"Net Asset Manager / OI : {fmt_pct(summary['net_pct'], 2, signed=True)}")
    print(f"Image sauvegardée : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
