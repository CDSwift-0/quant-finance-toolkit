import time
from io import StringIO
from pathlib import Path

import requests
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CFTC COT — TFF Futures Only (S&P 500)
# Objectif: extraire clairement les positions institutionnelles:
# Asset Manager (long, short, net) + open interest + graphiques
# ============================================================

# Dataset Socrata (CFTC public reporting) : TFF - Futures Only
DATASET_ID = "gpe5-46if"
BASE_URL = f"https://publicreporting.cftc.gov/resource/{DATASET_ID}.csv"

# Paramètres d’analyse
START_DATE = "1995-01-01"
MARKET_LIKE = "%S&P 500%"

# Si vous voulez forcer un marché exact, mettez la string exacte ici.
# Exemple:
# TARGET_MARKET_EXACT = "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE"
TARGET_MARKET_EXACT = None

# Ordre de préférence si plusieurs marchés matchent
PREFERRED_MARKET_KEYWORDS = [
    "E-MINI S&P 500 STOCK INDEX",
    "E-MINI S&P 500",
    "MICRO E-MINI S&P 500",
    "S&P 500 STOCK INDEX",
]

# Pagination et robustesse réseau
CHUNK_SIZE = 50000
TIMEOUT = 60
MAX_RETRIES = 6

# Sorties
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def request_csv(session: requests.Session, params: dict) -> str:
    """
    Requête CSV robuste avec backoff en cas de rate-limit (429) ou erreurs transitoires.
    """
    for attempt in range(MAX_RETRIES):
        r = session.get(BASE_URL, params=params, timeout=TIMEOUT)

        if r.status_code in (429, 502, 503, 504):
            wait = 2 ** attempt
            print(f"Serveur occupé (HTTP {r.status_code}). Nouvelle tentative dans {wait}s…")
            time.sleep(wait)
            continue

        if not r.ok:
            print("\n--- Requête envoyée ---")
            print(r.url)
            print("\n--- Réponse serveur (début) ---")
            print(r.text[:2000])
            r.raise_for_status()

        return r.text

    raise RuntimeError("Échec des requêtes après plusieurs tentatives (rate limit / serveur).")


def fetch_all(session: requests.Session, where: str, order: str) -> pd.DataFrame:
    """
    Télécharge toutes les lignes correspondant au filtre 'where' via pagination ($offset).
    """
    offset = 0
    chunks = []

    while True:
        params = {
            "$where": where,
            "$order": order,
            "$limit": CHUNK_SIZE,
            "$offset": offset,
        }
        csv_text = request_csv(session, params)
        df_chunk = pd.read_csv(StringIO(csv_text))

        if df_chunk.empty:
            break

        chunks.append(df_chunk)

        if len(df_chunk) < CHUNK_SIZE:
            break

        offset += CHUNK_SIZE
        print(f"Téléchargé {offset} lignes…")

    if not chunks:
        return pd.DataFrame()

    return pd.concat(chunks, ignore_index=True)


def pick_market(markets: list[str]) -> str:
    """
    Choisit un marché parmi une liste, soit via TARGET_MARKET_EXACT,
    soit via PREFERRED_MARKET_KEYWORDS, sinon le premier.
    """
    if TARGET_MARKET_EXACT:
        for m in markets:
            if m == TARGET_MARKET_EXACT:
                return m
        raise RuntimeError(
            "TARGET_MARKET_EXACT est défini mais ne correspond à aucun marché dans la liste retournée."
        )

    lowered = [str(m).lower() for m in markets]
    for kw in PREFERRED_MARKET_KEYWORDS:
        kw_l = kw.lower()
        for i, m in enumerate(markets):
            if kw_l in lowered[i]:
                return m

    return markets[0]


def save_line_plot(dates, series_list, labels, title, y_label, out_path: Path):
    plt.figure()
    for s, lab in zip(series_list, labels):
        plt.plot(dates, s, label=lab)
    plt.title(title)
    plt.xlabel("Date (report week)")
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    session = requests.Session()
    session.headers.update({"User-Agent": "CFTC-COT-Downloader/1.0"})

    print("Téléchargement CFTC — TFF Futures Only (S&P 500)…\n")

    # Colonnes “standard” de ce dataset (celles que vous voyez dans votre traceback)
    market_col = "market_and_exchange_names"
    date_col = "report_date_as_yyyy_mm_dd"

    # Étape 1 : récupérer les marchés correspondant au filtre (diagnostic)
    markets_where = (
        f"{market_col} like '{MARKET_LIKE}' "
        f"AND {date_col} >= '{START_DATE}'"
    )
    markets_df = fetch_all(session, where=markets_where, order=f"{date_col} ASC")

    if markets_df.empty:
        print("Aucune donnée retournée. Essayez d’élargir MARKET_LIKE ou de modifier START_DATE.")
        return

    markets = (
        markets_df[market_col]
        .dropna()
        .drop_duplicates()
        .astype(str)
        .tolist()
    )

    print("Marchés correspondant à votre filtre :\n")
    for m in markets:
        print(m)

    chosen_market = pick_market(markets)
    print(f"\nMarché sélectionné : {chosen_market}\n")

    # Étape 2 : télécharger uniquement ce marché
    chosen_market_escaped = chosen_market.replace("'", "''")
    where = (
        f"{market_col} = '{chosen_market_escaped}' "
        f"AND {date_col} >= '{START_DATE}'"
    )
    df = fetch_all(session, where=where, order=f"{date_col} ASC")

    if df.empty:
        print("Le marché sélectionné n’a retourné aucune ligne (inattendu).")
        return

    # Étape 3 : colonnes institutionnelles (Asset Manager)
    # Elles existent dans votre sortie: asset_mgr_positions_long / asset_mgr_positions_short
    inst_long_col = "asset_mgr_positions_long"
    inst_short_col = "asset_mgr_positions_short"
    oi_col = "open_interest_all"

    missing = [c for c in [market_col, date_col, inst_long_col, inst_short_col] if c not in df.columns]
    if missing:
        raise RuntimeError(f"Colonnes manquantes dans le dataset: {missing}")

    # Nettoyage types
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    df[inst_long_col] = pd.to_numeric(df[inst_long_col], errors="coerce")
    df[inst_short_col] = pd.to_numeric(df[inst_short_col], errors="coerce")
    if oi_col in df.columns:
        df[oi_col] = pd.to_numeric(df[oi_col], errors="coerce")

    # Étape 4 : table “claire” (ce que vous voulez réellement)
    out = pd.DataFrame({
        "date": df[date_col],
        "market": df[market_col],
        "institutional_long": df[inst_long_col],
        "institutional_short": df[inst_short_col],
    })
    out["institutional_net"] = out["institutional_long"] - out["institutional_short"]

    if oi_col in df.columns:
        out["open_interest"] = df[oi_col]
        out["institutional_net_pct_oi"] = 100.0 * out["institutional_net"] / out["open_interest"]

    # Étape 5 : sauvegardes
    raw_path = OUT_DIR / "cot_sp500_tff_futures_only_raw.csv"
    clean_path = OUT_DIR / "cot_sp500_institutional_asset_mgr_clean.csv"
    df.to_csv(raw_path, index=False)
    out.to_csv(clean_path, index=False)

    print(f"CSV brut sauvegardé : {raw_path}")
    print(f"CSV institutionnel (clair) sauvegardé : {clean_path}\n")

    print("Aperçu (10 dernières lignes) :\n")
    print(out.tail(10).to_string(index=False))
    print("")

    # Étape 6 : graphiques
    fig1 = OUT_DIR / "institutional_long_short.png"
    save_line_plot(
        out["date"],
        [out["institutional_long"], out["institutional_short"]],
        ["Institutional long", "Institutional short"],
        "CFTC TFF — S&P 500 — Asset Manager (Long vs Short)",
        "Nombre de contrats",
        fig1
    )

    fig2 = OUT_DIR / "institutional_net.png"
    save_line_plot(
        out["date"],
        [out["institutional_net"]],
        ["Institutional net (long - short)"],
        "CFTC TFF — S&P 500 — Asset Manager Net",
        "Contrats nets",
        fig2
    )

    if "institutional_net_pct_oi" in out.columns:
        fig3 = OUT_DIR / "institutional_net_pct_open_interest.png"
        save_line_plot(
            out["date"],
            [out["institutional_net_pct_oi"]],
            ["Net % Open Interest"],
            "CFTC TFF — S&P 500 — Net Institutional (% Open Interest)",
            "% de l'open interest",
            fig3
        )
        print(f"Graphiques sauvegardés :\n{fig1}\n{fig2}\n{fig3}\n")
    else:
        print(f"Graphiques sauvegardés :\n{fig1}\n{fig2}\n")

    print("Terminé.")


if __name__ == "__main__":
    main()
