import requests
import pandas as pd

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

URL = "https://www.aaii.com/sentimentsurvey/sent_results"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9,fr;q=0.8",
    "Connection": "keep-alive",
    "Referer": "https://www.aaii.com/sentimentsurvey",
}

# ------------------------------------------------------------
# Téléchargement de la page HTML
# ------------------------------------------------------------

print("Téléchargement de la page AAII…")
response = requests.get(URL, headers=HEADERS)
response.raise_for_status()  # erreur si HTTP != 200

html = response.text

# ------------------------------------------------------------
# Extraction des tableaux HTML avec pandas.read_html
# ------------------------------------------------------------

print("Extraction des tableaux HTML…")
tables = pd.read_html(html)

if not tables:
    raise RuntimeError(
        "Aucun tableau trouvé dans la page AAII. "
        "Le format du site a peut-être changé."
    )

# Chercher le tableau qui contient les colonnes Date / Bullish / Neutral / Bearish
target_df = None
for t in tables:
    cols = [str(c).strip().lower() for c in t.columns]
    if ("date" in cols
        and any("bull" in c for c in cols)
        and any("bear" in c for c in cols)):
        target_df = t.copy()
        break

if target_df is None:
    print("Tableaux trouvés mais sans colonnes typiques, on prend le premier par défaut.")
    target_df = tables[0].copy()

df = target_df

# ------------------------------------------------------------
# Nettoyage des colonnes
# ------------------------------------------------------------

# Normalisation des noms de colonnes
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace(r"[^0-9a-zA-Z_]+", "", regex=True)
)

# Conversion de la date
date_col_candidates = [c for c in df.columns if "date" in c]
if date_col_candidates:
    date_col = date_col_candidates[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
else:
    date_col = None

# Conversion des colonnes de pourcentages en numérique quand c’est possible
for col in df.columns:
    if col == date_col:
        continue
    df[col] = (
        df[col]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors="ignore")

# Tri par date si disponible
if date_col is not None:
    df = df.sort_values(by=date_col).reset_index(drop=True)

# ------------------------------------------------------------
# Affichage et sauvegarde
# ------------------------------------------------------------

print("\nColonnes détectées :")
print(df.columns.tolist())

print("\nAperçu des 10 dernières lignes :\n")
print(df.tail(10))

output_file = "aaii_sentiment_from_html.csv"
df.to_csv(output_file, index=False)
print(f"\nDonnées sauvegardées dans le fichier : {output_file}")
