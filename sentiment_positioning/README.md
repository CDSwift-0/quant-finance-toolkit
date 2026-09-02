# Sentiment & Positioning

Ce dossier regroupe deux outils Python consacrés au sentiment de marché et au positionnement des investisseurs sur les actions américaines.

## Fichiers

### CFTC.py

Télécharge directement les données publiques **CFTC Traders in Financial Futures (TFF) — Futures Only** pour l'E-mini S&P 500 et analyse le positionnement des **Asset Managers / investisseurs institutionnels**.

Le script calcule notamment :

- les positions longues, courtes et de spreading
- la position nette des Asset Managers
- la position nette en pourcentage de l'Open Interest
- une EMA sur 26 semaines
- un z-score glissant sur 52 semaines
- le percentile du positionnement sur environ 3 ans
- les variations sur 1 et 4 semaines
- la médiane historique

Il génère un seul fichier :

`institutional_net_pct_open_interest.png`

Le graphique présente l'historique du positionnement institutionnel ainsi qu'un panneau de synthèse du dernier rapport CFTC.

### aaii.py

Télécharge la page de résultats du **AAII Sentiment Survey**, détecte automatiquement le tableau contenant les données de sentiment, normalise les colonnes et convertit les dates et pourcentages lorsque cela est possible.

Le script génère :

`aaii_sentiment_from_html.csv`

Ce fichier contient les données historiques disponibles du sondage AAII, notamment les proportions d'investisseurs bullish, neutral et bearish lorsque ces colonnes sont présentes dans la source.

## Installation

Python 3.10 ou plus récent est recommandé.

```bash
pip install requests pandas matplotlib lxml
```

## Utilisation

Pour générer le graphique de positionnement CFTC :

```bash
python CFTC.py
```

Pour récupérer les données AAII :

```bash
python aaii.py
```

Les deux scripts nécessitent une connexion Internet.

## Sources

Les données proviennent des sources publiques suivantes :

- U.S. Commodity Futures Trading Commission — CFTC Public Reporting
- American Association of Individual Investors — AAII Sentiment Survey

Les structures des sources externes peuvent évoluer et nécessiter une adaptation du code si leurs API ou pages HTML changent.
