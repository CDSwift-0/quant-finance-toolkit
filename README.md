# Quant Finance Toolkit

A collection of quantitative-finance research tools for systematic screening, market positioning, market structure, and valuation.

The repository is organized into independent modules. Each module has its own README describing the files that are actually present in the current repository snapshot.

## Repository structure

| Module | Purpose | Current status |
| --- | --- | --- |
| `long_short` | Systematic equity and commodity long/short screening, plus a TradingView SMA-spread indicator | Active |
| `sentiment_positioning` | CFTC institutional positioning and AAII investor sentiment | Active |
| `valuation` | S&P 500 sector valuation and relative intrinsic-value analysis | Active |
| `market_structure` | S&P 500 regression-breadth analysis from rolling log-price regressions | Active |
| `global_macro` | US rates, inflation, yield-curve and recession dashboard | Active |
| `machine_learning` | Reserved for machine-learning market models | No executable script currently committed |
| `market_intelligence` | Cross-asset market dashboard with volatility, options, credit stress and sector breadth | Active |

## Installation

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows:

```powershell
.venv\Scripts\activate
```

## Main scripts

### Long / short research

```bash
python long_short/mom.py
python long_short/commo.py
```

The folder also contains `tradingview_sma_spread.pine`, a Pine Script v6 indicator intended for TradingView's Pine Editor.

### Sentiment and positioning

```bash
python sentiment_positioning/CFTC.py
python sentiment_positioning/aaii.py
```

### Market structure

```bash
python market_structure/market_breadth.py
```

The script generates `breadth_extremes.png` and `breadth_net.png` in the `market_structure` directory.

### Global macro

```bash
python global_macro/taux.py
```

The desktop dashboard tracks Fed Funds, headline and core CPI, PCE inflation, the 10Y–2Y yield-curve spread, a recession-probability series, an ex-post real-rate proxy and their 12-month dynamics. It loads cached observations immediately, then refreshes the individual series independently from public sources.

### Market intelligence

```bash
python market_intelligence/market_intelligence.py
```

The local dashboard combines cross-asset performance, VIX, SPX put/call open interest, MOVE, US sovereign CDS and a full-width S&P 500 sector-participation indicator based on distance from 50-day moving averages.

### Valuation

```bash
python valuation/DCF.py
```

The valuation script produces a single chart named `sp500_sector_dcf_chart.PNG` in the `valuation` directory.

## Optional API configuration

The valuation module can use a Financial Modeling Prep API key when available. Copy `.env.example` to `.env` and add:

```text
FMP_API_KEY=your_key_here
```

The `.env` file is excluded from version control and should never be committed.

## Data sources

Depending on the module, the repository uses externally hosted data from sources including Yahoo Finance, CFTC Public Reporting, AAII, State Street SPDR holdings, FRED, DBnomics, Financial Modeling Prep, Alpha Spread, ValueInvesting.io, Wikipedia, and other public web sources referenced by the scripts.

External APIs and web pages can change without notice, so data availability and parsing logic may require maintenance over time.

## Disclaimer

These tools are intended for research and analytical use only. They do not constitute investment advice, and externally sourced data should be independently verified before being used for financial decisions.
