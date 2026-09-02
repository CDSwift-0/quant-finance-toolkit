# Quant Finance Toolkit

A collection of Python research tools for market analysis, macro monitoring, valuation, sentiment and positioning, market-regime classification, and systematic long/short screening.

The repository is organized as independent modules. Most tools can be run directly as desktop applications or standalone scripts.

## Modules

| Module | Description | Main entry point |
| --- | --- | --- |
| `global_macro` | Fed Funds, CPI/core CPI, FedWatch probabilities, yield-curve and recession indicators | `taux.py` |
| `market_intelligence` | Cross-asset performance, sector structure, market breadth, volatility, put/call and CDS monitoring | `market_intelligence.py` |
| `market_structure` | S&P 500 breadth above the 200-day moving average and regression-channel extreme breadth | `lancer.py` |
| `machine_learning` | Unsupervised market-regime classification using a Gaussian mixture model and transition probabilities | `ml.py` |
| `sentiment_positioning` | AAII sentiment and CFTC institutional positioning dashboard | `SD.py` |
| `valuation` | Sector-level S&P 500 valuation workflow combining holdings, financial data and DCF-style estimates | `DCF.py` |
| `long_short` | Equity and commodity long/short screens plus a TradingView SMA-spread indicator | `mom.py`, `commo.py` |

## Installation

Python 3.10+ is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

The graphical applications use Tkinter, which is included with many Python distributions. On some Linux systems it must be installed separately through the system package manager.

## Running the tools

From the repository root:

```bash
python global_macro/taux.py
python market_intelligence/market_intelligence.py
python market_structure/lancer.py
python machine_learning/ml.py
python sentiment_positioning/SD.py
python valuation/DCF.py
python long_short/mom.py
python long_short/commo.py
```

The long/short folder also contains `tradingview_sma_spread.pine`, a Pine Script v6 indicator intended to be pasted into TradingView's Pine Editor.

## Optional valuation API key

The valuation module can use Financial Modeling Prep as one of its external valuation sources. The key is optional because the script also contains fallback data sources.

Copy `.env.example` to `.env` and add your key if you want to enable it:

```bash
cp .env.example .env
```

```text
FMP_API_KEY=your_key_here
```

Never commit the `.env` file.

## Data sources

The tools rely on public or externally hosted market and macroeconomic data, including Yahoo Finance, FRED, the U.S. Bureau of Labor Statistics, CFTC public reporting, State Street ETF holdings, AAII pages, Truflation and other public web sources used by the individual modules. Availability and schemas may change over time.

Local caches and generated outputs are intentionally excluded from version control. They are recreated when the scripts run.

## Notes

These projects are research and analytical tools. They are not intended to provide investment advice or guarantee trading performance. External data should be independently verified before being used for financial decisions.
