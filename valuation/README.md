# S&P 500 Sector Valuation

Sector-level valuation workflow for the 11 GICS sectors of the S&P 500.

## File

### `DCF.py`

The script retrieves major holdings for the Select Sector SPDR ETFs, obtains current market prices, collects externally available intrinsic-value estimates, and aggregates the results at sector and S&P 500 level.

By default, the five largest holdings in each sector ETF are analyzed in parallel.

## Methodology

The workflow uses the following sector ETFs:

- XLK — Information Technology
- XLV — Health Care
- XLF — Financials
- XLY — Consumer Discretionary
- XLC — Communication Services
- XLI — Industrials
- XLP — Consumer Staples
- XLE — Energy
- XLU — Utilities
- XLRE — Real Estate
- XLB — Materials

Holdings and sector weights are sourced from State Street SPDR data.

For each selected company, the script attempts to obtain an external intrinsic-value estimate from available providers. Supported sources include Financial Modeling Prep when an API key is configured, Alpha Spread, and ValueInvesting.io. Yahoo Finance analyst target prices are used only as a fallback proxy when no usable intrinsic-value estimate is available.

Company-level upside or downside is calculated relative to the current market price. Results are weighted by ETF holding weights and then aggregated by sector. Sector results are subsequently combined using S&P 500 sector weights.

The classification thresholds are:

- undervalued: estimated upside of at least +10%;
- fairly valued: between −10% and +10%;
- overvalued: estimated downside of at least −10%.

## Output

The script creates a single persistent output file in the same directory:

```text
sp500_sector_dcf_chart.PNG
```

No output directory, CSV, Excel, SVG, or other persistent intermediate file is created.

## Usage

Run with default settings:

```bash
python DCF.py
```

Optional parameters:

```bash
python DCF.py --top-n 5 --workers 12 --timeout 12
```

A custom output path can be supplied with `--output`.

## Optional Financial Modeling Prep API key

If you want to enable Financial Modeling Prep as a valuation source, create a repository-level `.env` file containing:

```text
FMP_API_KEY=your_key_here
```

The script can still run without this key because alternative valuation sources are implemented.

## Dependencies

The main dependencies are `pandas`, `numpy`, `requests`, `yfinance`, `matplotlib`, `openpyxl`, and optionally `python-dotenv`.

## Limitations

The output combines third-party valuation estimates rather than constructing a fully independent bottom-up DCF model for every company. Provider availability, page structure, analyst estimates, holdings data, and market prices can change over time.

The resulting chart is intended for research and comparison, not as a definitive estimate of intrinsic value or investment advice.
