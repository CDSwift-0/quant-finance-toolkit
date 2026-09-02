# Long / Short Research

Systematic research tools for equity and commodity long/short selection, together with a TradingView indicator based on the SMA21/SMA200 spread.

## Files

### `mom.py`

Equity long/short screen built around the relationship between the 21-day and 200-day simple moving averages.

The script downloads up to ten years of adjusted daily prices from Yahoo Finance for a broad U.S. equity universe. It calculates:

- SMA21 and SMA200;
- the percentage spread between SMA21 and SMA200;
- the SMA200/SMA21 ratio;
- three-month price performance;
- short- and medium-window slopes of the spread;
- historical spread volatility over five years when available, with a three-year fallback;
- the current spread expressed as a multiple of its historical standard deviation.

The strategy first identifies the most extreme bullish and bearish SMA ratios. It then ranks candidates by three-month performance and uses the recent and main spread slopes only as confirmation filters.

By default, it targets three long and three short positions with a 50% gross allocation on each side.

Run:

```bash
python mom.py
```

Force a fresh Yahoo Finance download:

```bash
python mom.py --refresh
```

Useful parameters include `--top`, `--extremes`, `--pente`, `--confirmation`, and `--pente-min`.

Results are written to the local `Resultats/` directory as dated CSV files, together with current-candidate and current-position files.

### `commo.py`

Commodity trend screen using Yahoo Finance futures data.

The script evaluates a basket including gold, silver, copper, aluminium, WTI crude oil, natural gas, coffee, sugar, cocoa, wheat, corn, and soybeans.

A directional signal is generated from three-month and one-year performance:

- long when both periods are positive;
- short when both periods are negative;
- neutral otherwise.

Eligible assets are then ranked using the SMA200/SMA21 ratio. The script selects up to two longs and two shorts and reports annualized realized volatility over approximately three months.

The final selection is exported as:

```text
positions_finales_YYYY-MM-DD.csv
```

Run:

```bash
python commo.py
```

### `tradingview_sma_spread.pine`

Pine Script v6 indicator for TradingView.

It transforms the SMA21/SMA200 spread with a zero-lag filter, estimates short- and medium-term structure with linear regressions, and builds bullish and bearish reversal scores from normalized slope, acceleration, jerk, slope divergence, and distance from recent extremes.

The indicator also includes:

- a 10-day moving average of the filtered spread;
- a dotted zero line;
- a five-year reference distribution;
- optional lower −1.5σ and −2σ reference levels;
- a dynamic green/red/neutral color gradient based on directional bias and structural alignment.

Paste the file into TradingView's Pine Editor to use it.

## Dependencies

The Python scripts require `pandas`, `numpy`, and `yfinance`. The repository-level `requirements.txt` contains the shared environment dependencies.

## Notes

These screens are research tools. Their rankings and signals are mechanical outputs of the implemented rules and should not be interpreted as standalone investment recommendations.
