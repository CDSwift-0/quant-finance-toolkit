# Sentiment & Positioning

Research tools for U.S. equity-market sentiment and institutional futures positioning.

## Files

### `CFTC.py`

Downloads public CFTC Traders in Financial Futures (TFF) Futures Only data for the E-mini S&P 500 and analyzes Asset Manager positioning.

The script uses:

- Asset Manager long positions;
- Asset Manager short positions;
- Asset Manager spreading positions;
- total open interest.

It calculates:

- net Asset Manager positioning as long minus short;
- net positioning as a percentage of open interest;
- a 26-week exponential moving average;
- one-week and four-week changes;
- a rolling 52-week z-score;
- a three-year percentile;
- the historical median.

The script produces one file in the same directory:

```text
institutional_net_pct_open_interest.png
```

The chart combines the historical net-positioning series with a compact summary of the latest CFTC report.

Run:

```bash
python CFTC.py
```

### `aaii.py`

Downloads the AAII Sentiment Survey results page and extracts the HTML table containing investor-sentiment data.

The script automatically searches for a table containing date, bullish, and bearish fields, normalizes column names, converts dates, and converts percentage-like values to numeric form when possible.

The cleaned dataset is exported as:

```text
aaii_sentiment_from_html.csv
```

Run:

```bash
python aaii.py
```

## Dependencies

The scripts use `requests`, `pandas`, `matplotlib`, and an HTML parser supported by `pandas.read_html` such as `lxml`.

## Data sources

- U.S. Commodity Futures Trading Commission — CFTC Public Reporting
- American Association of Individual Investors — AAII Sentiment Survey

Both scripts depend on external data structures that may change over time.
