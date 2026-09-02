# Market Intelligence

Interactive cross-asset market-monitoring dashboard built with Python and Tkinter.

The module is contained in a single executable file:

```text
market_intelligence.py
```

It combines equity performance, volatility, options positioning, rates, credit stress, cross-asset confirmation and sector participation in one local dashboard. The selected horizon can be changed directly in the interface from 3 months to 10 years.

## What the dashboard monitors

The dashboard follows several complementary layers of market information rather than relying on a single indicator.

### Top-level KPIs

The header displays:

- S&P 500 performance;
- Nasdaq 100 performance;
- Russell 2000 / small-cap performance;
- VIX level;
- SPX put/call open-interest ratio;
- US 10-year yield;
- sector-participation score.

Performance figures use the horizon selected in the interface. The volatility, rate and participation readings use the latest available observation.

### Market regime

The **Lecture de marché** panel combines trend, volatility, participation and options information into a compact regime label.

The script distinguishes between conditions such as:

- `Risk-on discipliné`;
- `Risque élevé`;
- `Marché sélectif`;
- `Rotation défensive`;
- `Équilibre à confirmer`.

The panel also checks whether SPY is above or below its 20-day and 50-day moving averages, identifies the strongest and weakest sector over the selected horizon, and compares cyclical sector performance with defensive sector performance.

### Market map

The **Carte de marché** groups the monitored instruments into indices, sectors and macro assets.

It shows the performance of major equity indices, the eleven S&P 500 sector ETFs and macro instruments such as long-duration Treasuries, gold, the US dollar and oil. Positive and negative performance are encoded visually for rapid comparison.

### Cross-asset confirmation

The **Cross-asset** panel compares the selected-period performance of:

- SPY;
- QQQ;
- IWM;
- TLT;
- GLD;
- UUP;
- USO.

It also calculates relative spreads such as `SPY - TLT`, `QQQ - IWM` and `GLD - UUP`, and displays correlations with the S&P 500.

This section is intended to determine whether the equity-market signal is being confirmed by bonds, commodities, the dollar and small-cap participation.

### VIX / S&P 500

The **VIX / S&P 500** chart plots equity volatility against the S&P 500.

Reference levels at 18, 25 and 35 are displayed, with visual zones used to distinguish relatively calm conditions from elevated volatility. The S&P 500 is plotted on a separate scale so that volatility changes can be read against market direction.

### SPX Put / Call Open Interest

The **SPX Put / Call OI** chart uses SPX option open interest rather than trading volume.

The ratio is calculated as:

```text
Put/Call OI = Put Open Interest / Call Open Interest
```

The dashboard also plots a 20-session moving average and reference levels around 1.45, 1.80 and 2.15.

The current reading is placed in historical context using its percentile in the available series. High readings are interpreted as stronger structural demand for downside protection, while low readings indicate relatively greater call exposure.

### XLY / XLP

The **XLY / XLP** ratio compares Consumer Discretionary with Consumer Staples.

A rising ratio generally indicates stronger cyclical risk appetite, while a falling ratio indicates relative strength in a more defensive sector. The S&P 500 is displayed alongside the ratio for confirmation.

### MOVE Index

The **MOVE Index** panel tracks volatility in the US Treasury market and compares it with the S&P 500.

The chart includes a 20-session moving average and reference levels at 80, 100 and 120. Elevated MOVE readings can indicate stress or instability in rates even when equity volatility remains contained.

### US 5-year CDS

The **CDS US 5 ans** panel tracks the available five-year US sovereign CDS series and compares it with the S&P 500.

This provides a separate credit-stress perspective alongside equity volatility and rates volatility.

---

## Sector participation — the main breadth chart

The final **Participation sectorielle** chart is deliberately larger than the other panels and spans the full dashboard width. It is the central breadth indicator of the module.

Its purpose is not simply to ask whether the S&P 500 is rising or falling. It asks a more important market-structure question:

> How broadly is the move being supported across the eleven S&P 500 sectors?

The calculation uses the eleven sector ETFs:

```text
XLK  Technology
XLC  Communication
XLY  Consumer Discretionary
XLF  Financials
XLI  Industrials
XLV  Health Care
XLE  Energy
XLP  Consumer Staples
XLU  Utilities
XLB  Materials
XLRE Real Estate
```

### 1. Distance from the 50-day moving average

For every sector, the script calculates:

```text
Distance = (Price / MM50 - 1) × 100
```

A positive value means the sector trades above its 50-day moving average. A negative value means it trades below it.

This distance contains more information than a simple binary above/below test because it measures the magnitude of the move around the trend.

### 2. Raw participation

The chart also computes the percentage of valid sectors currently above their 50-day moving average:

```text
Raw participation =
number of sectors above MM50
──────────────────────────── × 100
number of valid sectors
```

This appears as **% secteurs > MM50** and as the current **BRUT** reading.

Because there are only eleven sectors, this raw percentage moves in discrete steps. It is useful, but it can be visually noisy.

### 3. Continuous MM50 participation score

The main blue series, **Score MM50**, is designed to produce a smoother and more informative measure of breadth.

For each sector, its distance from the MM50 is transformed through a logistic function:

```text
Sector score = 100 / (1 + exp(-(Distance / 3.2)))
```

This means that a sector exactly at its MM50 contributes approximately 50 to the score. A sector increasingly above its MM50 contributes progressively closer to 100, while a sector increasingly below its MM50 contributes progressively closer to 0.

The sector scores are then averaged and smoothed with an 8-session exponentially weighted moving average.

The resulting breadth score therefore reflects both:

- how many sectors are above or below trend;
- how far each sector is from that trend.

This is why **Score MM50 is not the same thing as the raw percentage of sectors above the MM50**.

### 4. Breadth regimes

The latest Score MM50 is classified as:

```text
>= 70   Participation large
<= 35   Participation fragile
35–70   Participation intermédiaire
```

The chart also measures the change in the breadth score over 20 trading sessions.

A change greater than +2 points is labelled **élargissement**, a change below -2 points is labelled **rétrécissement**, and smaller changes are treated as broadly stable.

### 5. Current-sector diagnostics

Below the time series, the dashboard shows a card for each sector with:

- sector ticker;
- sector name;
- current percentage distance from its MM50;
- whether it is above or below the MM50;
- a small bar proportional to the magnitude of that distance.

This makes it possible to identify which sectors are driving the aggregate breadth reading instead of looking only at the headline score.

### 6. S&P 500 normalized benchmark

The breadth chart can also display an **S&P 500 normalisé** series.

This is not an S&P 500 return series. For the selected horizon, SPY is rescaled between 0 and 100 using the minimum and maximum values observed in that period:

```text
Normalized SPY =
(SPY - period minimum)
────────────────────── × 100
(period maximum - period minimum)
```

Its purpose is visual comparison with breadth, especially for identifying divergences.

For example, a rising normalized S&P 500 accompanied by a falling participation score can indicate that index performance is being carried by a narrowing group of sectors. Conversely, an expanding participation score during a market advance indicates broader confirmation.

The normalized benchmark should therefore be interpreted as a shape-comparison tool, not as a performance percentage.

### 7. Long-horizon readability

The last chart adapts automatically to the selected horizon.

When the number of observations becomes large, the script reduces the number of rendered points to approximately the useful horizontal resolution of the canvas. This affects display efficiency only; the underlying calculations still use the full data.

On longer horizons:

- the raw `% secteurs > MM50` series is hidden by default once the chart becomes dense;
- the normalized S&P 500 is hidden by default on very dense horizons;
- the main Score MM50 remains visible;
- year labels are reduced to avoid visual overload.

The hidden series can still be restored by clicking their legend entries.

### How to read the participation chart

The chart is most useful when interpreted jointly with the S&P 500 rather than in isolation.

A rising market with a rising participation score suggests broader confirmation. A rising market with deteriorating participation suggests narrowing leadership. A falling market with a very low participation score indicates broad weakness, while an improving participation score during or after a decline can indicate that internal market conditions are stabilizing before the headline index fully reflects it.

The score is a market-breadth diagnostic, not a standalone timing rule.

## Data sources

The script uses several external sources:

- **Yahoo Finance via `yfinance`** for market prices, sector ETFs, VIX, MOVE and the US 10-year yield proxy;
- **Options Analysis Suite endpoint** for SPX put and call open interest;
- **World Government Bonds** for the US five-year sovereign CDS history.

The data-loading functions run independent sources in parallel when possible.

## Cache and fallback behaviour

A local folder named:

```text
.market_intelligence_cache
```

is created next to the script.

Fresh cache files are preferred to unnecessary network requests. The script can also use older cached observations when live requests fail.

If no usable live or cached data are available, the script contains deterministic synthetic fallback generators for market prices, SPX put/call data and US CDS data. This keeps the interface operational, but synthetic fallback data must not be interpreted as real market observations.

The footer of the dashboard reports whether each source is using live data, cache, stale cache or synthetic fallback. This source-status line should always be checked before interpreting the dashboard.

## Horizons

The interface supports:

```text
3 months
6 months
1 year
3 years
5 years
10 years
```

Changing the horizon recalculates period-dependent performance and chart windows while reusing the in-memory source bundle for up to five minutes unless a forced refresh is requested.

## Interaction

The dashboard includes:

- a fixed horizon selector;
- a manual data-refresh button;
- smooth vertical scrolling;
- crosshairs and tooltips on time-series charts;
- clickable legends to hide or restore individual series;
- fixed-size regular panels and a larger full-width participation panel.

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The main Python dependencies used by this module are `numpy`, `pandas`, `requests` and `yfinance`. Tkinter is part of the standard Python distribution on many systems but may need to be installed separately depending on the operating system.

## Run

From the repository root:

```bash
python 01_market_intelligence/market_intelligence.py
```

The application opens as a local desktop window; it does not start a web server.

## Notes

External web endpoints can change without notice. A source that currently works may require future parser or endpoint maintenance.

This module is intended for market research and monitoring. It does not constitute investment advice.
