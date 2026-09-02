# Market Structure

S&P 500 market-breadth research based on rolling log-price regression extremes.

The module is intentionally reduced to a single executable script: `market_breadth.py`. It replaces the previous multi-file workflow and has no graphical user interface.

## Method

The script retrieves the current S&P 500 constituent list, downloads adjusted weekly prices from Yahoo Finance, and fits a rolling linear regression to each constituent's log-price series.

For each valid stock, the latest regression residual is expressed in standard-deviation units. The script then measures:

- the share of constituents above the positive sigma threshold;
- the share below the negative sigma threshold;
- the net difference between those two groups;
- the number of constituents with valid observations.

The rolling regressions are computed from vectorized rolling sums rather than repeated `numpy.polyfit` calls.

## Default parameters

The default analysis starts in 1995, uses a 500-week rolling regression window, and defines an extreme observation as a residual beyond ±1.5 standard deviations.

## Usage

```bash
python market_structure/market_breadth.py
```

Force fresh Yahoo Finance data:

```bash
python market_structure/market_breadth.py --refresh
```

Change the rolling window and threshold:

```bash
python market_structure/market_breadth.py --window 500 --threshold 1.5
```

## Outputs

The script creates exactly two PNG files next to the script:

```text
breadth_extremes.png
breadth_net.png
```

`breadth_extremes.png` plots the percentage of valid S&P 500 constituents above and below the selected regression threshold.

`breadth_net.png` plots the net balance, defined as the percentage above the positive threshold minus the percentage below the negative threshold.

A temporary operating-system cache is used for repeated runs. Cache files are not stored in the repository directory.

## Dependencies

```bash
pip install numpy pandas requests yfinance matplotlib lxml
```

The constituent list is sourced from Wikipedia and prices are sourced from Yahoo Finance. If the S&P 500 constituent request fails, the script falls back to a smaller hard-coded universe.
