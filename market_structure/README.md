# Market Structure

Experimental S&P 500 breadth research based on the position of individual constituents relative to rolling log-price regression channels.

## Files

### `regression_largeur.py`

The script retrieves the current S&P 500 constituent list from Wikipedia, with a smaller hard-coded fallback universe if the web request fails. Historical adjusted prices are downloaded from Yahoo Finance and resampled to weekly observations.

For each stock, the model fits a rolling linear regression to log prices and measures the latest regression residual in standard-deviation units.

Default parameters:

- start date: 1995-01-01;
- rolling window: 500 weekly observations;
- extreme threshold: ±1.5σ.

The resulting breadth series contains:

- the percentage of valid constituents above +1.5σ;
- the percentage below −1.5σ;
- the net difference between the two groups;
- the number of constituents with valid observations.

Downloaded prices and calculated breadth data are cached in `.regression_cache/`.

### `lancer.py`

This is a lightweight launcher that imports `main` from `dashboard_marche.py`.

## Current status

The market-structure module is experimental in the current repository snapshot. `dashboard_marche.py`, which is required by `lancer.py`, is not currently committed.

The computation logic for regression breadth is present in `regression_largeur.py`, but the plotting path should be treated as work in progress rather than a stable standalone interface.

## Dependencies

The module uses `numpy`, `pandas`, `requests`, `yfinance`, and `matplotlib`.
