# Long / Short Research

Systematic screening tools for equities and commodities.

`mom.py` ranks an equity universe using the SMA200/SMA21 structure, recent performance and a secondary slope-confirmation rule before selecting long and short candidates.

`commo.py` applies trend and moving-average filters to a basket of commodity futures and exports the selected positions.

`tradingview_sma_spread.pine` is a Pine Script v6 indicator built around the SMA21/SMA200 spread, a zero-lag trend filter, reversal scoring and lower standard-deviation reference levels.

Run the Python screens:

```bash
python mom.py
python commo.py
```
