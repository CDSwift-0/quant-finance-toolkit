# Global Macro

This module contains a desktop dashboard for US interest rates, inflation and macroeconomic conditions.

## Dashboard

`taux.py` displays:

- the effective Federal Funds rate;
- headline CPI, core CPI and PCE inflation;
- the US 10-year minus 2-year Treasury yield spread;
- the Chauvet–Piger recession-probability series;
- an ex-post real-rate proxy calculated as Fed Funds minus CPI inflation;
- the 12-month change in the main indicators.

When a series cannot be loaded and no cached observation is available, the affected block displays **Bientôt mis à jour** instead of remaining blank.

## Run

From the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 07_global_macro/taux.py
```

The program uses Tkinter for the interface and Matplotlib for the charts. Tkinter is included with many Python installations but may need to be installed separately on some Linux distributions.

## Data and behavior

FRED is the principal source. The script also tries public fallback routes when the main CSV endpoint is unavailable. No API key is required.

Each series is downloaded independently in a background thread, so one unavailable endpoint does not block the rest of the dashboard. A local cache is used to shorten later launches and changing the displayed horizon does not trigger a new download.

External data providers can change their endpoints or formats without notice. The dashboard is intended for research and does not constitute investment advice.
