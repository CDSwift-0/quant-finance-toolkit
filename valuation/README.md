# S&P 500 Sector Valuation

Valuation workflow for the 11 GICS sectors of the S&P 500. The script retrieves major ETF holdings and company financial information, estimates company-level intrinsic values using DCF-style assumptions and aggregates the results at sector and index level.

Run the graphical application:

```bash
python DCF.py
```

The Financial Modeling Prep API key is optional. If used, place it in the repository-level `.env` file as `FMP_API_KEY`. Output tables and charts are generated locally and excluded from version control.
