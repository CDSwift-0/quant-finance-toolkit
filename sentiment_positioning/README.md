# Sentiment and Positioning

Dashboard combining retail-investor sentiment with institutional futures positioning. AAII survey data are used for sentiment, while CFTC Traders in Financial Futures data are used to track S&P 500 Asset Manager long, short and net positioning.

Run the dashboard:

```bash
python SD.py
```

The auxiliary scripts `aaii.py` and `CFTC.py` can be run independently for data extraction and inspection. Generated CSV files, plots and caches are ignored by Git.
