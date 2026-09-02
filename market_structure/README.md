# Market Structure

This module combines two breadth views of the S&P 500:

- the share of constituents trading above their 200-day moving average;
- the share of constituents located beyond ±1.5 standard deviations from a rolling log-price regression channel.

`dashboard_marche.py` integrates both views into a Tkinter dashboard. `regression_largeur.py` can also be run independently to generate the regression-breadth charts.

Run the integrated dashboard:

```bash
python lancer.py
```
