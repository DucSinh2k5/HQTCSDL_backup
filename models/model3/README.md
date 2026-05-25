# HQTCSDL Model 3 - Trading Signal Classification

Model 3 phan loai tin hieu BUY/HOLD/SELL dua tren future return bang XGBoost Classifier.

Mac dinh:

- SELL: future return <= -1%
- HOLD: -1% < future return < 1%
- BUY: future return >= 1%

Chay train va tao report:

```bash
python main.py
```

Chay walk-forward backtest:

```bash
python walk_forward.py
```

Tao dashboard HTML tu cac report da sinh:

```bash
python dashboard_model3.py
```

Output chinh:

- `models/trading_signal_xgb_classifier.pkl`
- `reports/predictions.csv`
- `reports/metrics.json`
- `reports/backtest.csv`
- `reports/backtest_metrics.json`
- `reports/model3_dashboard.html`
