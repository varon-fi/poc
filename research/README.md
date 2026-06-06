# Research pipeline (reproduces the model + findings)

These scripts assume a local `data/` dir of OHLC CSVs and were run from a research venv.
Pipeline:

1. `fetch_binance_all.py 4h` — pull all Binance USDⓂ-perp 4h history (2019→now) to `data/`.
2. `ta_test.py data/binance_all_4h.csv 6` — TA-Lib features + expanding walk-forward (LightGBM)
   directional test (the strongest result: 6/6 walk-forward folds, mean SR 1.41).
3. `train_model.py` — train + freeze the model to `../model/ta_lgbm.txt`.
4. `verify_ta_dir.py` — survivorship/capacity, cost & funding stress, per-year (it survives these).
5. `verify_ta2.py` — the disqualifiers: recent-holdout IC ≈ 0, and **zero-shot on Hyperliquid 2026
   SR −1.12** → the edge decayed and does not transfer → hence forward validation.
6. `trend_at_scale.py` — confirms the 4h MA-cross rule is mediocre at scale (SR 0.34 over 6.7y).

Paths are hardcoded to `/tmp/poc_research/data`; adjust to your local `data/` dir.
