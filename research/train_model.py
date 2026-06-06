"""Train + persist the candidate TA-feature directional model on the large Binance universe.
This produces a FIXED model whose FORWARD performance the validation harness will measure.
(Periodically re-run offline to refresh; forward validation always scores a frozen model.)
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from ta_test import build

BIN = "/tmp/poc_research/data/binance_all_4h.csv"
CACHE = "/tmp/poc_research/data/_bin_feats.parquet"
OUTDIR = "/home/varon/.openclaw/agents/varon/workspace/github/varon-fi/poc/models"
import os; os.makedirs(OUTDIR, exist_ok=True)

import os
if os.path.exists(CACHE):
    df = pd.read_parquet(CACHE)
    FCOLS = [c for c in df.columns if c not in ("y", "fwd1", "symbol", "ts", "vol")]
else:
    df, FCOLS = build(BIN)

m = lgb.LGBMRegressor(n_estimators=500, num_leaves=31, learning_rate=0.02, subsample=0.8,
                      colsample_bytree=0.7, min_child_samples=500, reg_lambda=10, n_jobs=6, verbosity=-1)
m.fit(df[FCOLS].values, df["y"].values)
m.booster_.save_model(f"{OUTDIR}/ta_lgbm.txt")
json.dump({"features": FCOLS, "horizon_bars": 6, "trained_on": "binance_usdm_perps_4h",
           "trained_through": str(df.ts.max()), "n_train_rows": int(len(df)),
           "model": "LightGBM directional (predict 6-bar fwd log-return)",
           "position": "sign(pred) with |pred|>0.005 threshold, vol-targeted 15% ann, cap 3x"},
          open(f"{OUTDIR}/ta_lgbm_meta.json", "w"), indent=2)
print(f"saved model ({len(FCOLS)} feats, trained on {len(df):,} rows thru {df.ts.max()}) -> {OUTDIR}/ta_lgbm.txt")
