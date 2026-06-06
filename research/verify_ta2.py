"""Deployability checks for the TA-feature directional model:
(1) turnover reduction -> wider cost headroom; (2) zero-shot transfer to actual Hyperliquid."""
import os, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from ta_test import build
warnings.filterwarnings("ignore")

BARS_YR = 2190.0
BIN = "/tmp/poc_research/data/binance_all_4h.csv"
HL = "/tmp/poc_research/data/ohlc_4h.csv"
TRAIN_END = pd.Timestamp("2025-08-01", tz="UTC")


def feats(path, cache):
    if os.path.exists(cache):
        df = pd.read_parquet(cache); FCOLS = [c for c in df.columns if c not in ("y","fwd1","symbol","ts","vol")]
        return df, FCOLS
    df, FCOLS = build(path)
    raw = pd.read_csv(path); raw["ts"] = pd.to_datetime(raw["ts"], utc=True)
    raw["vol"] = raw.groupby("symbol")["close"].pct_change().transform(lambda s: s.ewm(span=60, min_periods=20).std())
    df = df.merge(raw[["symbol","ts","vol"]], on=["symbol","ts"], how="left")
    df["vol"] = df["vol"].fillna(df["vol"].median())
    df.to_parquet(cache); return df, FCOLS


def book(s, fee_bps=4.32, fund_apr=0.20, voltgt=True):
    s = s.sort_values(["symbol","ts"]).copy()
    if voltgt: s["pos2"] = s["pos"] * np.clip(0.15/np.sqrt(BARS_YR)/(s["vol"]+1e-9), 0, 3)
    else: s["pos2"] = s["pos"]
    s["dp"] = s.groupby("symbol")["pos2"].diff().fillna(s["pos2"])
    fund = fund_apr*4.0/8760.0
    s["net"] = s["pos2"]*s["fwd1"] - fee_bps/1e4*s["dp"].abs() - fund*s["pos2"].abs()
    per = s.groupby("ts")["net"].mean().sort_index().values
    if len(per)<5 or per.std()==0: return dict(sr=0.0,ret=0.0,turn=0.0)
    eq=np.cumprod(1+per)
    return dict(sr=round(float(per.mean()/per.std()*np.sqrt(BARS_YR)),2), ret=round(100*(eq[-1]-1),1),
                turn=round(float(s.groupby("ts")["dp"].apply(lambda x:x.abs().mean()).mean()),3))


dfb, FCOLS = feats(BIN, "/tmp/poc_research/data/_bin_feats.parquet")
tr = dfb[dfb.ts < TRAIN_END]
m = lgb.LGBMRegressor(n_estimators=500, num_leaves=31, learning_rate=0.02, subsample=0.8,
                      colsample_bytree=0.7, min_child_samples=500, reg_lambda=10, n_jobs=6, verbosity=-1)
m.fit(tr[FCOLS].values, tr["y"].values)

# (1) turnover reduction on Binance OOS (post TRAIN_END)
te = dfb[dfb.ts >= TRAIN_END].copy(); te["pred"] = m.predict(te[FCOLS].values)
print(f"Binance OOS (post {TRAIN_END.date()}) rows {len(te):,} IC {spearmanr(te.pred,te.fwd1).correlation:+.4f}")
print("=== TURNOVER REDUCTION (Binance OOS, vol-targeted, 4.32bps+20%fund) ===")
te["pos"] = np.sign(te["pred"]); print(f"  raw sign        : {book(te)}")
for thr in (0.002, 0.005, 0.01):
    te["pos"] = np.where(te["pred"] > thr, 1.0, np.where(te["pred"] < -thr, -1.0, 0.0))
    print(f"  threshold {thr:<5} : {book(te)}")
# EMA-smoothed prediction (hysteresis)
te = te.sort_values(["symbol","ts"])
te["pred_s"] = te.groupby("symbol")["pred"].transform(lambda s: s.ewm(span=4).mean())
te["pos"] = np.sign(te["pred_s"]); print(f"  EMA-smoothed    : {book(te)}")
# cost headroom of best (threshold 0.005) across fees
te["pos"] = np.where(te["pred"]>0.005,1.0,np.where(te["pred"]<-0.005,-1.0,0.0))
print("  thr0.005 cost sweep:", {f: book(te, fee_bps=f)["sr"] for f in (4.32,9,15,25)})

# (2) ZERO-SHOT to actual Hyperliquid
dfh, FCOLSh = feats(HL, "/tmp/poc_research/data/_hl_feats.parquet")
dfh = dfh.copy(); dfh["pred"] = m.predict(dfh[FCOLS].values)
print(f"\n=== ZERO-SHOT on HYPERLIQUID (Binance-trained model, {dfh.symbol.nunique()} HL syms) IC {spearmanr(dfh.pred,dfh.fwd1).correlation:+.4f} ===")
dfh["pos"] = np.sign(dfh["pred"])
print(f"  full HL: {book(dfh)}")
SPLIT=pd.Timestamp('2026-02-26',tz='UTC')
print(f"  HL test (>=2026-02-26): {book(dfh[dfh.ts>=SPLIT])}")
dfh["pos"] = np.where(dfh["pred"]>0.005,1.0,np.where(dfh["pred"]<-0.005,-1.0,0.0))
print(f"  HL test thr0.005: {book(dfh[dfh.ts>=SPLIT])}")
