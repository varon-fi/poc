"""Test the 'rich TA feature set' hypothesis: can a model on TA-Lib price/volume indicators
+ candlestick patterns learn a REGIME-ROBUST directional edge that price-alone could not?

Expanding-window walk-forward (train on all past, predict next fold) on multi-regime Binance.
LightGBM (robust tabular baseline). Net of 4.32bps taker + 20% APR funding. vs trend rule & buy-hold.
"""
import sys, warnings
import numpy as np
import pandas as pd
import talib
import lightgbm as lgb
warnings.filterwarnings("ignore")

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/poc_research/data/binance_4h.csv"
H = int(sys.argv[2]) if len(sys.argv) > 2 else 6          # forward horizon (bars) for target/direction
BARS_YR = 2190.0
FEE = 4.32 / 1e4
FUND = 0.20 * 4.0 / 8760.0
CDL = ["CDLENGULFING", "CDLHAMMER", "CDLSHOOTINGSTAR", "CDLDOJI", "CDLMORNINGSTAR",
       "CDLEVENINGSTAR", "CDL3WHITESOLDIERS", "CDL3BLACKCROWS", "CDLHARAMI", "CDLPIERCING"]


def ta_features(g):
    o, h, l, c, v = (g[x].values.astype(float) for x in ("open", "high", "low", "close", "volume"))
    f = {}
    f["rsi"] = talib.RSI(c, 14) / 100 - 0.5
    macd, sig, hist = talib.MACD(c)
    f["macd_hist"] = hist / (c + 1e-9)
    f["adx"] = talib.ADX(h, l, c, 14) / 100
    f["pdi"] = talib.PLUS_DI(h, l, c, 14) / 100
    f["mdi"] = talib.MINUS_DI(h, l, c, 14) / 100
    f["cci"] = np.clip(talib.CCI(h, l, c, 14) / 200, -2, 2)
    f["mfi"] = talib.MFI(h, l, c, v, 14) / 100 - 0.5
    f["roc12"] = talib.ROC(c, 12) / 100
    k, d = talib.STOCH(h, l, c)
    f["stochk"] = k / 100 - 0.5
    f["willr"] = talib.WILLR(h, l, c, 14) / 100
    f["aroon"] = talib.AROONOSC(h, l, 14) / 100
    f["natr"] = talib.NATR(h, l, c, 14) / 100
    up, mid, lo = talib.BBANDS(c, 20)
    f["bb_pctb"] = (c - lo) / (up - lo + 1e-9) - 0.5
    f["bb_w"] = (up - lo) / (mid + 1e-9)
    f["obv_sl"] = pd.Series(talib.OBV(c, v)).pct_change(12).values
    f["adosc"] = talib.ADOSC(h, l, c, v) / (talib.OBV(c, v) + 1e-9)
    sar = talib.SAR(h, l)
    f["sar_d"] = (c - sar) / (c + 1e-9)
    for p in (20, 50, 100):
        f[f"ma_{p}"] = c / (talib.SMA(c, p) + 1e-9) - 1.0
    # candlestick patterns (net + a few key ones), scaled -1..1
    cdl_sum = np.zeros_like(c)
    for name in CDL:
        s = getattr(talib, name)(o, h, l, c) / 100.0
        cdl_sum = cdl_sum + s
        f[name.lower()] = s
    f["cdl_net"] = cdl_sum
    F = pd.DataFrame(f, index=g.index)
    return F


def build(path):
    df = pd.read_csv(path); df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    feats, tgt, r1 = [], [], []
    for s, g in df.groupby("symbol"):
        g = g.sort_values("ts")
        F = ta_features(g)
        lc = np.log(g["close"])
        F["y"] = (lc.shift(-H) - lc).values          # forward H-bar log return (target)
        F["fwd1"] = g["close"].pct_change().shift(-1).values  # next-bar simple return (PnL)
        F["symbol"] = s; F["ts"] = g["ts"].values
        feats.append(F)
    out = pd.concat(feats, ignore_index=True)
    FCOLS = [c for c in out.columns if c not in ("y", "fwd1", "symbol", "ts")]
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=FCOLS + ["y", "fwd1"]).reset_index(drop=True)
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out, FCOLS


def book(sub):
    sub = sub.sort_values(["symbol", "ts"])
    sub["dp"] = sub.groupby("symbol")["pos"].diff().fillna(sub["pos"])
    sub["net"] = sub["pos"] * sub["fwd1"] - FEE * sub["dp"].abs() - FUND * sub["pos"].abs()
    per = sub.groupby("ts")["net"].mean().sort_index().values
    if len(per) < 5 or per.std() == 0: return dict(sharpe=0.0, ret=0.0, turn=0.0)
    eq = np.cumprod(1 + per)
    return dict(sharpe=round(float(per.mean() / per.std() * np.sqrt(BARS_YR)), 2),
                ret=round(100 * (eq[-1] - 1), 1),
                turn=round(float(sub.groupby("ts")["dp"].apply(lambda x: x.abs().mean()).mean()), 4))


def main():
    df, FCOLS = build(PATH)
    print(f"DATA {df.symbol.nunique()} syms, {len(df):,} rows, {df.ts.min().date()}..{df.ts.max().date()} | {len(FCOLS)} TA feats | H={H}")
    times = np.sort(df["ts"].unique())
    NF = 8
    bnds = [times[int(len(times) * i / NF)] for i in range(NF)] + [times[-1] + np.timedelta64(1, "ns")]
    fold_sr, oos = [], []
    for i in range(2, NF):  # need >=2 folds of history to train
        tr = df[df.ts < bnds[i]]; te = df[(df.ts >= bnds[i]) & (df.ts < bnds[i + 1])]
        m = lgb.LGBMRegressor(n_estimators=400, num_leaves=31, learning_rate=0.02, subsample=0.8,
                              colsample_bytree=0.7, min_child_samples=500, reg_lambda=10, n_jobs=6, verbosity=-1)
        m.fit(tr[FCOLS].values, tr["y"].values)
        pred = m.predict(te[FCOLS].values)
        te = te.copy(); te["pos"] = np.sign(pred)               # directional
        from scipy.stats import spearmanr
        ic = spearmanr(pred, te["y"].values).correlation
        bk = book(te)
        fold_sr.append(bk["sharpe"]); oos.append(te.assign(pred=pred))
        print(f"  fold {i} [{pd.Timestamp(bnds[i]).date()}..{pd.Timestamp(bnds[i+1]).date()}] IC {ic:+.4f} | net SR {bk['sharpe']:>6} ret {bk['ret']:>6}% turn {bk['turn']}")
    print(f"\n  WALK-FORWARD: folds positive {sum(s>0 for s in fold_sr)}/{len(fold_sr)} | mean SR {round(np.mean(fold_sr),3)}")
    allo = pd.concat(oos)
    print(f"  pooled OOS IC {pd.concat(oos).pipe(lambda d: __import__('scipy.stats',fromlist=['spearmanr']).spearmanr(d['pred'], d['y']).correlation):+.4f}")
    print(f"  pooled OOS book: {book(allo)}")
    # feature importance (top 12)
    imp = pd.Series(m.feature_importances_, index=FCOLS).sort_values(ascending=False).head(12)
    print("  top features:", ", ".join(f"{k}({int(v)})" for k, v in imp.items()))


if __name__ == "__main__":
    main()
