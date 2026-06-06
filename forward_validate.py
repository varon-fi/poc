#!/usr/bin/env python3
"""Forward paper-validation harness for directional Hyperliquid-perp models.

WHY: every BACKTEST edge in this repo (trend rule, DRQN/QR-DQN, DMN, meta-labeling,
TA-LightGBM) either decayed in the recent regime or failed to transfer to Hyperliquid.
The only test that can't be fooled by historical/cross-venue artifacts is FORWARD
out-of-sample. This harness, run on a schedule (every 4h), records each candidate model's
intended positions on LIVE Hyperliquid data and scores them against what actually happens
next — net of taker fee + funding — accumulating honest OOS vs always-flat over time.

OBSERVE-ONLY: logs intentions and paper PnL. It does NOT place orders.

Run:  python3 forward_validate.py     (idempotent; safe to run every 4h via cron)
"""
import json, os, time, urllib.request
import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = f"{HERE}/model/ta_lgbm.txt"
META = json.load(open(f"{HERE}/model/ta_lgbm_meta.json"))
RES = f"{HERE}/forward_results"; os.makedirs(RES, exist_ok=True)
PRED_LOG = f"{RES}/predictions.jsonl"      # appended each run: per-bar target positions per model
REPORT = f"{RES}/forward_report.json"

# liquid Hyperliquid perps (skipped automatically if unavailable)
UNIVERSE = ["BTC", "ETH", "SOL", "XRP", "DOGE", "LINK", "AVAX", "LTC", "BCH", "ARB", "OP",
            "SUI", "APT", "INJ", "TIA", "SEI", "WLD", "ADA", "NEAR", "ATOM", "AAVE", "UNI", "DOT", "FIL", "HYPE"]
FEE = 4.32 / 1e4
FUND_PER_BAR = 0.20 * 4.0 / 8760.0
BARS_YR = 2190.0
VTGT = 0.15 / np.sqrt(BARS_YR)
THR = 0.005
FCOLS = META["features"]
import talib  # TA-Lib indicators (same feature set the model was trained on)
CDL = ["CDLENGULFING","CDLHAMMER","CDLSHOOTINGSTAR","CDLDOJI","CDLMORNINGSTAR","CDLEVENINGSTAR",
       "CDL3WHITESOLDIERS","CDL3BLACKCROWS","CDLHARAMI","CDLPIERCING"]


def ta_features(g):
    o,h,l,c,v = (g[x].values.astype(float) for x in ("open","high","low","close","volume"))
    f={}
    f["rsi"]=talib.RSI(c,14)/100-0.5; _,_,hist=talib.MACD(c); f["macd_hist"]=hist/(c+1e-9)
    f["adx"]=talib.ADX(h,l,c,14)/100; f["pdi"]=talib.PLUS_DI(h,l,c,14)/100; f["mdi"]=talib.MINUS_DI(h,l,c,14)/100
    f["cci"]=np.clip(talib.CCI(h,l,c,14)/200,-2,2); f["mfi"]=talib.MFI(h,l,c,v,14)/100-0.5
    f["roc12"]=talib.ROC(c,12)/100; k,d=talib.STOCH(h,l,c); f["stochk"]=k/100-0.5
    f["willr"]=talib.WILLR(h,l,c,14)/100; f["aroon"]=talib.AROONOSC(h,l,14)/100; f["natr"]=talib.NATR(h,l,c,14)/100
    up,mid,lo=talib.BBANDS(c,20); f["bb_pctb"]=(c-lo)/(up-lo+1e-9)-0.5; f["bb_w"]=(up-lo)/(mid+1e-9)
    f["obv_sl"]=pd.Series(talib.OBV(c,v)).pct_change(12).values; f["adosc"]=talib.ADOSC(h,l,c,v)/(talib.OBV(c,v)+1e-9)
    sar=talib.SAR(h,l); f["sar_d"]=(c-sar)/(c+1e-9)
    for p in (20,50,100): f[f"ma_{p}"]=c/(talib.SMA(c,p)+1e-9)-1.0
    cs=np.zeros_like(c)
    for nm in CDL: s=getattr(talib,nm)(o,h,l,c)/100.0; cs=cs+s; f[nm.lower()]=s
    f["cdl_net"]=cs
    return pd.DataFrame(f, index=g.index)


def hl_candles(coin, bars=600):
    end = int(time.time() * 1000); start = end - bars * 4 * 3600 * 1000
    req = urllib.request.Request("https://api.hyperliquid.xyz/info",
        data=json.dumps({"type": "candleSnapshot", "req": {"coin": coin, "interval": "4h",
                         "startTime": start, "endTime": end}}).encode(),
        headers={"Content-Type": "application/json"})
    k = json.loads(urllib.request.urlopen(req, timeout=30).read())
    now = time.time() * 1000
    rows = [{"symbol": coin, "ts": pd.to_datetime(x["t"], unit="ms", utc=True),
             "open": float(x["o"]), "high": float(x["h"]), "low": float(x["l"]),
             "close": float(x["c"]), "volume": float(x["v"]), "count": int(x["n"])}
            for x in k if x["T"] <= now]                      # only COMPLETE bars
    return pd.DataFrame(rows)


def current_positions():
    booster = lgb.Booster(model_file=MODEL_FILE)
    out = []
    for s in UNIVERSE:
        try:
            g = hl_candles(s)
        except Exception:
            continue
        if len(g) < 150:
            continue
        g = g.sort_values("ts").reset_index(drop=True)
        F = ta_features(g).reset_index(drop=True)
        g["vol"] = g["close"].pct_change().ewm(span=60, min_periods=20).std()
        d = pd.concat([g[["symbol", "ts", "close", "vol"]], F], axis=1).dropna(subset=FCOLS + ["vol"])
        if d.empty:
            continue
        last = d.iloc[-1]
        pred = float(booster.predict(np.array([last[FCOLS].values]))[0])
        vscale = float(min(VTGT / (last["vol"] + 1e-9), 3.0))
        ta_ml = (1.0 if pred > THR else -1.0 if pred < -THR else 0.0) * vscale
        ma = d["close"].rolling(24).mean().iloc[-1] - d["close"].rolling(96).mean().iloc[-1]
        trend = float(np.sign(ma)) * vscale
        out.append({"symbol": s, "bar_ts": last["ts"].isoformat(), "close": float(last["close"]),
                    "ta_ml": ta_ml, "trend": trend, "flat": 0.0, "buyhold": vscale})
    return pd.DataFrame(out)


def append_log(cur):
    bar = cur["bar_ts"].iloc[0] if len(cur) else None
    logged = set()
    if os.path.exists(PRED_LOG):
        for line in open(PRED_LOG):
            try: logged.add(json.loads(line)["bar_ts"])
            except Exception: pass
    new = 0
    with open(PRED_LOG, "a") as fh:
        for _, r in cur.iterrows():
            if r["bar_ts"] in logged:
                continue
            fh.write(json.dumps(r.to_dict()) + "\n"); new += 1
    return new


def score_forward():
    """Recompute the forward paper record from the prediction log (idempotent)."""
    if not os.path.exists(PRED_LOG):
        return {}
    rows = [json.loads(l) for l in open(PRED_LOG) if l.strip()]
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    df["bar_ts"] = pd.to_datetime(df["bar_ts"], utc=True)
    df = df.sort_values(["symbol", "bar_ts"])
    df["ret"] = df.groupby("symbol")["close"].pct_change().shift(-1)   # realized next-bar return
    models = ["ta_ml", "trend", "flat", "buyhold"]
    rep = {"bars_logged": int(df["bar_ts"].nunique()),
           "scored_bars": int(df.dropna(subset=["ret"])["bar_ts"].nunique()),
           "first_bar": str(df["bar_ts"].min()), "last_bar": str(df["bar_ts"].max()), "models": {}}
    for m in models:
        d = df.dropna(subset=["ret"]).copy()
        d["dp"] = d.groupby("symbol")[m].diff().fillna(d[m])
        d["net"] = d[m] * d["ret"] - FEE * d["dp"].abs() - FUND_PER_BAR * np.abs(d[m])
        per = d.groupby("bar_ts")["net"].mean().sort_index()
        if len(per) >= 2 and per.std() > 0:
            eq = float(np.prod(1 + per.values))
            sr = float(per.mean() / per.std() * np.sqrt(BARS_YR))
        else:
            eq, sr = 1.0, 0.0
        rep["models"][m] = {"forward_ret_pct": round(100 * (eq - 1), 3),
                            "forward_ann_sharpe": round(sr, 3), "n_scored": int(len(per))}
    return rep


def main():
    cur = current_positions()
    n_new = append_log(cur) if len(cur) else 0
    rep = score_forward()
    rep["generated_at"] = pd.Timestamp.utcnow().isoformat()
    rep["universe_live"] = int(len(cur))
    json.dump(rep, open(REPORT, "w"), indent=2)
    print(f"logged {len(cur)} symbols ({n_new} new bar-rows) | bars scored so far: {rep.get('scored_bars',0)}")
    if rep.get("models"):
        print("FORWARD paper performance (since inception, net 4.32bps+20%APR funding):")
        for m, v in rep["models"].items():
            print(f"  {m:8s}: ret {v['forward_ret_pct']:>7}%  ann_SR {v['forward_ann_sharpe']:>6}  (n={v['n_scored']})")
        print("  -> meaningful only after weeks of accumulation; judge ta_ml/trend vs flat & buyhold.")


if __name__ == "__main__":
    main()
