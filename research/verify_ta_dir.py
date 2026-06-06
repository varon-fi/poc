"""Adversarial verification of the TA-feature directional edge: survivorship (liquid &
long-lived subsets), cost stress, vol-targeting, per-year — all from one walk-forward pass."""
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from ta_test import build
warnings.filterwarnings("ignore")

PATH = "/tmp/poc_research/data/binance_all_4h.csv"
BARS_YR = 2190.0


def main():
    df, FCOLS = build(PATH)
    raw = pd.read_csv(PATH); raw["ts"] = pd.to_datetime(raw["ts"], utc=True)
    raw["dvol"] = raw["close"] * raw["volume"]
    raw["vol"] = raw.groupby("symbol")["close"].pct_change().transform(lambda s: s.ewm(span=60, min_periods=20).std())
    df = df.merge(raw[["symbol", "ts", "dvol", "vol"]], on=["symbol", "ts"], how="left")
    df["vol"] = df["vol"].fillna(df["vol"].median())

    # walk-forward, collect OOS predictions ONCE
    times = np.sort(df["ts"].unique()); NF = 8
    bnds = [times[int(len(times) * i / NF)] for i in range(NF)] + [times[-1] + np.timedelta64(1, "ns")]
    oos = []
    for i in range(2, NF):
        tr = df[df.ts < bnds[i]]; te = df[(df.ts >= bnds[i]) & (df.ts < bnds[i + 1])]
        m = lgb.LGBMRegressor(n_estimators=400, num_leaves=31, learning_rate=0.02, subsample=0.8,
                              colsample_bytree=0.7, min_child_samples=500, reg_lambda=10, n_jobs=6, verbosity=-1)
        m.fit(tr[FCOLS].values, tr["y"].values)
        te = te.copy(); te["pred"] = m.predict(te[FCOLS].values)
        oos.append(te[["symbol", "ts", "pred", "fwd1", "dvol", "vol"]])
    O = pd.concat(oos, ignore_index=True)
    print(f"OOS rows {len(O):,} | pooled IC {spearmanr(O.pred, O.fwd1).correlation:+.4f}")

    def bk(o, fee_bps=4.32, fund_apr=0.20, voltgt=False, universe=None, ts_min=None, ts_max=None):
        s = o
        if universe is not None: s = s[s.symbol.isin(universe)]
        if ts_min is not None: s = s[s.ts >= ts_min]
        if ts_max is not None: s = s[s.ts < ts_max]
        s = s.sort_values(["symbol", "ts"]).copy()
        if voltgt:
            s["pos"] = np.sign(s["pred"]) * np.clip(0.15 / np.sqrt(BARS_YR) / (s["vol"] + 1e-9), 0, 3)
        else:
            s["pos"] = np.sign(s["pred"])
        s["dp"] = s.groupby("symbol")["pos"].diff().fillna(s["pos"])
        fund = fund_apr * 4.0 / 8760.0
        s["net"] = s["pos"] * s["fwd1"] - fee_bps / 1e4 * s["dp"].abs() - fund * s["pos"].abs()
        per = s.groupby("ts")["net"].mean().sort_index().values
        if len(per) < 5 or per.std() == 0: return dict(sr=0.0, ret=0.0)
        eq = np.cumprod(1 + per)
        return dict(sr=round(float(per.mean() / per.std() * np.sqrt(BARS_YR)), 2), ret=round(100 * (eq[-1] - 1), 1))

    # liquidity & longevity subsets
    med_dvol = O.groupby("symbol")["dvol"].median().sort_values(ascending=False)
    liq150 = set(med_dvol.head(150).index); liq50 = set(med_dvol.head(50).index)
    span = df.groupby("symbol")["ts"].agg(["min", "count"])
    longlived = set(span[(span["min"] <= pd.Timestamp("2021-06-01", tz="UTC")) & (span["count"] >= 4000)].index)

    print("\n=== SURVIVORSHIP / CAPACITY (vol-targeted, 4.32bps+20%fund) ===")
    print(f"  full universe      : {bk(O, voltgt=True)}")
    print(f"  liquid top-150     : {bk(O, voltgt=True, universe=liq150)}")
    print(f"  liquid top-50      : {bk(O, voltgt=True, universe=liq50)}")
    print(f"  long-lived (>=2021, full hist, {len(longlived)} syms): {bk(O, voltgt=True, universe=longlived)}")

    print("\n=== COST STRESS (full, vol-targeted) ===")
    for f in (0, 1.5, 4.32, 9, 15, 25):
        print(f"  fee {f:>4}bps + funding 20%: {bk(O, fee_bps=f, voltgt=True)}")
    print("\n=== FUNDING STRESS (full, vol-targeted, fee 4.32) ===")
    for fa in (0.0, 0.20, 0.50, 1.0):
        print(f"  funding {int(fa*100):>3}%APR: {bk(O, fund_apr=fa, voltgt=True)}")

    print("\n=== PER YEAR (full, vol-targeted) ===")
    for yr in range(2021, 2027):
        lo = pd.Timestamp(f"{yr}-01-01", tz="UTC"); hi = pd.Timestamp(f"{yr+1}-01-01", tz="UTC")
        print(f"  {yr}: {bk(O, voltgt=True, ts_min=lo, ts_max=hi)}")

    print(f"\n=== raw (sign, no vol-target) full: {bk(O, voltgt=False)} | vol-targeted full: {bk(O, voltgt=True)}")


if __name__ == "__main__":
    main()
