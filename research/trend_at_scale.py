"""Stress-test the 4h MA-crossover trend rule on the LARGEST perp dataset (all Binance
USDT perps, max history). Equal-weight long/short, net of taker fee + funding drag.

Outputs: full-period, per-CALENDAR-YEAR (regime), walk-forward folds, per-symbol breadth,
MA fast/slow parameter plateau, cost sweep, vs buy-hold & always-flat. Note: universe is
currently-listed symbols (SURVIVORSHIP-biased) — flagged, quantified where possible.
"""
import sys
import numpy as np
import pandas as pd

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/poc_research/data/binance_all_4h.csv"
TF = "4h" if "4h" in PATH else "1d"
BARS_YR = 2190.0 if TF == "4h" else 365.0
FEE = 4.32 / 1e4
FUND = 0.20 * (4.0 if TF == "4h" else 24.0) / 8760.0
MA_A, MA_B = (24, 96) if TF == "4h" else (6, 24)


def load():
    df = pd.read_csv(PATH)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values(["symbol", "ts"])
    df["fwd"] = df.groupby("symbol")["close"].pct_change().shift(-1)
    return df


def positions(df, a, b):
    out = []
    for s, g in df.groupby("symbol"):
        g = g.sort_values("ts")
        sig = np.sign((g["close"].rolling(a).mean() - g["close"].rolling(b).mean()).values)
        out.append(pd.DataFrame({"symbol": s, "ts": g["ts"].values, "pos": np.nan_to_num(sig)}))
    p = pd.concat(out, ignore_index=True)
    p["ts"] = pd.to_datetime(p["ts"], utc=True)
    return p


def metrics(series):
    s = np.asarray(series, float); s = s[np.isfinite(s)]
    if len(s) < 5 or s.std() == 0:
        return dict(ret=0.0, sharpe=0.0, mdd=0.0, n=len(s))
    eq = np.cumprod(1 + s); peak = np.maximum.accumulate(eq)
    return dict(ret=round(100 * (eq[-1] - 1), 1),
                sharpe=round(float(s.mean() / s.std() * np.sqrt(BARS_YR)), 2),
                mdd=round(100 * float(np.max((peak - eq) / peak)), 1), n=len(s))


def book(df, pos, fee=FEE, fund=FUND, ts_min=None, ts_max=None, universe=None):
    sub = pos.merge(df[["symbol", "ts", "fwd"]], on=["symbol", "ts"], how="left")
    if universe is not None: sub = sub[sub.symbol.isin(universe)]
    if ts_min is not None: sub = sub[sub.ts >= ts_min]
    if ts_max is not None: sub = sub[sub.ts < ts_max]
    sub = sub.sort_values(["symbol", "ts"])
    sub["dp"] = sub.groupby("symbol")["pos"].diff().fillna(sub["pos"])
    sub["net"] = sub["pos"] * sub["fwd"].fillna(0) - fee * sub["dp"].abs() - fund * sub["pos"].abs()
    per = sub.groupby("ts")["net"].mean().sort_index()
    return metrics(per.values), per


def main():
    df = load()
    nyears = (df.ts.max() - df.ts.min()).days / 365.25
    print(f"DATA: {df.symbol.nunique()} symbols | {len(df):,} rows | {df.ts.min().date()} .. {df.ts.max().date()} (~{nyears:.1f}y) | {TF}")
    pos = positions(df, MA_A, MA_B)

    full, per = book(df, pos)
    print(f"\n=== TREND RULE MA-cross({MA_A}/{MA_B}), equal-wt long/short, fee {FEE*1e4}bps + 20%APR funding ===")
    print(f"  FULL: {full}")

    print("\n  per CALENDAR YEAR (regime):")
    for yr in range(df.ts.min().year, df.ts.max().year + 1):
        lo = pd.Timestamp(f"{yr}-01-01", tz="UTC"); hi = pd.Timestamp(f"{yr+1}-01-01", tz="UTC")
        m, _ = book(df, pos, ts_min=lo, ts_max=hi)
        # buy-hold (equal-weight long) for the year
        bh, _ = book(df, pos.assign(pos=1.0), fee=0.0, fund=0.0, ts_min=lo, ts_max=hi)
        print(f"    {yr}: rule SR {m['sharpe']:>6} ret {m['ret']:>7}% mdd {m['mdd']:>5}% (n={m['n']:>5}) | buy-hold SR {bh['sharpe']:>6} ret {bh['ret']:>7}%")

    # walk-forward (12 folds)
    times = np.sort(df["ts"].unique())
    nf = 12; b = [times[int(len(times) * i / nf)] for i in range(nf)] + [None]
    wf = [book(df, pos, ts_min=b[i], ts_max=b[i + 1])[0]["sharpe"] for i in range(nf)]
    print(f"\n  walk-forward ({nf} folds) SR: {wf}\n  folds positive: {sum(s>0 for s in wf)}/{nf} | mean {round(np.mean(wf),2)}")

    # per-symbol breadth
    persym = []
    for s, g in df.groupby("symbol"):
        if len(g) < 200: continue
        m, _ = book(df, pos, universe=[s])
        persym.append(m["sharpe"])
    persym = np.array(persym)
    print(f"\n  per-symbol (>=200 bars, {len(persym)} symbols): median SR {np.median(persym):.2f} | "
          f"fraction SR>0 {np.mean(persym>0):.2f} | fraction SR>0.5 {np.mean(persym>0.5):.2f}")

    # parameter plateau
    print("\n  MA fast/slow PARAMETER PLATEAU (full-period SR):")
    fasts = [12, 18, 24, 36, 48] if TF == "4h" else [3, 6, 9, 12]
    slows = [48, 72, 96, 168, 240] if TF == "4h" else [18, 24, 36, 48]
    for a in fasts:
        cells = []
        for bb in slows:
            if a >= bb: cells.append("   -  "); continue
            m, _ = book(df, positions(df, a, bb))
            cells.append(f"{m['sharpe']:>6.2f}")
        print(f"    fast {a:>3}: " + " ".join(cells) + f"   (slow={slows})")

    # cost sweep
    print("\n  cost sweep (full-period SR):")
    for fb in [0.0, 1.5, 4.32, 9.0]:
        m, _ = book(df, pos, fee=fb / 1e4)
        print(f"    fee {fb:>4}bps: SR {m['sharpe']}")

    bh_full, _ = book(df, pos.assign(pos=1.0), fee=0.0, fund=0.0)
    print(f"\n  BASELINES full: buy-hold(eq-wt long) SR {bh_full['sharpe']} ret {bh_full['ret']}% | always-flat SR 0.0")


if __name__ == "__main__":
    main()
