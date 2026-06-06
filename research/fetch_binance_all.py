"""Pull the LARGEST relevant perp dataset: ALL Binance USDM perpetual USDT pairs,
max history, at a chosen interval. For robust trend-rule validation across regimes."""
import sys, json, time, urllib.request
import pandas as pd

INTERVAL = sys.argv[1] if len(sys.argv) > 1 else "4h"
OUT = f"/tmp/poc_research/data/binance_all_{INTERVAL}.csv"
START = int(pd.Timestamp("2019-01-01", tz="UTC").timestamp() * 1000)
LIMIT = 1500
BASE = "https://fapi.binance.com/fapi/v1"

def get(url):
    return json.loads(urllib.request.urlopen(url, timeout=30).read())

info = get(f"{BASE}/exchangeInfo")
syms = [s["symbol"] for s in info["symbols"]
        if s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"]
print(f"universe: {len(syms)} USDT perpetuals | interval {INTERVAL}", flush=True)

rows = []
for j, sym in enumerate(syms):
    start = START; n = 0
    while True:
        try:
            k = get(f"{BASE}/klines?symbol={sym}&interval={INTERVAL}&startTime={start}&limit={LIMIT}")
        except Exception as e:
            print(sym, "err", repr(e)[:60], flush=True); break
        if not k:
            break
        for c in k:
            rows.append((sym.replace("USDT", ""), c[0], float(c[1]), float(c[2]), float(c[3]),
                         float(c[4]), float(c[5]), int(c[8])))
        n += len(k); start = k[-1][0] + 1
        if len(k) < LIMIT:
            break
        time.sleep(0.28)
    if j % 25 == 0:
        print(f"  [{j}/{len(syms)}] {sym}: {n} bars (rows so far {len(rows)})", flush=True)

df = pd.DataFrame(rows, columns=["symbol", "tms", "open", "high", "low", "close", "volume", "count"])
df["ts"] = pd.to_datetime(df["tms"], unit="ms", utc=True)
df = df.drop(columns="tms").drop_duplicates(["symbol", "ts"]).sort_values(["symbol", "ts"])
df.to_csv(OUT, index=False)
print(f"\nSAVED {OUT}: {len(df)} rows, {df.symbol.nunique()} symbols, {df.ts.min()} .. {df.ts.max()}", flush=True)
