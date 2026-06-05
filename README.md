# Directional Trading POC — Hyperliquid Perps

Research POC for a **directional** (long / flat / short) trading strategy on Hyperliquid
perpetual futures. After an exhaustive, leakage-disciplined search across approaches, the
strategy that survives honest out-of-sample, cost- and funding-aware validation is classic
**trend-following**: a moving-average crossover at the 4-hour horizon.

> **Deliverable:** [`mom_perp_poc.ipynb`](mom_perp_poc.ipynb) — self-contained, reproducible.

## TL;DR result

**4h MA-crossover (24/96), equal-weight long/short across 7 liquid core perps**
(BTC, ETH, SOL, LINK, ARB, OP, HYPER), on **real Hyperliquid data**, held-out test
(2026-02-26 → 2026-06-05), net of 4.5 bps taker fee:

| Metric (held-out test) | Strategy (20% APR funding) | Strategy (no funding) | Buy & Hold | Always-Flat |
| --- | --- | --- | --- | --- |
| Return | **+18.5%** | +25.2% | −19.3% | 0% |
| Annualized Sharpe | **1.59** | 2.03 | −1.12 | 0 |
| Max drawdown | 18% | 17% | 36% | 0% |

Robustness: broad parameter plateau (19/30 fast×slow configs survive funding + walk-forward),
positive on 5–6 of 7 symbols, 4/6 walk-forward folds positive, survives funding to ~50% APR
(turnover only ~0.02 flips/bar), and positive in **both** up- and down-market sub-periods.
It made money while buy-hold lost 19% → genuine two-sided trend capture, not long beta.

## ⚠️ Honest caveats — this is a paper-probe, not a funded strategy

- The dataset is **~8 months and mostly one regime** (a sustained alt-bear). The up-market
  evidence is a short window; the two oldest walk-forward folds are negative. A sustained
  **bull / chop** out-of-sample is **unproven**.
- Treat this as **paper-probe grade**. The right next step is forward **shadow → paper**
  validation to accumulate genuine out-of-sample data across regimes (the platform already
  streams live Hyperliquid data) before any capital is sized.

## Quick start

```bash
pip install -r requirements.txt          # numpy, pandas, matplotlib suffice for the POC
jupyter nbconvert --to notebook --execute --inplace mom_perp_poc.ipynb
# or open interactively:
jupyter notebook mom_perp_poc.ipynb
```

The notebook loads bundled real data (`data/hyperliquid_4h_core.csv`), runs the leakage-free
backtest, walk-forward, per-symbol and regime breakdowns, funding stress, and baselines, then
writes artifacts to `validation_results/`.

## Data

- **Source:** real Hyperliquid 4h OHLC for the core perps, exported from the platform
  Postgres `ohlcs` table (the live ingestion the platform collects). Bundled as
  `data/hyperliquid_4h_core.csv` for reproducibility.
- **Refresh:** the last cell of the notebook contains an optional helper to re-pull candles
  from the public Hyperliquid API (`/info` `candleSnapshot`) — no keys required.

## Methodology / anti-overfitting discipline

These are the practices that separated the real edge from artifacts:

- **No lookahead:** signals use only data up to bar *t*; PnL uses the next-bar return.
  Momentum/MA windows are computed on **full per-symbol history** (a head-of-window slice
  bug had previously inflated results — fixed and re-baselined).
- **Realistic costs:** taker fee on turnover **and** perp **funding** drag on held exposure
  (stress-tested 0–50% APR). High-turnover strategies that ignore this look great and aren't.
- **Walk-forward** (6 contiguous OOS folds), **regime split** (up vs down market),
  **per-symbol** breakdown, and **parameter-plateau** checks (robust region, not a single
  lucky config).
- **Baselines:** every result is judged against **always-flat** and **buy & hold**.

## Research journey — why trend-following, and what was rejected

This repo began as a reproduction of [conditionWang/DRQN_Stock_Trading](https://github.com/conditionWang/DRQN_Stock_Trading)
(see `drqn_trading.ipynb`, `train_drqn.py`, `validate_drqn.py`). The honest findings:

1. **The DRQN reference's headline returns were a leverage artifact** — it trades a fixed
   *share* count financed by borrowing (the balance only absorbs P&L), i.e. implicit
   ~2–70× leverage with no equity floor. Remove the leverage and the edge disappears.
2. **High-frequency / microstructure directional** (1–5 min; order-book imbalance, OFI, CVD
   from the platform's `features_microstructure`) has **no edge after taker costs** —
   information coefficients ≈ 0 and flip sign out-of-sample. Microstructure alpha is a
   maker / market-making phenomenon, out of scope for a directional taker strategy.
3. **RL / ML** (QR-DQN, LightGBM) **overfit and underperformed a simple rule.** Algorithm
   sophistication was not the lever; cost realism and evaluation discipline were.
4. **Plain `sign(momentum)` TSMOM** only looked good due to the head-of-window bug + ignoring
   funding; corrected, its entire edge was the short leg in one regime.
5. **MA-crossover trend-following won** because its hysteresis slashes turnover (so funding
   and fees are amortized) and it captures trend two-sidedly — the most-documented systematic
   edge in finance (managed futures / CTAs), now validated on Hyperliquid perps.

## Files

| Path | What |
| --- | --- |
| `mom_perp_poc.ipynb` | **The deliverable** — validated 4h MA-crossover trend POC |
| `data/hyperliquid_4h_core.csv` | Bundled real Hyperliquid 4h OHLC (core perps) |
| `validation_results/mom_perp_poc_*` | Saved metrics (JSON) + equity-curve plot (PNG) |
| `drqn_trading.ipynb`, `train_drqn.py`, `validate_drqn.py` | Original DRQN reference reproduction (kept for history; superseded — see findings above) |
