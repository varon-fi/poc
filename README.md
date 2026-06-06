# Hyperliquid Perp Directional — Forward Paper-Validation Harness

A scheduled, **observe-only** harness that measures candidate directional models' **forward,
out-of-sample** performance on live Hyperliquid perpetuals. It is the conclusion of an
extensive research program whose central finding is blunt: **every backtest edge we found
either decayed in the current regime or failed to transfer to Hyperliquid.** The only test
that can't be fooled by historical/cross-venue artifacts is *forward* OOS — so that is what
this builds.

> `forward_validate.py` — pulls live Hyperliquid 4h data, records each model's intended
> positions, scores them against what actually happens next (net of fee + funding), and
> accumulates an honest forward record vs always-flat. **It does not place orders.**

## Why forward validation (the research findings)

Tested rigorously (leakage-free, cost- and funding-aware, walk-forward, regime-split,
survivorship/capacity checks) on real Hyperliquid data + 6.7 years / 528 symbols of Binance
perps:

| Approach | Backtest looked like | What killed it |
| --- | --- | --- |
| DRQN / QR-DQN (RL) | overfit | underperformed a 2-param rule; sample-starved |
| 5-min ML + microstructure | IC ≈ 0 | no edge after taker cost |
| DMN (Sharpe-loss NN) + 5yr transfer | val Sharpe < 0 | overfit; didn't generalize even on Binance |
| Meta-labeling NN | BCE = random | can't predict trade success |
| 4h MA-crossover **rule** | SR ~2 on Hyperliquid 2026 | a lucky window — SR 0.34 / loses to buy-hold over 6.7y |
| **TA-feature LightGBM** (the best) | SR 1.08, every regime 2021–26, survivorship-clean | **IC ≈ 0 in the recent holdout; SR −1.12 zero-shot on Hyperliquid 2026 — decayed + doesn't transfer** |

The TA-feature model (RSI/ADX/MACD/NATR/… on the full Binance universe) was a *genuine
historical edge* — but it decayed to ~0 in 2025–26 on **both** venues. The pattern is
consistent with directional crypto-alpha being competed away. So: **no backtest is
trustworthy; only forward results count.**

## What the harness does

- Pulls live **4h** candles for a liquid Hyperliquid universe from the public API
  (`candleSnapshot`), using only **complete** bars (no look-ahead on the in-progress bar).
- Computes the same TA-Lib features the model was trained on, loads the frozen
  `model/ta_lgbm.txt`, and forms **vol-targeted** target positions for four candidates:
  - `ta_ml` — the TA-feature LightGBM (threshold + 15% vol target, ≤ ~3× per-name);
  - `trend` — 4h MA-cross(24/96) rule;
  - `flat` — always-flat (the bar everything must beat);
  - `buyhold` — vol-targeted long.
- Appends each bar's intended positions to `forward_results/predictions.jsonl`, then
  **recomputes the entire forward record** (idempotent) by scoring logged positions against
  realized next-bar returns, **net of 4.32 bps taker + 20% APR funding**, and writes
  `forward_results/forward_report.json`.

Run it every 4h (cron / platform scheduler):

```bash
python3 forward_validate.py     # idempotent; observe-only
```

The forward numbers are meaningless until **weeks** of bars accumulate; then the question is
simply: does `ta_ml` (or `trend`) beat `flat` and `buyhold` *forward*, net of cost? If yes,
promote toward the platform's shadow→paper pipeline; if not, we have honestly ruled out
directional ML on current data.

## Model

`model/ta_lgbm.txt` (+ `ta_lgbm_meta.json`) — LightGBM predicting the 6-bar (24h) forward
log-return from 31 TA-Lib indicators, trained on 2.49M bars of Binance USDⓂ perps (528
symbols, 2019–2026). Reproduce via `research/` (fetch → features → train). Periodically
re-train offline; forward validation always scores a *frozen* model.

## Honest status

This is **not** a profitable strategy — it is the apparatus to find out whether one exists
*going forward*, without fooling ourselves again. The strongest candidate (`ta_ml`) has a
real historical edge that has decayed; forward validation will confirm or refute whether it
(or anything) works in the live regime. Roadmap if forward results turn positive: richer data
axes (funding/OI/liquidations/on-chain as they accumulate) and the platform shadow→paper path.

## Files

| Path | What |
| --- | --- |
| `forward_validate.py` | the forward paper-validation harness (self-contained) |
| `model/ta_lgbm.txt`, `model/ta_lgbm_meta.json` | frozen TA-feature model + metadata |
| `forward_results/` | runtime log + report (gitignored; accumulates where scheduled) |
| `research/` | reproduction pipeline (Binance fetch, TA features, training, full backtests) |
| `drqn_trading.ipynb`, `train_drqn.py`, `validate_drqn.py` | original DRQN reference (superseded — see findings) |
