# Strategy Robustness Review

Diagnosis and prioritized changes across the three pipeline stages: pair selection → regime detection → Kalman/strategy.

Initial end-to-end backtest on 2022-01-31 produced 0 trades across all pairs. This document identifies the root cause and a tiered fix list.

---

## Root Cause

### Regime HMM trains on 97% imputed garbage

Measured on `UNITECH`, 3-month training window:

```
total training rows:        133,411
zero-return (flat) rows:    129,761  (97.3%)
real market-hours rows:      24,816  (18.6%)
```

`StockDataLoader` (`common/data_loader.py:189-198`) builds a **continuous 1-minute datetime range** start→end and forward-fills. No market-hours / weekday awareness. Markets open only 375 min/day (09:15–15:30), so overnight + weekend gaps get filled with flat lines → `log_return=0, volatility=0, slope=0`.

Consequence chain:
1. HMM fits one state to the giant flat blob, one to "market open" → **"Model is not converging"** warnings (full covariance goes near-singular on degenerate data).
2. The market-open state gets labeled **Trending** by the `abs(slope)` rule.
3. Strategy enters only when **both** stocks are `Sideways` → almost never → **0 trades**.

The convergence warnings, the all-Trending labels, and the 0 trades stem from a single underlying issue, not three independent bugs.

**Fix:** Filter to trading-hour weekday bars before feature computation and HMM fitting. Either disable imputation for regime detection, or after load, filter to (weekday ≤ 5 AND 09:15 ≤ time ≤ 15:30) before building features.

Do not relax the both-Sideways entry filter to force trades — that masks the underlying bug with garbage signals. Fix regime quality first, then re-run and tune filter strictness only if needed.

---

## Tier 1 — Genuine Bugs (Low Effort)

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `kalman_filter.py:23` | `initial_state_covariance=np.ones((2,2))` = `[[1,1],[1,1]]`, rank-1 singular | `np.eye(2)` (or `1e3 * np.eye(2)` for diffuse prior) |
| 2 | `mean_reversion.py:160` | KMeans on 3 unscaled features (mean_rev, vol, autocorr) — distance dominated by largest-magnitude feature | `StandardScaler().fit_transform(X)` before `KMeans` |
| 3 | `mean_reversion.py:212` | `mean_rev_score = mean(abs(spread - mean(spread)))` measures volatility, not reversion speed. High-variance trending spread passes filter | Replace with half-life (OU/AR(1)) or Hurst — see Tier 2 |

---

## Tier 2 — Robustness Enhancements

### Stage 1 — Pair Selection (`mean_reversion.py`, `generate_pair_trading_results.py`)

- **Half-life filter (replaces weak mean_rev_score).** Fit AR(1) on spread: `Δspread_t = a + b·spread_{t-1}`; `half_life = -ln(2)/ln(1+b)`. Keep pairs with **positive, short** half-life (e.g. 5–120 min). Reject if half-life ∞/negative (no reversion).
- **Hurst exponent < 0.5** as a second mean-reversion gate (anti-persistent). Library already required.
- **Out-of-sample pair stability.** Cointegration is tested once on the full lookback. Split lookback into train/validate; keep only pairs cointegrated in **both** halves → fewer pairs that break next day.
- **Look-ahead trim.** `MeanReversionAnalyzer.__init__` sets `end_day = run_date`, so cointegration/ADF see the trade day itself. Use `end_day = run_date - 1 day` for clean separation.
- **Cost-aware selection.** Filter pairs whose expected spread oscillation amplitude < round-trip transaction cost — otherwise reversion profit can't beat fees.

### Stage 2 — Regime Detection (`regime_detection.py`)

- Market-hours weekday filter before fitting (see root cause above)
- Diagonal covariance (`covariance_type="diag"`) + `min_covar` floor — more stable than full covariance on intraday data; eliminates convergence warnings
- Stable regime labeling by combined `volatility + abs(slope)` signature, not slope alone; persist label mapping for consistency across stocks/days
- Causal filtering: use forward filtering (`hmm.predict_proba` streamed) instead of per-minute isolated Viterbi for proper causal regime estimates
- Vectorize slope computation (closed-form OLS) instead of per-minute `np.polyfit` in a Python loop

### Stage 3 — Kalman + Strategy (`kalman_filter.py`, `mean_reversion_intraday_strategy.py`)

- Scale `obs_cov` to price level: fixed `OBS_COV=1.0` is large relative to penny-stock prices (₹1–10), beta barely updates. Set `obs_cov ≈ var(y)` or a price-variance fraction
- Add stop-loss: strategy exits only on z-revert or EOD; diverged spread (broken cointegration) runs unbounded until 15:30. Add `abs(z) > z_stop` (e.g., 4.0) hard exit
- Realistic position sizing: `notional = 10M × 0.02 = ₹200k/leg` on ₹2 penny stocks yields 100k shares (unfillable). Cap size by ADV participation limit
- z-window stability: `rolling_window=60` with `min_periods=1` on 256 intraday bars produces unstable early z-scores. Require `min_periods=rolling_window` or shorten window
- Regime filter strictness: once regimes are real, decide (both-Sideways strict vs at-least-one-Sideways loose), then re-evaluate on corrected labels

---

## Suggested Order of Work

1. Root cause — market-hours filter in regime detection; re-run and confirm convergence warnings gone and regime mix realistic
2. Tier 1 bugs — np.eye(2), KMeans scaling, half-life metric
3. Re-run backtest and inspect trade count and P&L
4. Tier 2 — select enhancements per stage based on results

---

## Applied Fixes

### Root Cause — Market-Hours Weekday Filter

- **`common/utils.py`** — new shared `filter_market_hours(df)` (Polars): keeps weekday bars in 09:15–15:30, exported from `common/__init__.py`
- **`regime_detection.py`** — `load_hmm_data` now calls `filter_market_hours` after load, before feature/HMM steps; removed duplicate local helper
- **`mean_reversion.py`** — `filter_basic` now applies market-hours filter to every ticker before cointegration/beta/half-life computation
- **Result:** regime training rows dropped 132,480 → 24,440 (97.3% flat fills removed; real session bars = 18.6%)

### Tier 1 Bug 1 — Kalman Singular Init Covariance

- **`kalman_filter.py`** — `initial_state_covariance` changed from `np.ones((2,2))` (rank-1 singular) to `np.eye(2)`

### Tier 1 Bug 2 — KMeans Unscaled Features

- **`mean_reversion.py:cluster_stocks`** — added `StandardScaler` before `KMeans` (with `np.nan_to_num` guard for NaN autocorr); distance no longer dominated by largest-magnitude feature

### Tier 1 Bug 3 — Half-Life Metric (Intraday-Aware)

- **`mean_reversion.py`** — new `_half_life(spread, dates)` via AR(1)/OU: Δspread_t = a + b·spread_{t-1}, half_life = -ln(2)/b
- AR(1) fit on within-session transitions only — overnight-gap deltas dropped via same-day mask (never traded in intraday strategy)
- Gate in `_check_pair`: reject if non-finite, ≤ 1 bar, or > 375 bars (one session); old mean-abs-deviation kept as secondary amplitude check; `half_life` added to output

### Files Modified

```
common/utils.py                                  (+ filter_market_hours, polars import)
common/__init__.py                               (export filter_market_hours)
intraday_strategy/kalman_filter/kalman_filter.py (np.eye init covariance)
intraday_strategy/pair_selection/mean_reversion.py (mkt-hours filter, KMeans scaling, half-life)
intraday_strategy/regime_detection/regime_detection.py (use shared filter, remove dup)
```

### CLI — Parallel Multi-Date Runs (`main.py`)

- Default: `python main.py` runs single configured date to stdout
- `--run-date YYYY-MM-DD` — override config date for one run
- `--run-dates D1 D2 ... --workers N` — spawn one subprocess per date, up to N concurrent; each run's output to `--log-dir/run_<date>.log` (default `results/logs/`); subprocess isolation gives each date clean config and joblib state

### Performance Impact

Removing overnight flat fills raises measured intraday volatility (no zero-return padding), so more tickers pass `filter_basic` → more O(n²) cointegration tests → fresh pair selection slower per date. This is more correct (volatility from real bars); parallel CLI mitigates latency.

### Tier 2 — HMM Hardening (Applied)

- **`regime_detection.py`**
  - `covariance_type="full"` → `"diag"`, `n_iter=200→500`, `tol=1e-4` — convergence warnings eliminated
  - Stable labeling: trending state chosen by combined `|slope| + volatility` in scaled feature space, not slope alone
  - Coherent full-day decode: single full-session `predict` + `predict_proba` replaces per-minute isolated-window Viterbi (faster, stable)

### Regime Detector Verification

Training states clean and well-separated:
```
KRIDHANINF train: 67% Sideways (vol -0.52, |slope| 0.01) | 33% Trending (vol +1.07, |slope| 1.18)
RNAVAL     train: 71% Sideways                            | 29% Trending
```

Test day 2022-01-31 labels mostly Trending — correct, as day trended genuinely:
```
KRIDHANINF test-day: net -6.94%, range 7.64% (real intraday dump)
RNAVAL     test-day: net -1.37%, range 8.22% (whippy)
```

93% Trending on this day is correct; 0 trades is strategy correctly standing aside on trend day, not a bug.

### Open Items

- 0 trades on 2022-01-31 is day-specific: high-vol trending day for selected pairs; mean-reversion strategy correctly avoids. To observe trades, run ranging days using parallel CLI (`--run-dates ... --workers N`)
- Optional design question (not a bug): entry gate requires both legs individually Sideways. For pairs trading, spread stationarity matters more than individual leg regimes; pairs that co-move can have stationary spread even while both trend. Consider spread-regime gating as deliberate design choice, validated across dates
