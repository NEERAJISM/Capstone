# Code Changes — Scope Review

All source changes, per file, with scope classification:
- **bug** — fixes incorrect behavior
- **method** — methodology correctness (affects results validity)
- **infra** — tooling/CLI, no strategy-logic change
- **diag** — instrumentation/logging only
- **chore** — cosmetic

No change alters the strategy's thesis or invents signals. The look-ahead fix (method scope) makes results more conservative.

---

## Strategy / Model Logic

### `intraday_strategy/pair_selection/mean_reversion.py`
- **method: Look-ahead fix** — `end_day = run_date` → `run_date − 1`. Pair selection (cointegration, ADF, beta, half-life) no longer sees traded session. Single change affecting result validity; makes selection out-of-sample
- **bug: KMeans scaling** — added `StandardScaler` before clustering; prevents distance metric domination by largest-magnitude feature
- **method: Reversion metric (half-life)** — replaced mean-absolute-deviation (spread amplitude) with AR(1)/OU intraday half-life (reversion speed); fit on within-session transitions only, excludes overnight deltas; gate rejects if non-finite, ≤1 bar, or >375 bars
- **bug: Market-hours filter** — loads real session bars only (weekday 09:15–15:30); drops imputed overnight/weekend flats

### `intraday_strategy/kalman_filter/kalman_filter.py`
- **bug** — `initial_state_covariance` changed from `np.ones((2,2))` (rank-1 singular) to `np.eye(2)`

### `intraday_strategy/regime_detection/regime_detection.py` (major rewrite)
- **bug: Duplicate removal** — removed unreachable first copy of `detect_regimes_train_test_rolling`
- **bug: Polars throughout** — rewrite from pandas to Polars (StockDataLoader returns Polars; pandas version crashes)
- **bug: Data load window** — `load_hmm_data` loaded `end=trading_date` (midnight) leaving trading-day bars empty; now loads through 15:30 and filters to market hours
- **method: HMM convergence** — `covariance_type` "full"→"diag", `n_iter` 200→500, `tol=1e-4`; convergence warnings eliminated
- **method: Regime labeling** — trending state selected by combined `|slope|+volatility` (scaled feature space), not slope alone; robust when features overlap
- **method: Full-day decode** — replaced per-minute isolated-window Viterbi (refit scaler on each 60-bar slice) with single full-session `predict` + `predict_proba` (faster, stable)

### `intraday_strategy/mean_reversion_intraday_strategy.py`
- **diag: Entry funnel counter** — logs per-pair rejection counts (in_window, flat_ready, rej_regime, rej_vol_filter, z_below_entry, entries) to identify which gate blocks entries
- **infra: Regime filter toggle** — respects `config.strategy.use_regime_filter` for benchmarking OFF vs ON; default `True` (unchanged unless toggled)

### `config.py`
- **infra** — added `StrategyConfig.use_regime_filter: bool = True`

---

## Data / Pipeline Plumbing

### `common/utils.py` + `common/__init__.py`
- **bug: Shared market-hours filter** — new `filter_market_hours(df)` (Polars): weekday + 09:15–15:30; exported

### `backtest/backtest.py`
- **bug: Import-time path freeze** — `load_regimes` default arg evaluated at import time, frozen to first run_date, broke multi-date runs; now resolved at call time

### `common/plots.py`
- **bug: Module-level directory freeze** — `plots_dir` evaluated at import time, wrote to wrong date's folder on multi-date runs; moved to call-time resolution

### `main.py`
- **infra: CLI expansion** — added argparse: `--run-date [YYYY-MM-DD]` (override single), `--run-dates [D1 D2...]` (sweep), `--workers N` (parallel); default unchanged (single run to stdout); multi-date runs spawn isolated subprocesses, each writes to `results/logs/run_<date>.log`

### `setup.sh`
- **bug: Case-sensitive check** — folder check `!= "capstone"` failed on `Capstone`; now case-insensitive

### Logging cleanup
- **chore: Log string arrows** — replaced `→` with `->` in log strings (cross-platform compatibility)

---

## Not Changed (Deliberately, To Stay in Scope)
- Strategy entry/exit rules, z thresholds, cost model, capital, risk, cooldown, min-hold — untouched
- Pair-selection cointegration/ADF tests — untouched (only selection window moved; half-life added as gate)
- No new strategy variants beyond regime on/off toggle for benchmarking

---

## Generated Files
- `RESULTS.md` — empirical results
- `CODE_CHANGES.md` — this file
- `STRATEGY_REVIEW.md` — diagnosis and fixes
- `reproduce_results.py` — regenerate backtest across Jan 2022
- `compute_stats.py` — aggregate trades to significance statistics
