# Empirical Results — Intraday Mean-Reversion Pair Trading (Indian Equities)

**Core thesis of the paper: intraday mean reversion on cointegrated pairs.** Regime detection is
an *optional add-on*, reported separately in §5.

**Policy: every number below is read from a backtest produced by the current code. No simulated,
illustrative, or hand-entered performance figures. Reproduction scripts: `reproduce_results.py`
(runs the sweep), `compute_stats.py` (aggregates trades → statistics).**

---

## 0. Data availability (hard constraint)

`data/` contains **only two months**: `Cash Data December 2021` and `Cash Data January 2022`
(3,848 CSVs each). Therefore:
- A 3–6 month backtest is **not possible** with data on disk (full set is the README Dropbox link).
- The prior `2022-05-31` artifacts are **unreproducible** (that month absent).
- Runnable trade window = **January 2022** (lookback ≤ ~15 days lands in Dec 2021–Jan 2022).

---

## 1. Experimental setup (what was actually run)

| Item | Value |
|------|-------|
| Trade dates | 2022-01-10, -13, -17, -20, -24, -27, -31 (7 sessions) |
| Universe | fixed 13 liquid-by-volume names¹ |
| Pair selection | cointegration (Engle–Granger) + ADF + **intraday half-life** filter, KMeans clustering |
| **Selection window** | **lookback 15 calendar days, ending `run_date − 1`** (strictly before the traded session — no look-ahead) |
| Hedge ratio | time-varying **Kalman** filter (causal) |
| Signal | z-score of Kalman spread; entry `\|z\|>2.5`, exit `\|z\|<0.2` |
| Costs | round-trip 0.31% of turnover (STT 0.025% + slippage 0.1% + brokerage 0.03%, both legs) |
| Regime filter | **OFF** for the core result (it is the §5 add-on) |

¹ The market-hours fix raised real-bar volatility so the `volatility_threshold=0.0005` basic
filter now passes ~500 tickers, making full-universe pair selection O(n²)-intractable. The run
fixes the universe to the 13 names the pre-fix filter selected, **purely for tractability**.
This is flagged as a bug to fix (recalibrate the threshold), not a result-shaping choice.

---

## 2. Core Result — Intraday Mean Reversion (Out-of-Sample Selection)

Regime filter OFF, selection window ends `run_date − 1`.

| Metric | Value |
|--------|------:|
| Trades | **11** |
| Gross P&L | ₹40,698.15 |
| Costs | −₹13,732.97 |
| **Net P&L** | **+₹26,965.18** |
| Win rate | **81.8%** (9/11) |
| Mean net / trade | ₹2,451 |
| Per-trade Sharpe | 1.068 |

**Daily P&L (all 5 trading days with trades are positive):**

| Date | Net P&L |
|------|--------:|
| 2022-01-10 | 8,361.33 |
| 2022-01-13 | 8,641.98 |
| 2022-01-17 | 3,441.93 |
| 2022-01-20 | 3,006.97 |
| 2022-01-27 | 3,512.96 |

---

## 3. The look-ahead test (methodological rigor)

Pair selection originally used `end_day = run_date`, leaking the traded session into the
cointegration / ADF / beta / half-life computation. Removing it (`end_day = run_date − 1`):

| | Biased (`end_day=run_date`) | Corrected (OOS) |
|--|------:|------:|
| Trades | 27 | 11 |
| Net P&L | +₹66,382 | +₹26,965 |
| Win rate | 77.8% | 81.8% |

Look-ahead inflated net P&L ~2.5×, but the edge **survives out-of-sample** — all five trading
days remain positive. The §2 table is the corrected, honest version. (The bias fix is committed
in `MeanReversionAnalyzer.__init__`.)

---

## 4. Statistical significance

| Basis | Value | Interpretation |
|-------|------:|----------------|
| Per-trade t-stat (n=11) | 3.54 | **Overstates** — trades within a day and shared-leg pairs are correlated |
| **Day-clustered** (n=5 days) | **t = 4.24** (df=4) | Honest unit; daily P&L i.i.d. across sessions |
| Daily Sharpe (unannualized) | **1.90** | n=5 |
| Bootstrap daily-mean 95% CI | [₹3,282, ₹7,515] | 10,000 resamples |
| Bootstrap P(daily-mean ≤ 0) | **0.0000** | |

**Caveat in bold: n = 5 trading days is a very small sample.** The significance is *indicative*,
not conclusive. A defensible journal claim needs the full multi-month dataset (§6).

---

## 5. Regime filter (the optional add-on) — does not help as implemented

Same sweep with the regime filter ON (enter only when **both** legs are `Sideways`):

| Arm | Trades | Net P&L |
|-----|-------:|--------:|
| Mean reversion only (core) | 11 | +₹26,965 |
| **+ regime filter (both-Sideways)** | **0** | **₹0** |

The regime gate blocks **100%** of entries (funnel diagnostic: `rej_regime` = every eligible
bar). Reason is structural, not data-dependent: cointegrated pairs **co-move**, so the two legs
are almost never simultaneously `Sideways`. The regime model itself is sound (HMM converges on
market-hours data; clean ~70/30 Sideways/Trending split) — but the **both-legs AND-gate is the
wrong way to apply it.** The theoretically correct object is the regime of the **spread**, not
each leg. As currently implemented the regime filter subtracts value; it is honestly an open
item, not a demonstrated enhancement.

---

## 6. Honest limitations

1. **Sample size.** 5 trading days, 11 trades, one month. Strong daily consistency, but too few
   sessions for a conclusive Sharpe/t-stat. Needs the full dataset.
2. **Position sizing / market impact.** ₹200,000 notional per leg ⇒ 9k–106k shares of ₹1–3
   penny stocks — far beyond realistic intraday volume. **The rupee P&L is an upper bound;
   market impact would erode it.** Report **Sharpe** (scale-free) as the headline, not ₹.
   Realistic sizing (ADV participation cap) + an impact term are required before the ₹ figure
   means anything.
3. **Penny-stock universe.** 13 distressed sub-₹10 names (coarse ticks, wide spreads). Costs
   already eat 34% of gross. A liquid mid/large-cap universe is the right robustness test.
4. **In-sample strategy parameters.** `z_entry`, `z_exit`, costs are fixed, not tuned/validated
   on a hold-out.
5. **Selection tractability hack.** Fixed universe (§1 note) until `volatility_threshold` is
   recalibrated for real bars.

---

## 7. Verdict and path to a publishable result

**The core claim — intraday mean reversion on cointegrated Indian-equity pairs is profitable
out-of-sample — holds on the available data** (11 trades, +₹26,965, 82% win, daily Sharpe 1.90,
day-clustered t = 4.24, all 5 days positive), subject to the §6 caveats (small n, unrealistic
sizing). It is a real, defensible *preliminary* result, not yet a journal-grade one.

To reach journal grade, in order:
1. **Get the full dataset** (README Dropbox) → run all sessions across 3–6 months. This alone
   converts n=5 days into a real sample and enables annualized Sharpe + Diebold–Mariano.
2. **Realistic sizing + market impact** → make the ₹ P&L meaningful (§6.2).
3. **Liquid universe** + recalibrated `volatility_threshold` (§6.3) → remove the penny-stock and
   tractability confounds.
4. **Reformulate the regime add-on** to gate on the **spread** regime (§5); only then test
   whether it adds value over the mean-reversion benchmark.
5. Out-of-sample parameter validation; report Sharpe with Lo (2002) SEs + stationary-bootstrap CI.

---

## 8. Reproducibility

```
python reproduce_results.py   # runs the Jan-2022 sweep (fixed universe, OOS selection), regime OFF
python compute_stats.py       # aggregates results/backtest/2022-01-*/ trades -> statistics in §2,§4
```
Key code state: `MeanReversionAnalyzer.end_day = run_date − 1` (no look-ahead);
`StrategyConfig.use_regime_filter` toggles §5; entry-funnel diagnostics print per pair.

---

## References

1. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). *Pairs Trading: Performance of a
   Relative-Value Arbitrage Rule.* Review of Financial Studies, 19(3), 797–827.
2. Do, B., & Faff, R. (2010). *Does Simple Pairs Trading Still Work?* Financial Analysts
   Journal, 66(4), 83–95.
3. Krauss, C. (2017). *Statistical Arbitrage Pairs Trading Strategies: Review and Outlook.*
   Journal of Economic Surveys, 31(2), 513–545.
4. Elliott, R. J., van der Hoek, J., & Malcolm, W. P. (2005). *Pairs Trading.* Quantitative
   Finance, 5(3), 271–276.
5. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis.* Wiley.
6. Avellaneda, M., & Lee, J.-H. (2010). *Statistical Arbitrage in the US Equities Market.*
   Quantitative Finance, 10(7), 761–782.
7. Engle, R. F., & Granger, C. W. J. (1987). *Co-integration and Error Correction.*
   Econometrica, 55(2), 251–276.
8. Hamilton, J. D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series
   and the Business Cycle.* Econometrica, 57(2), 357–384.
9. Diebold, F. X., & Mariano, R. S. (1995). *Comparing Predictive Accuracy.* Journal of Business
   & Economic Statistics, 13(3), 253–263.
10. Politis, D. N., & Romano, J. P. (1994). *The Stationary Bootstrap.* JASA, 89(428), 1303–1313.
11. Lo, A. W. (2002). *The Statistics of Sharpe Ratios.* Financial Analysts Journal, 58(4), 36–52.
12. Bowen, D., Hutchinson, M. C., & O'Sullivan, N. (2010). *High-Frequency Equity Pairs Trading:
    Transaction Costs, Speed of Execution, and Patterns in Returns.* Journal of Trading, 5(3), 31–38.

*Citations are standard, verifiable works; confirm exact pages against the publisher before
inclusion in the manuscript.*
