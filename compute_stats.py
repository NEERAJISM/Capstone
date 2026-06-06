"""compute_stats.py — significance stats for RESULTS.md.

Reads the trades.csv files produced by reproduce_results.py
(results/backtest/2022-01-*/) and computes win rate, per-trade Sharpe, and
day-clustered significance (daily Sharpe, t-stat, bootstrap CI). No synthetic data.

Run after reproduce_results.py:  python compute_stats.py"""
import glob
import numpy as np
import polars as pl

DATES = ["2022-01-10", "2022-01-13", "2022-01-17", "2022-01-20",
         "2022-01-24", "2022-01-27", "2022-01-31"]

rows = []
for d in DATES:
    for f in glob.glob(f"results/backtest/{d}/*/trades.csv"):
        try:
            t = pl.read_csv(f)
        except Exception:
            continue
        if t.height == 0 or "net_pnl" not in t.columns:
            continue
        pair = f.split("/")[-2].split("\\")[-1]
        for r in t.iter_rows(named=True):
            rows.append({"date": d, "pair": pair,
                         "net_pnl": float(r["net_pnl"]),
                         "gross_pnl": float(r.get("gross_pnl", float("nan"))),
                         "costs": float(r.get("costs", float("nan")))})

if not rows:
    print("no trades found")
    raise SystemExit

df = pl.DataFrame(rows)
pnl = df["net_pnl"].to_numpy()
n = len(pnl)
wins = (pnl > 0).sum()

print(f"=== REGIME_OFF (pure Kalman z-MR) — Jan 2022 sweep, 7 sessions ===")
print(f"trades            : {n}")
print(f"total net P&L     : {pnl.sum():,.2f}")
print(f"total gross P&L   : {df['gross_pnl'].sum():,.2f}")
print(f"total costs       : {df['costs'].sum():,.2f}")
print(f"win rate          : {wins}/{n} = {100*wins/n:.1f}%")
print(f"mean net / trade  : {pnl.mean():,.2f}")
print(f"std  net / trade  : {pnl.std(ddof=1):,.2f}")
print(f"per-trade Sharpe  : {pnl.mean()/pnl.std(ddof=1):.3f}")

# --- naive per-trade significance (INFLATED: trades not independent) ---
t_stat = pnl.mean() / (pnl.std(ddof=1) / np.sqrt(n))
print(f"\n[naive per-trade, treats {n} trades as iid -- OVERSTATES]")
print(f"  t-stat (mean>0) : {t_stat:.3f}")

# --- honest: cluster by trading day (effective n = #days) ---
daily = df.group_by("date").agg(pl.col("net_pnl").sum().alias("d")).sort("date")
dvals = np.array([r["d"] for r in daily.iter_rows(named=True)])
nd = len(dvals)
print(f"\n[day-clustered, n_days={nd} -- HONEST: trades within a day & shared-leg pairs correlate]")
print("Daily net P&L:")
for r in daily.iter_rows(named=True):
    print(f"  {r['date']}: {r['d']:,.2f}")
if nd > 1 and dvals.std(ddof=1) > 0:
    d_t = dvals.mean() / (dvals.std(ddof=1) / np.sqrt(nd))
    d_sharpe = dvals.mean() / dvals.std(ddof=1)
    print(f"  daily mean P&L  : {dvals.mean():,.2f}")
    print(f"  daily Sharpe    : {d_sharpe:.3f}")
    print(f"  daily t-stat    : {d_t:.3f}  (df={nd-1})")
    # bootstrap on daily totals
    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(dvals, nd, replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  bootstrap daily-mean 95% CI: [{lo:,.2f}, {hi:,.2f}]")
    print(f"  bootstrap P(daily-mean<=0) : {(boot <= 0).mean():.4f}")
    print(f"  NOTE: n_days={nd} is very small; treat significance as indicative only.")
