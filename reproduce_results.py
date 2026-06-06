"""reproduce_results.py — regenerate the RESULTS.md backtest.

Runs the pipeline (pair selection -> regime -> backtest) across the Jan-2022 trade
days, regime filter OFF (core pure-Kalman z-score mean reversion), fixed tractable
universe, lookback 15, out-of-sample selection. Writes results/backtest/2022-01-*/.

Run from the repo root:  python reproduce_results.py
Then analyze with:        python compute_stats.py

Data on disk = Dec 2021 + Jan 2022 only. All numbers are real (no synthetic data)."""
import csv
from pathlib import Path
from config import config

UNIVERSE = [
    "JPPOWER", "RCOM", "UNITECH", "FCONSUMER", "RHFL", "VIKASPROP", "IVC",
    "ROLLT", "ORIENTALTL", "KRIDHANINF", "ROLTA", "RNAVAL", "PROZONINTU",
]
DATES = [
    "2022-01-10", "2022-01-13", "2022-01-17", "2022-01-20",
    "2022-01-24", "2022-01-27", "2022-01-31",
]
config.data.tickers_universe = UNIVERSE
config.pair_selection.lookback_days = 15

import main as m


def read_leaderboard(d):
    lb = Path(config.data.output_dir) / "backtest" / d / "pairs_leaderboard.csv"
    trades = 0
    pnl = 0.0
    rows = 0
    if lb.exists():
        with open(lb) as fh:
            for row in csv.DictReader(fh):
                trades += int(row["trades"])
                pnl += float(row["net_pnl"])
                rows += 1
    return rows, trades, pnl


results = {False: []}
for use_regime in (False,):  # core = pure mean reversion (regime is an extra)
    config.strategy.use_regime_filter = use_regime
    arm = "REGIME_ON" if use_regime else "REGIME_OFF"
    for d in DATES:
        config.data.run_date = d
        psdir = Path(config.data.output_dir) / "pair_selection" / d
        psdir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== ARM {arm} | RUN {d} =====")
        try:
            m.main()
        except Exception as e:
            print(f"[{arm}|{d}] FAILED: {e}")
            results[use_regime].append((d, None, None, None))
            continue
        rows, trades, pnl = read_leaderboard(d)
        results[use_regime].append((d, rows, trades, pnl))

print("\n\n================ SWEEP SUMMARY ================")
for use_regime in (False,):
    arm = "REGIME_ON " if use_regime else "REGIME_OFF"
    tot_t = 0
    tot_p = 0.0
    print(f"\n--- {arm} ---")
    print(f"{'date':12}{'pairs':>7}{'trades':>8}{'net_pnl':>14}")
    for row in results[use_regime]:
        d, rows, t, p = row
        if rows is None:
            print(f"{d:12}{'ERR':>7}")
            continue
        print(f"{d:12}{rows:>7}{t:>8}{p:>14.4f}")
        tot_t += t
        tot_p += p
    print(f"{'TOTAL':12}{'':>7}{tot_t:>8}{tot_p:>14.4f}")
