import argparse
import datetime
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from intraday_strategy import get_pair_json_path_cached, save_regime_detected
import json
import pandas as pd
from backtest import run_pair
import os
from common import StockDataLoader, get_logger
from config import config

logger = get_logger(__name__)


def main():
    # Generate pair selection results
    pair_json_path = get_pair_json_path_cached()

    with open(pair_json_path, "r") as f:
        clusters = json.load(f)

    all_results = []
    for cid, cdata in clusters.items():
        pairs = cdata.get("pairs", [])
        logger.info(pairs)
        if not pairs:
            continue
        logger.info(f"\n=== Running {cid} ===")
        for a_pair in pairs:
            stock_a, stock_b = a_pair["tickers"]

            save_regime_detected(stock_a)
            save_regime_detected(stock_b)

            trade_date = datetime.datetime.strptime(config.data.run_date, "%Y-%m-%d")

            # Build intraday start & end datetimes
            start_dt = pd.to_datetime(
                f"{trade_date.strftime('%Y-%m-%d')} {config.strategy.start_time}"
            )
            end_dt = pd.to_datetime(
                f"{trade_date.strftime('%Y-%m-%d')} {config.strategy.end_time}"
            )

            loader = StockDataLoader(
                base_dir=config.data.data_dir,
                start=start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                end=end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                tickers=[stock_a, stock_b],
                select_columns=["close"],
                impute=True,
            )

            result = run_pair(stock_a, stock_b, loader)
            if result:
                all_results.append(result)

    # Save leaderboard
    if all_results:
        leaderboard = pd.DataFrame(all_results)
        leaderboard.sort_values("net_pnl", ascending=False, inplace=True)

        # Save results to output directory
        results_dir = config.data.output_dir / "backtest" / str(config.data.run_date)
        os.makedirs(results_dir, exist_ok=True)

        leaderboard_path = results_dir / "pairs_leaderboard.csv"
        leaderboard.to_csv(leaderboard_path, index=False)

        logger.info(f"\n=== Leaderboard saved: {leaderboard_path} ===")

        # Save detailed results
        summary = {
            "run_timestamp": datetime.datetime.now().isoformat(),
            "config": {
                "run_date": config.data.run_date,
                "lookback_days": config.pair_selection.lookback_days,
                "n_pairs_tested": len(all_results),
                "n_clusters": len(clusters),
            },
            "metrics": {
                "total_trades": leaderboard["trades"].sum().item(),
                "total_pnl": leaderboard["net_pnl"].sum().item(),
                "avg_win_rate": leaderboard["win_rate"].mean().item(),
                "avg_daily_sharpe": leaderboard["daily_sharpe"].mean().item(),
            },
        }

        with open(results_dir / "backtest_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    else:
        logger.warning("No valid results generated from any pairs.")


def _run_one(run_date: str, log_dir: Path) -> tuple[str, int, str]:
    """Launch a single-date pipeline in its own process, redirecting its output
    to a per-date log file. Subprocess isolation gives each run a clean config
    (run_date) and avoids joblib state bleeding between dates."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{run_date}.log"
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--run-date", run_date],
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
    return run_date, proc.returncode, str(log_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pair-trading backtest pipeline (pair selection -> regime -> backtest)."
    )
    p.add_argument(
        "--run-date",
        help="Single run date YYYY-MM-DD; overrides config. Default: config value.",
    )
    p.add_argument(
        "--run-dates",
        nargs="+",
        metavar="YYYY-MM-DD",
        help="Multiple run dates executed as parallel subprocesses.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel run-date workers for --run-dates (default: 1 = sequential).",
    )
    p.add_argument(
        "--log-dir",
        default="results/logs",
        help="Directory for per-run logs when using --run-dates (default: results/logs).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.run_dates:
        # Fan-out: one subprocess per date, up to --workers concurrent.
        log_dir = Path(args.log_dir)
        logger.info(
            "Launching %d run(s) across %d worker(s); logs -> %s",
            len(args.run_dates),
            args.workers,
            log_dir,
        )
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_run_one, d, log_dir): d for d in args.run_dates
            }
            for fut in as_completed(futures):
                date, code, path = fut.result()
                status = "OK" if code == 0 else f"FAILED (exit {code})"
                logger.info("[%s] %s -> %s", date, status, path)
    else:
        # Default single run (current behaviour). --run-date overrides config.
        if args.run_date:
            config.data.run_date = args.run_date
        main()
