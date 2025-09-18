import datetime

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
            start_dt = pd.to_datetime(f"{trade_date.strftime('%Y-%m-%d')} {config.strategy.start_time}")
            end_dt = pd.to_datetime(f"{trade_date.strftime('%Y-%m-%d')} {config.strategy.end_time}")

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
        results_dir = config.data.output_dir / "backtest"
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


if __name__ == "__main__":
    main()
